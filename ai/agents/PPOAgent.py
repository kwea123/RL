from ai.structure.SumTree import *
import numpy as np
import tensorflow as tf
import gym
import sklearn.preprocessing
import sklearn.pipeline
from sklearn.kernel_approximation import RBFSampler

class PPOAgent:
    def __init__(self, env:gym.Env, n_actions, n_features, action_low, action_high, featurize=False, reward_decay=0.95,
                 actor_learning_rate=0.01, critic_learning_rate=0.01, learning_rate_decay=0.95,
                 tau=1.0):
        self.env = env
        self.state_size = n_features
        self.action_size = n_actions
        self.action_low = action_low
        self.action_high = action_high
        self.gamma = reward_decay   # discount rate
        self.actor_model_set = False
        self.critic_model_set = False
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate # often larger than actor_learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.tau = tau # soft update
        self.batch_size = 32
        self.memory = [] # store (s, a, r)
        self.featurize = featurize
        if featurize:
            self._init_featurizer()
        self._construct_nets()
        
    def _construct_nets(self):
        self.sess = tf.Session()
        
        # inputs
        self.S = tf.placeholder(tf.float32, [None, self.state_size], 'state')
        self.A = tf.placeholder(tf.float32, [None, self.action_size], 'action')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        with tf.variable_scope('Actor'):
            norm_dist = self._build_a(self.S, scope='eval', trainable=True)
            norm_dist_ = self._build_a(self.S, scope='target', trainable=False)
        with tf.variable_scope('Critic'):
            self._build_c(self.S)
        
        # networks parameters
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.c_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic')
        
        # target net update
        self.soft_replace = [tf.assign(ta, (1 - self.tau) * ta + self.tau * ea)
                             for ta, ea in zip(self.at_params, self.ae_params)]
        
        # sample action
        self.sample_action = norm_dist.sample(1)[0]
        
        # losses
        closs = tf.reduce_mean(tf.square(self.advantage))
        self.ctrain = tf.train.AdamOptimizer(self.critic_learning_rate).minimize(closs, var_list=self.c_params)
        ratio = norm_dist.prob(self.A) / (norm_dist_.prob(self.A)+1e-10)
        surr = ratio * self.advantage
        self.aloss = -tf.reduce_mean(tf.minimum(surr,
                                                tf.clip_by_value(ratio, 0.8, 1.2)*self.advantage))
        self.atrain = tf.train.AdamOptimizer(self.actor_learning_rate).minimize(self.aloss, var_list=self.ae_params)
        
        # initialise
        self.sess.run(tf.global_variables_initializer())
    
    def _build_a(self, s, scope, trainable): # policy
        with tf.variable_scope(scope):
            l1 = tf.layers.dense(s, 100, tf.nn.relu, trainable=trainable)
            l2 = tf.layers.dense(l1, 50, tf.nn.relu, trainable=trainable)
            mu = max(np.abs(self.action_low), np.abs(self.action_high)) * tf.layers.dense(l2, self.action_size, tf.nn.tanh, trainable=trainable)
            self.sigma = tf.layers.dense(l2, self.action_size, tf.nn.softplus, trainable=trainable)
            norm_dist = tf.distributions.Normal(loc=mu, scale=self.sigma)
            self.actor_model_set = True
            return norm_dist
    
    def _build_c(self, s): # state value
        l1 = tf.layers.dense(s, 300, activation=tf.nn.relu, name='l1')
        v = tf.layers.dense(l1, 1)
        self.discounted_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
        self.advantage = self.discounted_r - v
        self.critic_model_set = True
    
    def choose_action(self, state): # normal distribution
        assert self.actor_model_set, 'actor model not set!'
        action = self.sess.run(self.sample_action, {self.S: state})[0]
        return np.clip(action, self.action_low, self.action_high)
    
    def remember(self, state, action, reward, next_state):
        self.memory += [[state[0], action, reward, next_state[0]]]
    
    def replay(self):
        assert self.actor_model_set, 'model not set!'
        assert self.critic_model_set, 'critic model not set!'
        memory = np.vstack(self.memory)
        states = np.vstack(memory[:,0])
        actions = np.vstack(memory[:,1])
        rewards = memory[:,2]
        last_next_state = memory[:,3][-1]
        
        discounted_ep_rs = np.zeros_like(rewards)
        running_add = self.sess.run(self.v, {self.tfs: [last_next_state]})[0]
        for t in reversed(range(0, len(memory))):
            running_add = running_add * self.gamma + rewards[t]
            discounted_ep_rs[t] = running_add
        
        [self.sess.run(self.atrain, {self.S: states, self.A: actions, self.discounted_r: discounted_ep_rs[:,np.newaxis]}) for _ in range(10)]
        [self.sess.run(self.ctrain, {self.S: states, self.discounted_r: discounted_ep_rs[:,np.newaxis]}) for _ in range(10)]
        self.sess.run(self.soft_replace) # update the weights
        
        self.actor_learning_rate *= self.learning_rate_decay
        self.critic_learning_rate *= self.learning_rate_decay
        self.memory = [] # empty the memory
        
    def _init_featurizer(self):
        # Feature preprocessing: Normalize to zero mean and unit variance
        # We use a few samples from the observation space to do this
        observation_examples = np.array([self.env.observation_space.sample() for _ in range(10000)])
        self.scaler = sklearn.preprocessing.StandardScaler()
        self.scaler.fit(observation_examples)
        
        # Used to converte a state to a featurized representation.
        # We use RBF kernels with different variances to cover different parts of the space
        self.featurizer = sklearn.pipeline.FeatureUnion([
                ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
                ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
                ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
                ("rbf4", RBFSampler(gamma=0.5, n_components=100))
                ])
        self.featurizer.fit(self.scaler.transform(observation_examples))
        self.state_size = 400
        
    def featurize_state(self, state):
        scaled = self.scaler.transform(state)
        featurized = self.featurizer.transform(scaled)
        return featurized
        
    def learn(self, n_episodes, visualize=False, verbose=0):
        assert self.actor_model_set, 'actor model not set!'
        assert self.critic_model_set, 'critic model not set!'
        rewards = []
        for i_episode in range(n_episodes):
            state = self.env.reset()
            state = state[np.newaxis, :]
            if self.featurize:
                state = self.featurize_state(state)
            r = 0
            while True:
                if visualize:
                    self.env.render()
                action = self.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)
                r += reward
                next_state = next_state[np.newaxis, :]
                if self.featurize:
                    next_state = self.featurize_state(next_state)
                self.remember(state, action, reward, next_state)
                # learn once
                if len(self.memory) == self.batch_size or done:
                    self.replay()
                state = next_state
                if done:
                    if verbose > 0:
                        print("episode:", i_episode+1, "rewards:", r)
                    rewards += [r]
                    break
        print("finished learning!")
        return rewards
    
    def test(self, n_episodes, visualize=True, verbose=1):
        assert self.actor_model_set, 'actor model not set!'
        assert self.critic_model_set, 'critic model not set!'
        rewards = []
        for i_episode in range(n_episodes):
            state = self.env.reset()
            state = state[np.newaxis, :]
            if self.featurize:
                state = self.featurize_state(state)
            r = 0
            while True:
                if visualize:
                    self.env.render()
                action = self.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)
                r += reward
                next_state = next_state[np.newaxis, :]
                if self.featurize:
                    next_state = self.featurize_state(next_state)
                state = next_state
                if done:
                    if verbose > 0:
                        print("episode:", i_episode+1, "rewards:", r)
                    rewards += [r]
                    break
        print("finished testing!")
        return rewards