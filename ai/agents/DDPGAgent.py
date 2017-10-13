from ai.structure.SumTree import *
import numpy as np
import tensorflow as tf
import gym
import sklearn.preprocessing
import sklearn.pipeline
from sklearn.kernel_approximation import RBFSampler
from tensorflow.contrib.learn.python.learn import trainable

class DDPGAgent:
    def __init__(self, env:gym.Env, n_actions, n_features, action_low, action_high, featurize=False, reward_decay=0.95,
                 actor_learning_rate=0.01, critic_learning_rate=0.01, learning_rate_decay=0.95,
                 memory_size=10000, priority_alpha=0.6, tau=0.9, variance=3):
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
        self.priority_alpha = priority_alpha
        self.tau = tau # soft update
        self.batch_size = 32
        self.memory = Memory(capacity=memory_size, a=priority_alpha)
        self.variance = variance # exploration
        self.memory_size = memory_size
        self.featurize = featurize
        if featurize:
            self._init_featurizer()
        self._construct_nets()
        
    def _construct_nets(self):
        self.sess = tf.Session()
        
        self.S = tf.placeholder(tf.float32, [None, self.state_size], 'state')
        self.S_ = tf.placeholder(tf.float32, [None, self.state_size], 'next_state')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        with tf.variable_scope('Actor'):
            self.a = self._build_a(self.S, scope='eval', trainable=True)
            self.a_ = self._build_a(self.S_, scope='target', trainable=False)
        with tf.variable_scope('Critic'):
            # assign self.a = a in memory when calculating q for td_error,
            # otherwise the self.a is from Actor when updating Actor
            q = self._build_c(self.S, self.a, scope='eval', trainable=True)
            q_ = self._build_c(self.S_, self.a_, scope='target', trainable=False)
        
        # networks parameters
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        # target net replacement
        self.soft_replace = [[tf.assign(ta, (1 - self.tau) * ta + self.tau * ea), tf.assign(tc, (1 - self.tau) * tc + self.tau * ec)]
                             for ta, ea, tc, ec in zip(self.at_params, self.ae_params, self.ct_params, self.ce_params)]

        q_target = self.R + self.gamma * q_
        # in the feed_dic for the td_error, the self.a should change to actions in memory
        self.td_error_element_wise = tf.squared_difference(q_target, q)
        self.td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
        self.ctrain = tf.train.AdamOptimizer(self.critic_learning_rate).minimize(self.td_error, var_list=self.ce_params)
           
        a_loss = - tf.reduce_mean(q)    # maximize the q
        self.atrain = tf.train.AdamOptimizer(self.actor_learning_rate).minimize(a_loss, var_list=self.ae_params)
        
        self.sess.run(tf.global_variables_initializer())
    
    def _build_a(self, s, scope, trainable): # policy
        with tf.variable_scope(scope):
            net = tf.layers.dense(s, 200, activation=tf.nn.relu, name='l1', trainable=trainable)
            net = tf.layers.dense(net, 100, activation=tf.nn.relu, name='l2', trainable=trainable)
            net = tf.layers.dense(net, 30, activation=tf.nn.relu, name='l3', trainable=trainable)
            a = tf.layers.dense(net, self.action_size, activation=tf.nn.tanh, name='a', trainable=trainable)
            self.actor_model_set = True
            return a * (self.action_high-self.action_low)/2 + (self.action_high+self.action_low)/2
    
    def _build_c(self, s, a, scope, trainable): # advantage value
        with tf.variable_scope(scope):
            n_l1 = 30
            w1_s = tf.get_variable('w1_s', [self.state_size, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.action_size, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            net = tf.layers.dense(net, 200, activation=tf.nn.relu, trainable=trainable)
            net = tf.layers.dense(net, 10, activation=tf.nn.relu, trainable=trainable)
            self.critic_model_set = True
            return tf.layers.dense(net, 1, trainable=trainable)  # Q(s,a)
    
    def remember(self, state, action, reward, next_state, done):
        if self.priority_alpha > 0: # prioritised
            self.memory.add((state, action, reward, next_state, done), 
                            self.error(state, action, reward, next_state))
        else: # non prioritised, every memory has priority 1
            self.memory.add((state, action, reward, next_state, done), 1)
            
    def error(self, state, action, reward, next_state):
        return self.sess.run(self.td_error, {self.S: state, self.a: [action], 
                                             self.R: [[reward]], self.S_: next_state})
        
    def choose_action(self, state, variance, low, high): # normal distribution
        assert self.actor_model_set, 'actor model not set!'
        action = self.sess.run(self.a, {self.S: state})[0]
        return np.clip(np.random.normal(action, variance), low, high)
    
    def replay(self, batch_size):
        assert self.actor_model_set, 'model not set!'
        assert self.critic_model_set, 'critic model not set!'
        minibatch = self.memory.sample(batch_size)
        idxs, states, actions, rewards, next_states = [], [], [], [], []
        for idx, (state, action, reward, next_state, done) in minibatch:
            idxs+=[idx]
            states+=[state]
            actions+=[action]
            rewards+=[reward]
            next_states+=[next_state]
        
        self.sess.run(self.atrain, {self.S: np.vstack(states)})
        self.sess.run(self.ctrain, {self.S: np.vstack(states), self.a: np.vstack(actions),
                                    self.R: np.vstack(rewards), self.S_: np.vstack(next_states)})
        self.sess.run(self.soft_replace) # update the weights
        
        if self.priority_alpha > 0: # prioritised, update
            errors = self.sess.run(self.td_error_element_wise, {self.S: np.vstack(states), self.a: np.vstack(actions),
                                                                self.R: np.vstack(rewards), self.S_: np.vstack(next_states)})
            for i in range(len(idxs)):
                self.memory.update(idxs[i], errors[i])
        
        self.actor_learning_rate *= self.learning_rate_decay
        self.critic_learning_rate *= self.learning_rate_decay
        
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
            counter = 0
            while counter<200:
                counter += 1
                if visualize:
                    self.env.render()
                action = self.choose_action(state, self.variance, self.action_low, self.action_high)
                next_state, reward, done, _ = self.env.step(action)
                r += reward
                next_state = next_state[np.newaxis, :]
                if self.featurize:
                    next_state = self.featurize_state(next_state)
                self.remember(state, action, reward, next_state, done)
                # learn when memory is full, every BATCH steps
                if len(self.memory) == self.memory_size and (counter%self.batch_size==0 or done or counter==200):
                    self.variance *= 0.99995
#                     self.replay(self.batch_size)
                    [self.replay(self.batch_size) for _ in range(10)]
                state = next_state
                if done or counter==200:
                    if verbose > 0:
                        print("episode:", i_episode+1, "rewards:", r, "explore var: %.2f" % self.variance)
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
                action = self.choose_action(state, 0, self.action_low, self.action_high)
                next_state, reward, done, _ = self.env.step(action)
                r += reward
                next_state = next_state[np.newaxis, :]
                if self.featurize:
                    next_state = self.featurize_state(next_state)
                state = next_state
#                 if done:
#                     if verbose > 0:
#                         print("episode:", i_episode+1, "rewards:", r)
#                     rewards += [r]
#                     break
        print("finished testing!")
        return rewards