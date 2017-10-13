import numpy as np
import tensorflow as tf
import gym

class PolicyGradientAgent:
    def __init__(self, env:gym.Env, n_actions, n_features, learning_rate=0.01,
                 reward_decay=0.95):
        self.env = env
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.states, self.actions, self.rewards = [],[],[] # for one episode
        self.model_set = False
        # set the input sizes
        with tf.name_scope('input'):
            self.tf_states = tf.placeholder(tf.float32, [None, self.n_features], name="observations")
            self.tf_acts = tf.placeholder(tf.int32, [None, ], name="actions_num")
            self.tf_vt = tf.placeholder(tf.float32, [None, ], name="actions_value")

    def get_input(self):
        return self.tf_states
    
    def set_model(self, model):
        # make sure the output has the correct size
        assert model.get_shape()[1] == self.n_actions, 'the output must have the same size as n_actions'
            
        self.all_act_prob = tf.nn.softmax(model, name='act_prob')
        
        with tf.name_scope('loss'):
            neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model,labels=self.tf_acts)
            loss = tf.reduce_mean(neg_log_prob*self.tf_vt)
        
        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)
            
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.model_set = True
    
    def choose_action(self, observation):
        prob_weights = self.sess.run(self.all_act_prob, feed_dict={self.tf_states:observation[np.newaxis,:]})
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())
        return action
    
    def greedy(self, observation):
        prob_weights = self.sess.run(self.all_act_prob, feed_dict={self.tf_states: observation[np.newaxis, :]})
        action = np.argmax(prob_weights.ravel())
        return action
    
    def remember(self, s, a, r):
        self.states.append(s)
        self.actions.append(a)
        self.rewards.append(r)
    
    def _discount_and_norm_rewards(self, normalize=True):
        discounted_ep_rs =np.zeros_like(self.rewards)
        running_add = 0
        for t in reversed(range(0, len(self.rewards))):
            running_add = running_add * self.gamma + self.rewards[t]
            discounted_ep_rs[t] = running_add
        if normalize:
            discounted_ep_rs -= np.mean(discounted_ep_rs)
            discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs
    
    def _one_episode(self):
        discounted_ep_rs_norm = self._discount_and_norm_rewards()
        
        self.sess.run(self.train_op, feed_dict={
            self.tf_states: np.vstack(self.states),
            self.tf_acts: np.array(self.actions),
            self.tf_vt: discounted_ep_rs_norm,
        })
        
        self.states, self.actions, self.rewards = [],[],[]
        
    def learn(self, n_episodes, visualize=False, verbose=0):
        assert self.model_set, 'the model has not been set'
        
        rewards = []
        for i_episode in range(n_episodes):
            state = self.env.reset()
            r = 0
            while True:
                if visualize:
                    self.env.render()
                # sample an action
                action = self.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)
                r += reward
                # save the info
                self.remember(state, action, reward)
                if done:
                    if verbose > 0:
                        print("episode:", i_episode+1, "rewards:", r)
                    # learn once
                    self._one_episode()
                    rewards += [r]
                    break
        
                state = next_state
        print("finished learning!")
        return rewards
    
    def test(self, n_episodes, visualize=True, verbose=1):
        assert self.model_set, 'the model has not been set'
        
        rewards = []
        for i_episode in range(n_episodes):
            state = self.env.reset()
            r = 0
            while True:
                if visualize:
                    self.env.render()
                # sample an action
                action = self.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)
                r += reward
                # save the info
                self.remember(state, action, reward)
                if done:
                    if verbose > 0:
                        print("episode:", i_episode+1, "rewards:", r)
                    # learn once
                    self._one_episode()
                    rewards += [r]
                    break
        
                state = next_state
        print("finished testing!")
        return rewards