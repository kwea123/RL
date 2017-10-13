import numpy as np
from keras.optimizers import Adam
import gym

class PolicyGradientAgent2:
    def __init__(self, env:gym.Env, n_actions, n_features, learning_rate=0.01,
                 reward_decay=0.95, learning_rate_decay=0.9):
        self.env = env
        self.action_size = n_actions
        self.state_size = n_features
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.gamma = reward_decay
        self.states, self.actions, self.rewards = [],[],[] # for one episode
        self.model_set = False
    
    def set_model(self, model):
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(lr=self.learning_rate))
        self.model = model
        self.model_set = True
            
    def choose_action(self, state): # stochastic policy
        assert self.model_set, 'model not set!'
        act_values = self.model.predict(state)
        action = np.random.choice(range(self.action_size), p=act_values[0])
        return action
    
    def greedy(self, state): # greedy policy
        assert self.model_set, 'model not set!'
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])
    
    def remember(self, s, a, r):
        self.states.append(s)
        y = np.zeros([self.action_size])
        y[a] = 1
        self.actions.append(y)
        self.rewards.append(r)
    
    def _discount_and_norm_rewards(self, normalize=True):
        discounted_ep_rs = np.zeros_like(self.rewards)
        running_add = 0
        for t in reversed(range(0, len(self.rewards))):
            running_add = running_add * self.gamma + self.rewards[t]
            discounted_ep_rs[t] = running_add
        if normalize:
            discounted_ep_rs -= np.mean(discounted_ep_rs)
            discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs
    
    def _one_episode(self):
        self.model.fit(np.vstack(self.states), np.vstack(self.actions), epochs=1, verbose=0,
                       sample_weight=self._discount_and_norm_rewards())
        self.states, self.actions, self.rewards = [],[],[]
        self.learning_rate *= self.learning_rate_decay
        
    def learn(self, n_episodes, visualize=False, verbose=0):
        assert self.model_set, 'model not set!'
        
        rewards = []
        for i_episode in range(n_episodes):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            r = 0
            while True:
                if visualize:
                    self.env.render()
                # sample an action
                action = self.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.reshape(next_state, [1, self.state_size])
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
        assert self.model_set, 'model not set!'
        
        rewards = []
        for i_episode in range(n_episodes):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            r = 0
            while True:
                if visualize:
                    self.env.render()
                # sample an action (greedy)
                action = self.greedy(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.reshape(next_state, [1, self.state_size])
                r += reward
                if done:
                    if verbose > 0:
                        print("episode:", i_episode+1, "rewards:", r)
                    rewards += [r]
                    break
                state = next_state
        print("finished testing!")
        return rewards
    
    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)