import gym
import numpy as np
# from collections import deque
from ai.structure.SumTree import *
# from keras.models import Sequential
# from keras.layers import Dense
from keras.optimizers import Adam

class DQNAgent:
    def __init__(self, env:gym.Env, n_actions, n_features, learning_rate=0.01,
                 reward_decay=0.95, learning_rate_decay=0.9, batch_size=32, 
                 memory_size=200000, priority_alpha = 0.5):
        self.env = env
        self.state_size = n_features
        self.action_size = n_actions
#         self.memory = deque(maxlen=memory_size)
        self.memory = Memory(capacity=memory_size, a=priority_alpha)
        self.priority_alpha = priority_alpha
        self.gamma = reward_decay   # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.001 # min epsilon
        self.epsilon_decay = 0.995
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.batch_size = batch_size
        self.model_set = False

    def set_model(self, model):
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        self.model = model
        self.model_set = True

    def remember(self, state, action, reward, next_state, done):
        if self.priority_alpha > 0: # prioritised
            self.memory.add((state, action, reward, next_state, done), 
                            self.error(state, action, reward, next_state, done))
        else: # non prioritised, every memory has priority 1
            self.memory.add((state, action, reward, next_state, done), 1)
        
    def error(self, state, action, reward, next_state, done):
        target = reward # if done
        if not done:
            target = (reward + self.gamma *
                      np.max(self.model.predict(next_state)[0]))
        target_f = self.model.predict(state) # get the current(former) estimate
        return abs(target - target_f[0][action])

    def choose_action(self, state): # epsilon greedy policy
        assert self.model_set, 'model not set!'
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        assert self.model_set, 'model not set!'
        minibatch = self.memory.sample(batch_size)
        states, target_fs = [], []
        for idx, (state, action, reward, next_state, done) in minibatch:
            target = reward # if done
            if not done:
                target = (reward + self.gamma *
                          np.max(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state) # get the current(former) estimate
            target_f[0][action] = target # approach the true value
            states+=[state]
            target_fs+=[target_f]
            if self.priority_alpha > 0: # prioritised, update
                self.memory.update(idx, self.error(state, action, reward, next_state, done))
        self.model.fit(np.vstack(states), np.vstack(target_fs), epochs=1, verbose=0)
        self.epsilon *= self.epsilon_decay
        if self.epsilon < self.epsilon_min:
            self.epsilon = self.epsilon_min
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
                action = self.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)
                r += reward
                next_state = np.reshape(next_state, [1, self.state_size])
                self.remember(state, action, reward, next_state, done)
                # learn once
                if len(self.memory) > self.batch_size:
                    self.replay(self.batch_size)
                state = next_state
                if done:
                    if verbose > 0:
                        print("episode:", i_episode+1, "rewards:", r)
                    rewards += [r]
                    break
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
                action = self.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)
                r += reward
                next_state = np.reshape(next_state, [1, self.state_size])
                state = next_state
                if done:
                    if verbose > 0:
                        print("episode:", i_episode+1, "rewards:", r)
                    rewards += [r]
                    break
        print("finished testing!")
        return rewards

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)