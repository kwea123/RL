import numpy as np
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
import tensorflow as tf
import gym

class ActorCriticAgent:
    def __init__(self, env:gym.Env, n_actions, n_features, reward_decay=0.95, actor_learning_rate=0.01, 
                 critic_learning_rate=0.01, learning_rate_decay=0.9):
        self.env = env
        self.state_size = n_features
        self.action_size = n_actions
        self.gamma = reward_decay   # discount rate
        self.actor_model_set = False
        self.critic_model_set = False
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate # often larger than actor_learning_rate
        self.learning_rate_decay = learning_rate_decay
    
    def set_actor_model(self, actor_model): # policy
        if self.action_size > 1: # discrete action space
            actor_model.compile(loss='categorical_crossentropy',
                          optimizer=Adam(lr=self.actor_learning_rate))
        else: # continuous action space
            def _loss(y_true, y_pred):
                td_error = y_true[:][0][0]
                _, log_prob_pred, entropy = y_pred[:][0][0], y_pred[:][0][1], y_pred[:][0][2]
                return - log_prob_pred * td_error - entropy
            actor_model.compile(loss=_loss,
                          optimizer=Adam(lr=self.actor_learning_rate))
        self.actor_model = actor_model
        self.actor_model_set = True
        
    def set_critic_model(self, critic_model): # state value (advantage value)!
        critic_model.compile(loss='mse',
                      optimizer=Adam(lr=self.critic_learning_rate))
        self.critic_model = critic_model
        self.critic_model_set = True
        
    def choose_action(self, state): # stochastic policy
        assert self.actor_model_set, 'actor model not set!'
        if self.action_size > 1: # discrete action space
            act_values = self.actor_model.predict(state)
            action = np.random.choice(range(self.action_size), p=act_values[0])
            return action
        # continuous action space
        [action, _, _] = self.actor_model.predict(state)[0]
        return [action]
    
    def random_action(self, state):
        return self.env.action_space.sample()
    
    def explore(self, n_episodes):
        assert self.actor_model_set, 'actor model not set!'
        assert self.critic_model_set, 'critic model not set!'
        for _ in range(n_episodes):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            while True:
                action = self.random_action(state) # random action
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.reshape(next_state, [1, self.state_size])
                
                # learn
                target = reward
                target = np.reshape(target, [1,1])
                if not done:
                    target += self.gamma * self.critic_model.predict(next_state)
                td_error = target - self.critic_model.predict(state)
                if self.action_size > 1: # discrete action space
                    act_values = np.zeros((1,self.action_size))
                    act_values[0][action] = td_error[0] # make a one-hot vector to compute cross entropy
                    self.actor_model.fit(state, act_values, epochs=1, verbose=0)
                else: # continuous action space
                    act_values = np.zeros(self.action_size)
                    act_values[0] = td_error[0]
                    self.actor_model.fit(state, act_values, epochs=1, verbose=0)
                self.critic_model.fit(state, target, epochs=1, verbose=0)
                
                state = next_state
                if done:
                    break
        print("finished exploring!")
    
    def learn(self, n_episodes, visualize=False, verbose=0):
        assert self.actor_model_set, 'actor model not set!'
        assert self.critic_model_set, 'critic model not set!'
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
                
                # learn
                target = reward
                target = np.reshape(target, [1,1])
                if not done:
                    target += self.gamma * self.critic_model.predict(next_state)
                td_error = target - self.critic_model.predict(state)
                if self.action_size > 1: # discrete action space
                    act_values = np.zeros((1,self.action_size))
                    act_values[0][action] = td_error[0] # make a one-hot vector to compute cross entropy
                    self.actor_model.fit(state, act_values, epochs=1, verbose=0)
                else: # continuous action space
                    act_values = np.zeros(self.action_size) # this is not used, so any value is ok
                    act_values[0] = td_error[0]
                    self.actor_model.fit(state, np.concatenate([action, act_values]).reshape(1, 2),
                                         epochs=1, verbose=0)
                self.critic_model.fit(state, target, epochs=1, verbose=0)
                self.actor_learning_rate *= self.learning_rate_decay
                self.critic_learning_rate *= self.learning_rate_decay
                
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