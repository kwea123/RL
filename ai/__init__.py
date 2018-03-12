from ai.agents.SarsaAgent import *
from ai.agents.PolicyGradientAgent import *
from ai.agents.PolicyGradientAgent2 import *
from ai.agents.DQNAgent import *
from ai.agents.DDQNAgent import *
from ai.agents.ActorCriticAgent import *
from ai.agents.DDPGAgent import *
from ai.agents.PPOAgent import *
from ai.agents.SarsaLambdaAgent import *
from ai.environments.gridworld import *
from ai.environments.arm_env import *
from ai.environments.arm_env_graph import *
from keras import backend as K
from keras.models import Model
from keras.layers import Dense, Input, add, Lambda, concatenate
from keras.utils import plot_model
import matplotlib.pyplot as plt
import gym

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def main():

#     env = GridWorldEnv(n_width=12,          # 水平方向格子数量
#                    n_height = 4,        # 垂直方向格子数量
#                    u_size = 60,         # 可以根据喜好调整大小
#                    default_reward = -1, # 默认格子的即时奖励值
#                    default_type = 0)    # 默认的格子都是可以进入的
#     env.action_space = spaces.Discrete(4)   # 设置行为空间数量
#     # 格子世界环境类默认使用0表示左，1：右，2：上，3:下，4,5,6,7为斜向行走
#     # 具体可参考_step内的定义
#     # 格子世界的观测空间不需要额外设置，会自动根据传输的格子数量计算得到
#     env.start = (0,0)
#     env.ends = [(11,0)]
#     for i in range(10):
#         env.rewards.append((i+1,0,-100))
#         env.ends.append((i+1,0))
#     env.types = [(5,1,1),(5,2,1)]
#     env.rewards.append((11,0,100))
#     env.refresh_setting()
#     env.reset()
#     env = SimpleGridWorld()
     
#     agent = SarsaLambdaAgent(env)
#     env.reset()
#     print("Learning...")  
#     agent.learning(lambda_ = 0.2,
#                    gamma=0.9, 
#                    alpha=0.1, 
#                    max_episode_num=500)

    env = gym.make('CartPole-v0')
#     env = gym.make('MountainCar-v0')
#     env = gym.make('Pendulum-v0')
#     env = gym.make("MountainCarContinuous-v0")
    
#     env = ArmEnv(mode='hard')
#     env.seed(1)
    
    #discrete action space required
#     agent = PolicyGradientAgent2(env,
#     n_actions=env.action_space.n,
#     n_features=env.observation_space.shape[0],
#     learning_rate=0.01,
#     reward_decay=0.95)
    
    agent = DQNAgent(env,
            n_actions=env.action_space.n,
            n_features=env.observation_space.shape[0],
            learning_rate=0.02, priority_alpha=0)
      
    state_inputs = Input(shape=(agent.state_size,))
    x = Dense(10, activation='relu', name='l1')(state_inputs)
    output = Dense(agent.action_size, activation='linear', name='l2')(x)
    model = Model(inputs=state_inputs, outputs=output)
    agent.set_model(model)
       
#     rewards_p = agent_p.learn(500,visualize=False,verbose=1)

#     agent = DDQNAgent(env,
#             n_actions=env.action_space.n,
#             n_features=env.observation_space.shape[0],
#             learning_rate=0.02, priority_alpha=0)
    
#     state_inputs = Input(shape=(agent.state_size,))
#     x = Dense(10, activation='tanh', name='l1')(state_inputs)
#     action_value_net = Dense(agent.action_size, activation='linear', name='l_a')(x)
# #     state_value_net = Dense(1, activation='linear', name='l_s')(x)
# #     Q_net = add([action_value_net, state_value_net]) # dueling q learning
#     model = Model(inputs=state_inputs, outputs=action_value_net)
#     agent.set_model(model)
    
     
#     agent = ActorCriticAgent(env,
#             n_actions=env.action_space.n, # discrete space
#             n_features=env.observation_space.shape[0],
#             actor_learning_rate=0.001,
#             critic_learning_rate=0.005,
#             learning_rate_decay=0.99)
#            
#     state_inputs = Input(shape=(agent.state_size,))
#     x = Dense(10, activation='relu', name='l1_actor')(state_inputs)
#     policy_net = Dense(agent.action_size, activation='softmax', name='l_p')(x)
#     y = Dense(10, activation='relu', name='l1_critic')(state_inputs)
#     state_value_net = Dense(1, activation='linear', name='l_s')(y)
#     actor_model = Model(inputs=state_inputs, outputs=policy_net)
#     critic_model = Model(inputs=state_inputs, outputs=state_value_net)
#     agent.set_actor_model(actor_model)
#     agent.set_critic_model(critic_model)
 
#     agent = DDPGAgent(env,
#             n_actions=1, # continuous action
#             n_features=env.observation_space.shape[0],
#             featurize=False, 
#             action_high=env.action_space.high[0],
#             action_low=env.action_space.low[0],
#             actor_learning_rate=0.001,
#             critic_learning_rate=0.002,
#             priority_alpha=0
#             )

#     agent = DDPGAgent(env,
#             n_actions=env.action_dim,
#             n_features=env.state_dim,
#             featurize=False, 
#             action_high=1,
#             action_low=-1,
#             actor_learning_rate=0.001,
#             critic_learning_rate=0.002,
#             priority_alpha=0
#             )

#     agent = PPOAgent(env,
#                     n_actions=1,
#                     n_features=env.observation_space.shape[0],
#                     featurize=False, 
#                     action_high=env.action_space.high[0],
#                     action_low=env.action_space.low[0],
#                     actor_learning_rate=0.001,
#                     critic_learning_rate=0.002
#                     )
    
#     tf.summary.FileWriter("./logs", agent.sess.graph)
    
    rewards = agent.learn(1000,visualize=False,verbose=1)
    plt.plot(rewards)
    plt.show()
#     agent.test(10)

if __name__ == "__main__":
    main()