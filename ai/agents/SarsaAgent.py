from random import random

class SarsaAgent(object):
    def __init__(self, env):
        # 保存一些Agent可以观测到的环境信息以及已经学到的经验
        self.env = env
        self.Q = {}  # {s0:[,,,,,,],s1:[,,,,,]} 数组内元素个数为行为空间大小
        self._initAgent()
        self.state = None

    def _get_state_name(self, state):   # 得到状态对应的字符串作为以字典存储的价值函数
        return str(state)               # 的键值，应针对不同的状态值单独设计，避免重复
                                        # 这里仅针对格子世界
    def _is_state_in_Q(self, s):
        return self.Q.get(s) is not None

    def _init_state_value(self, s_name, randomized = True):
        if not self._is_state_in_Q(s_name):
            self.Q[s_name] = {}
            for action in range(self.env.action_space.n):
                default_v = random() / 10 if randomized is True else 0.0
                self.Q[s_name][action] = default_v

    def _assert_state_in_Q(self, s, randomized=True):
        # 　cann't find the state
        if not self._is_state_in_Q(s):
            self._init_state_value(s, randomized)
    
    def _get_Q(self, s, a):
        self._assert_state_in_Q(s, randomized=True)
        return self.Q[s][a]

    def _set_Q(self, s, a, value):
        self._assert_state_in_Q(s, randomized=True)
        self.Q[s][a] = value

    def _initAgent(self):
        self.state = self.env.reset()
        s_name = self._get_state_name(self.state)
        self._assert_state_in_Q(s_name, randomized = False)
    
    # using simple decaying epsilon greedy exploration
    def _curPolicy(self, s, episode_num, use_epsilon):
        epsilon = 1.00 / (episode_num+1)
        if use_epsilon and random() < epsilon:  
            action = self.env.action_space.sample()
        else:
            Q_s = self.Q[s]
            action = int(max(Q_s, key=Q_s.get)) # choose best action
        return action

    # Agent依据当前策略和状态决定下一步的动作
    def choosePolicy(self, s, episode_num, use_epsilon=True):
        return self._curPolicy(s, episode_num, use_epsilon)

    def act(self, a):
        return self.env.step(a)

    # sarsa learning
    def learning(self, gamma, alpha, max_episode_num):
        # self.Position_t_name, self.reward_t1 = self.observe(env)
        time_in_episode, num_episode = 0, 0

        while num_episode < max_episode_num:
            self.state = self.env.reset()
            s0 = self._get_state_name(self.state)
            self.env.render() # show graphic interface
            a0 = self.choosePolicy(s0, num_episode, use_epsilon = True)
            
            time_in_episode = 0
            total_reward = 0
            done = False
            while not done:
                s1, r1, done, _ = self.act(a0)
                self.env.render() # show graphic interface
                s1 = self._get_state_name(s1)
                self._assert_state_in_Q(s1, randomized = True)
                # use_epsilon = False means Q-learning
                a1 = self.choosePolicy(s1, num_episode, use_epsilon=True)
                old_q = self._get_Q(s0, a0)
                q_prime = self._get_Q(s1, a1)
                td_target = r1 + gamma * q_prime
                #alpha = alpha / num_episode
                new_q = old_q + alpha * (td_target - old_q)
                self._set_Q(s0, a0, new_q)

                s0, a0 = s1, a1
                time_in_episode += 1
                total_reward += r1

            print("Episode {0} takes {1} steps, reward = {2}.".format(
                num_episode, time_in_episode, total_reward)) # after the episode is done
            num_episode += 1
        return   
