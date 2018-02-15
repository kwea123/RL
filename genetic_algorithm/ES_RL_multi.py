import gym
import numpy as np
import multiprocessing as mp

#env = gym.make('CartPole-v0') # feature = 4, action = 2, discrete
env = gym.make('Pendulum-v0') # feature = 3, action = 1, continuous range (-2,2)

class ES():
    def __init__(self, sigma):
        self.shapes, self.oldnet = self.build_net()
        self.net = np.copy(self.oldnet)
        self.sigma = sigma
        
    def build_net(self):
        def linear(n_in, n_out):  # network linear layer
            w = np.random.randn(n_in * n_out).astype(np.float32) * .1
            b = np.random.randn(n_out).astype(np.float32) * .1
            return (n_in, n_out), np.concatenate((w, b))
        s0, p0 = linear(3, 20)
        s1, p1 = linear(20, 1)
        return [s0, s1], np.concatenate((p0, p1))
    
    def reset_net(self):
        self.net = np.copy(self.oldnet)
    
    def mutate(self):
        noise = np.random.randn(self.net.shape[0])
        self.net += self.sigma*noise
        return noise
    
    def update(self, learning_rate, noises, advs):
        gradient = np.dot(noises.T, advs)
        self.oldnet += learning_rate*gradient
    
    def choose_action(self, state):
        start = 0
        state = state[np.newaxis, :]
        for s in self.shapes:
            n_w, n_b = s[0]*s[1], s[1]
            state = np.tanh(state.dot(self.net[start:start+n_w].reshape(s))+self.net[start+n_w:start+n_w+n_b])
            start += n_w+n_b
        #if state[0] > 0:
        #    return 0
        #return 1
        return 2*state[0]
    
def job(es):
    state = env.reset()
    es.reset_net() # reset the old params
    noise_i = es.mutate() # mutate the child
    r_i = 0
    while True: # run simulation
        action = es.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        state = next_state
        r_i += reward
        if done:
            break
    return noise_i, r_i

if __name__ == "__main__":

    npop = 20
    sigma = 1e-1
    alpha = 5e-2

    es = ES(sigma)

    rank = np.arange(1, npop + 1)
    util_ = np.maximum(0, np.log(npop / 2 + 1) - np.log(rank))
    utility = util_ / util_.sum() - 1 / npop # it doesn't converge using utility this time...

    pool = mp.Pool()

    #train
    for e in range(500):
        jobs = [pool.apply_async(job, (es,)) for i in range(npop)] # each es in the job is a COPY
        results = np.array([j.get() for j in jobs])
        noises = results[:,0]
        rewards = results[:,1]
        print('\rbest reward for ep', e+1, ':', np.max(rewards), end=' '*10)
        #ranks = np.argsort(rewards)[::-1]
        #noises = np.vstack(noises)[ranks]
        #rewards = rewards[ranks].astype(np.float32)
        noises = np.vstack(noises)
        rewards = rewards.astype(np.float32)
        rewards = (rewards - np.mean(rewards))/np.std(rewards)
        es.update(alpha/(npop*sigma), noises, rewards)

    print('\nfinished training!')

    es.reset_net() # so we need to update the true ES in the main process
    
    #test
    for e in range(5):
        state = env.reset()
        r = 0
        while True:
            env.render()
            action = es.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            r += reward
            if done:
                print('reward for ep', e+1, ':', r)
                break

    print('finished testing!')
    env.close()