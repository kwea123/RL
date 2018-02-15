# simple example: minimize a quadratic around some solution point
import numpy as np
import matplotlib.pyplot as plt
solution = np.array([0.5, 0.1])
def f(w): return -np.sum((w - solution)**2)

npop = 50      # population size
sigma = 0.1    # noise standard deviation
alpha = 0.001  # learning rate
w = np.random.randn(2) # initial guess

plt.ion()
plt.figure(figsize=(20,10))
objs = []

for i in range(200):
    N = np.random.randn(npop, 2)
    w_tries = []
    R = np.zeros(npop)
    for j in range(npop):
        w_try = w + sigma*N[j]
        R[j] = f(w_try)
        w_tries += [w_try]
    
    for o in objs:
        o.remove()
    w_tries = np.vstack(w_tries)
    s1 = plt.scatter(w_tries[:,0], w_tries[:,1], c='b', s=20)
    s2 = plt.scatter(solution[0], solution[1], c='r', s=50)
    s3 = plt.scatter(w[0], w[1], c='g', s=50)
    objs = [s1, s2, s3]
    plt.title('Generation %d'%(i+1))
    plt.pause(0.05)
    
    A = (R - np.mean(R)) / np.std(R)
    w = w + alpha/(npop*sigma) * np.dot(N.T, A)
    #sigma *= 0.95
    #alpha *= 0.99

plt.ioff()
plt.show()