import numpy as np
import matplotlib.pyplot as plt

DNA_SIZE = 10            # DNA length
POP_SIZE = 100           # population size
CROSS_RATE = 0.8         # mating probability (DNA crossover)
MUTATION_RATE = 0.003    # mutation probability
N_GENERATIONS = 50
X_BOUND = [0, 5]         # x upper and lower bounds


def F(x): return np.sin(10*x)*x + np.cos(2*x)*x     # to find the maximum of this function

# find non-zero fitness for selection
def get_fitness(pred): return pred + 1e-3 - np.min(pred)

# convert binary DNA to decimal and normalize it to a range(0, 5)
def translateDNA(pop): return pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2**DNA_SIZE-1) * X_BOUND[1]

def select(pop, fitness):    # nature selection wrt pop's fitness
    idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True,
                           p=fitness/fitness.sum())
    return pop[idx]

def crossover(loser_winner):      # crossover for loser
    for i in range(DNA_SIZE):
        if np.random.rand() < CROSS_RATE:
            loser_winner[0, i] = loser_winner[1, i]  # assign winners genes to loser
    return loser_winner

def mutate(loser_winner):         # mutation for loser
    for i in range(DNA_SIZE):
        if np.random.rand() < MUTATION_RATE:
            loser_winner[0, i] = 1-loser_winner[0, i]
    return loser_winner


pop = np.random.randint(2, size=(POP_SIZE, DNA_SIZE))   # initialize the pop DNA

plt.ion()       # something about plotting
plt.figure(figsize=(20,10))
x = np.linspace(*X_BOUND, 200)
plt.plot(x, F(x))
objs = []

for i in range(N_GENERATIONS):
    F_values = F(translateDNA(pop))    # compute function value by extracting DNA
    fitness = get_fitness(F_values)

    # something about plotting
    for o in objs:
        o.remove()
    sca = plt.scatter(translateDNA(pop), F_values, s=200, lw=0, c='red', alpha=0.5)
    plt.title('Generation %d'%(i+1))
    txt1 = plt.text(0, 2, 'Best fitness : '+str(np.max(fitness)), fontsize=15)
    txt2 = plt.text(0, 4, 'Most fitted DNA : '+str(translateDNA(pop[np.argmax(fitness), :])), fontsize=15)
    objs = [sca, txt1, txt2]
    plt.pause(0.05)

    # GA part (evolution)
    for _ in range(POP_SIZE):
        sub_pop_idx = np.random.choice(np.arange(0, POP_SIZE), size=2, replace=False)
        sub_pop = pop[sub_pop_idx]
        fitness = get_fitness(F(translateDNA(sub_pop)))
        loser_winner = sub_pop[np.argsort(fitness)]
        loser_winner = crossover(loser_winner)
        loser_winner = mutate(loser_winner)
        pop[sub_pop_idx] = loser_winner

plt.ioff()
plt.show()