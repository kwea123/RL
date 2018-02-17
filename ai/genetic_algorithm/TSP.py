import numpy as np
import matplotlib.pyplot as plt

DNA_SIZE = 20
POP_SIZE = 500
CROSS_RATE = 0.5
MUTATION_RATE = 0.02
N_GENERATIONS = 200

def get_fitness(pop):
    distances = np.sum(np.sqrt(np.sum(np.square(np.diff(points[pop], axis=1)), axis=1)), axis=1)
    fitness = np.exp(DNA_SIZE/distances)
    return fitness, distances
    
def select(pop, fitness):
    idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True, p=fitness/fitness.sum())
    return pop[idx]

def crossover(parent, pop):
    if np.random.rand() < CROSS_RATE:
        i_ = np.random.randint(0, POP_SIZE, size=1) # select another individual
        cross_points = np.random.randint(0, 2, size=DNA_SIZE).astype(np.bool)
        keep_city = parent[~cross_points]
        swap_city = pop[i_, np.isin(pop[i_].ravel(), keep_city, invert=True)]
        parent = np.concatenate((keep_city, swap_city))
    return parent

def mutate(child):
    for point in range(DNA_SIZE):
        if np.random.rand() < MUTATION_RATE:
            swap_point = np.random.randint(0, DNA_SIZE)
            child[point], child[swap_point] = child[swap_point], child[point]
    return child

points = np.random.rand(DNA_SIZE,2)
pop = np.vstack([np.random.permutation(DNA_SIZE) for _ in range(POP_SIZE)])

plt.ion()       # something about plotting
plt.figure(figsize=(10,10))
plt.scatter(points[:,0], points[:,1])
objs = []

for i in range(N_GENERATIONS):
    fitness, distances = get_fitness(pop)
    best_idx = np.argmax(fitness)
    points2 = points[pop[best_idx]]
    
    for o in objs:
        o.remove()
    plot = plt.plot(points2[:,0], points2[:,1], 'r')
    plt.title('Generation %d'%(i+1))
    txt = plt.text(0.8, 0.05,'distance : %f'%distances[best_idx], fontsize=15)
    objs = plot + [txt]
    plt.pause(0.05)

    # GA part (evolution)
    pop = select(pop, fitness)
    pop_copy = pop.copy()
    for parent in pop:
        child = crossover(parent, pop_copy)
        child = mutate(child)
        parent = child       # parent is replaced by its child

plt.ioff()
plt.show()