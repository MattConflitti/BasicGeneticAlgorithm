import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import random
import math
import datetime

## GENETIC ALGORITHM

MUTATE_RATE = 0.1
#keep pop size a multiple of 4
POP_SIZE = 100
NUM_GENS = 100
CROSS_RATE = 0.5
GENOME_SIZE = 2
MUTATE_BOUND = [-2.5,2.5]
WINDOW = [0,50]
NUM_ELITES = 1

def get_fitness(genome):
#    return np.sum(genome)
#    return -3*genome[0]**8 + 7*genome[1]**7 - 1*genome[2]**6 + 4*genome[3]**5 - 3*genome[4]**4 + 11*genome[5]**3 - 5.5*genome[6]**2 + 2*genome[7] + genome[8] 
#    return 3*math.exp(-((genome[0]+1)**2+(genome[1]+1)**2))
    x = genome[0]
    y = genome[1]
    return math.exp(-(math.cos(x/2)+math.cos(y/4))) + math.cos(x)
    
def select(genomes):
    random.shuffle(genomes)
    fittest = []
    for i in range(int(POP_SIZE/2)):
        p1 = genomes[i]
        p2 = genomes[2*i]
        if get_fitness(p1) >= get_fitness(p2):
            fittest.append(p1)
        else:
            fittest.append(p2)
    return fittest
        
    
def crossover(fittest):
    offspring = []
    elite = sorted(fittest.copy(), key=get_fitness, reverse=True)[0:NUM_ELITES]
    while len(offspring) < POP_SIZE:
        random.shuffle(fittest)
        females = fittest[0:int(len(fittest)/2)]
        males = fittest[int(len(fittest)/2):len(fittest)]
        for i in range(len(males)):
            mom = females[i]
            dad = males[i]
            if np.random.rand() < CROSS_RATE:
                child = []
                for i in range(len(dad)):
                    if np.random.rand() < 0.5:
                        child.append(dad[i])
                    else:
                        child.append(mom[i])
                offspring.append(child)
            else:
                #clone both parents
                offspring.append(dad)
                offspring.append(mom)
                
    return [*elite,*random.sample(offspring,POP_SIZE-NUM_ELITES)]
                
    
def mutate(offspring, gen_num):
    for child in offspring:
        for gene in range(GENOME_SIZE):
            if np.random.rand() < get_mutation_rate(gen_num) and (WINDOW[0] <= child[gene] + np.random.uniform(*MUTATE_BOUND) <= WINDOW[1]):
                child[gene] = child[gene] + np.random.uniform(*MUTATE_BOUND)
    
#initialize population all at starting point
genomes = POP_SIZE*[[12.5,25]]
#for i in range(POP_SIZE):
#    genomes.append(random.sample(range(0,10), GENOME_SIZE))
#    genomes.append([0 for _ in range(GENOME_SIZE)])

def get_mutation_rate(gen_num):
    #use 0.03 for ~80-120
    return MUTATE_RATE*math.exp(-(MUTATE_RATE/2)*gen_num)

rounds = []
for i in range(NUM_GENS):
    points = []
    top = select(genomes)
    offspring = crossover(top)
    mutate(offspring, i)
    genomes = offspring
    top = sorted(genomes.copy(), key=get_fitness, reverse=True)
    if i % 10 == 0:
        print(get_fitness(top[0]),top[0])
    for g in top:
        points.append((g[0], g[1]))
    rounds.append(points)
    
    #stop condition
#    if get_fitness(top[0]) > 8.3728:
#        break   

def fun(x, y):
  return math.exp(-(math.cos(x/2)+math.cos(y/4))) + math.cos(x)

delta = 0.05
x = np.arange(0, 20, delta)
y = np.arange(0, 50, delta)
X, Y = np.meshgrid(x, y)
Z1 = mlab.bivariate_normal(X, Y, 1.0, 1.0, 0.0, 0.0)
zs = np.array([fun(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])
Z = zs.reshape(X.shape)

norm = cm.colors.Normalize(vmax=abs(Z).max(), vmin=-abs(Z).max())
cmap = cm.PRGn

levels = np.arange(0, 10, 2)

filenames = []
for i,points in enumerate(rounds):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.contourf(X, Y, Z, levels,
                cmap=cm.get_cmap(cmap, len(levels)-1),
                norm=norm)
    ax.autoscale(False) # To avoid that the scatter changes limits
    
    ax.scatter(*zip(*points), color='red')
    #plt.show()
    if i % 2 == 0:
        file = 'plot'+ str(i) +'.png'
        fig.savefig(file)
        filenames.append(file)
    plt.close(fig)


import imageio
images = []
for filename in filenames:
    images.append(imageio.imread(filename))

st = datetime.datetime.now().strftime('%Y%m%d%H%M')
imageio.mimsave('climber_'+ st +'.gif', images)
