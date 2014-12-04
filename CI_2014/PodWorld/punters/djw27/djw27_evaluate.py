
# This code would be a good place to start your evolution coding.
from operator import itemgetter, attrgetter
import math,random, copy, pickle
import numpy
import pprint

import sys
sys.path.append("../../")
from pod import world,gui,pods

pp = pprint.PrettyPrinter(indent=4)

TEST=True

if TEST:
    GUI=True
else:
    GUI=False

FORWARDS_ANGLE = math.radians(0)
LEFT_ANGLE = math.radians(90)
RIGHT_ANGLE = math.radians(-90)

def equip_car(pod):

    sensors=[ pods.Sensor(angle=FORWARDS_ANGLE,name="Forwards"),
              pods.Sensor(angle=LEFT_ANGLE/5,name="Eigth-Left"),
              pods.Sensor(angle=LEFT_ANGLE/3,name="Quarter-Left"),
              pods.Sensor(angle=RIGHT_ANGLE/5,name="Eigth-Right"),
              pods.Sensor(angle=RIGHT_ANGLE/3,name="QuarterRight"),
            ]

    pod.addSensors(sensors)
    pod.col=(0,255,0)
    pod.data=[0,0,0,0,0,0,0]    # default control system parameters



def controller(pod,control,NN):

    inputValues = [ pod.sensors[0].val,
                    pod.sensors[1].val,
                    pod.sensors[2].val,
                    pod.sensors[3].val,
                    pod.sensors[4].val,
                    pod.state.vel,
                    pod.state.slip,
                  ]
    pod.data = NN.update(inputValues)

    """
    Simplest possible control system. The NN calculates
    values for acceleration, braking and left/right for
    any given sensor values
    """

    control.up=pod.data[0]
    control.down=pod.data[1]
    control.left=pod.data[2]
    control.right=pod.data[3]

    if pod.state.age < 0.2:
        control.up=1.0
        control.down=0.0
    """

    if pod.data[0] > 0:
        control.up=pod.data[0]
        control.down=0
    else:
        control.up=0
        control.down=abs(pod.data[0])
    if pod.data[1] > 0:
        control.left=pod.data[1]
        control.right=0
    else:
        control.left=0
        control.right=abs(pod.data[1])
        """

def evaluate(pod,NN,simple_gui):
    """
    Showing how you can evaluate the performance of your car.
    """

    TIME_LIMIT=40.0
    TRIPS_LIMIT=60.0

    # reset the state of the car before starting
    pod.reset()

    while True:

        if GUI:
            mess=str(pod.state)
            simple_gui.set_message(mess)
            simple_gui.display()

            if simple_gui.check_for_quit():
                break

        if pod.state.collide:
            dist=pod.state.pos_trips-pod.state.neg_trips+pod.state.seg_pos
            age=pod.state.age
            return dist

        if pod.state.age > TIME_LIMIT:
            dist=pod.state.pos_trips-pod.state.neg_trips
            return dist

        if pod.state.pos_trips - pod.state.neg_trips > TRIPS_LIMIT:
            return TRIPS_LIMIT + (TIME_LIMIT-pod.state.age)

        controller(pod,control,NN)
        pod.step(control)

class NeuralNet(object):
    """
    Generalised class for
    generating a neural net
    """
    def __init__(self, layerVector, weights):
        # Number of input, hidden and output nodes
        self.layerVector = layerVector
        self.layerVector[0] = layerVector[0]+1 # Adding in the bias
        self.ni = self.layerVector[0]
        self.nh = self.layerVector[1]
        self.no = self.layerVector[2]
        # Set the input and output weights
        self.weights = weights
        self.wi = weights[0]
        self.wo = weights[1]

        # Allocate memory for output
        # of each neuron
        self.out = []

        for i in range(len(self.layerVector)):
            # Create an array for each 'layer'
            # of output values i.e. hidden
            # and final output
            self.out.append([])
            # Starting at the hidden layer,
            # append zeros representing the
            # output locations
            for j in range(self.layerVector[i]):
                self.out[i].append(0.0)


    def update(self, inputValues):
        self.inputValues = inputValues
        self.inputValues.append(1) # add bias

        # Check we have the right number of inputs
        if len(self.inputValues) != self.ni:
            print "Incorrect number of inputs"

        # assign content to input layer
        for i in range(self.layerVector[0]):
            self.out[0][i] = self.inputValues[i]

        # For each layer
        for i in range(len(self.layerVector)-1):
            # For each neuron sum the hidden layer
            for j in range(self.layerVector[i+1]):
                sum = 0.0;
                # Sum the results for each neuron
                # of the input layer * weighting
                # to the current neuron in the
                # hidden layer
                for k in range(self.layerVector[i]):
                    sum += self.out[i][k] * self.weights[i][k][j]

                # Apply transfer function of choice
                self.out[i+1][j] = sigmoid(sum)

        # Return the outputs
        return self.out[len(self.layerVector)-1]


class GeneticAlgorithm(object):
    """
    Generalised class for
    GA functions
    """
    def __init__(self, layerVector, popSize):
        # Number of input, hidden and output nodes
        self.ni = layerVector[0]+1 # Adding in the bias
        self.nh = layerVector[1]
        self.no = layerVector[2]
        self.popSize = popSize

    def generatePop(self):
        # Initialise population
        pop = []
        # Loop through the population size
        for i in range(self.popSize):
            # Reset the gene each loop
            gene = []
            # Initialise input weights
            wi = [ [0.0]*self.nh for i in range(self.ni) ]
            # Initialise output weights
            wo = [ [0.0]*self.no for j in range(self.nh) ]
            # randomize node weight matrices
            randomizeMatrix(wi, -0.2, 0.2)  # How does changing this range effect the result?
            randomizeMatrix(wo, -2.0, 2.0)  # How does changing this range effect the result?
            # combine weights into a gene
            gene = [wi, wo]
            # add the gene to the population
            pop.append(gene)

        return pop

    def mutate(self, m):
        for i in range(len(m)):
            for j in range(len(m[i])):
                for k in range(len(m[i][j])):
                    if random.random() < MUTATEPROB:
                        m[i][j][k] = random.uniform(-1.0, 1.0)

    def mutateLess(self, m):
        for i in range(len(m)):
            for j in range(len(m[i])):
                for k in range(len(m[i][j])):
                    if random.random() < MUTATEPROB:
                        m[i][j][k] += random.uniform(-1.0, 1.0)*0.1

    def mutateMore(self, m):
        for i in range(len(m)):
            for j in range(len(m[i])):
                for k in range(len(m[i][j])):
                    if random.random() < MUTATEPROB*2:
                        m[i][j][k] += random.uniform(-1.0, 1.0)

    def mate(self,a,b):
        a=copy.deepcopy(a) # Is this necessary?
        b=copy.deepcopy(b) # Is this necessary?
        size = len(a)+len(b)
        i=(random.randint(1,size-1) + 1)/2
        ret = a[0:i]+b[i:size]
        return ret

    def clone(self, m):
        return copy.deepcopy(m)

    def nextGen(self, rankedPop, mutateFlag, mutateCount):
        newPop = []
        wi = []
        wo = []
        # Separate the weights and fitnesses
        # from the old ranked population
        oldRankedPop, fitnesses = zip(*rankedPop)
        # Copy the elite across
        for m in oldRankedPop[0:NELITE]:
            newPop.append(m)

        # If we are stuck in a rut then generate
        # some brand new weights
        if mutateCount >=1000:
            new = self.generatePop()
            newPop.append(new[NELITE-1:POPSIZE])
        else:
            for m in range(len(oldRankedPop[NELITE:POPSIZE])):
                wi.append(oldRankedPop[m][0])
                wo.append(oldRankedPop[m][1])

            while len(newPop) < POPSIZE:
                i = random.randint(0,NSELECT-1)
                if random.random() < CROSSOVERPROB:
                    j = random.randint(0,NSELECT-1)
                    while i==j:
                        j = random.randint(0,NSELECT-1)

                    geneInput = self.mate(wi[i],wi[j])
                    geneOutput = self.mate(wo[i],wo[j])

                    if mutateFlag:
                        if random.random() < MUTATEPROB:
                            self.mutateMore([geneInput,geneOutput])
                    else:
                        if random.random() < MUTATEPROB:
                            self.mutateLess([geneInput,geneOutput])
                else:
                    geneInput = self.clone(wi[i])
                    geneOutput = self.clone(wo[i])
                    self.mutate([geneInput,geneOutput])

                newPop.append([geneInput,geneOutput])

        return newPop

"""
Misc functions beyond this line
"""

def sigmoid(value):
    """
    Calculates the sigmoid .

    """

    try:
        value = 1.0 / (1.0 + math.exp(-value))
    except OverflowError:
        value = 0.0

    return value

def tanh(value):
    """
    This function calculates the hyperbolic tangent function.

    """

    return math.tanh(value)

def linear(value):
    """
    This function simply returns the value given to it.

    """

    return value


def randomizeMatrix(matrix, a, b):
    """
    This function will fill a given matrix with random values.
    """
    for i in range ( len (matrix) ):
        for j in range ( len (matrix[0]) ):
            matrix[i][j] = random.uniform(a,b)

def savePop(pop,filename="pop.dat"):
    """
    Saves a population
    """

    fout=open(filename,'w')
    pickle.dump(pop,fout)
    fout.close()

def loadPop(filename="pop.dat"):
    """
    Loads a saved population
    """
    fin=open(filename,'r')
    b=pickle.load(fin)
    fin.close()
    return b

def pairPop(pop,inputNeurons,hiddenNeurons,outputNeurons):
    weights, fitnesses = [], []
    for i in range(len(pop)):
        layerVector = [inputNeurons, hiddenNeurons, outputNeurons]
        # Select an example weight to process through the neural
        # net. In practice this would loop through the population.
        weight = pop[i] # This is a [[6x(5+1)],[6x4]] matrix
        NN = NeuralNet(layerVector, weight)
        fitness=0
        for each in worlds:
            # create  the world
            world=each
            dt=world.dt

            pod=pods.CarPod(world)
            equip_car(pod)


            if GUI:
                frames_per_sec=int(1/dt)
                #frames_per_sec=200
                simple_gui=gui.SimpleGui(frames_per_sec=frames_per_sec,world=world,pods=[pod])

                fitness+=evaluate(pod, NN, simple_gui)
            else:
                fitness+=evaluate(pod, NN, 0)

        weights.append(weight)
        fitnesses.append(fitness)

    return zip(weights, fitnesses)




worlds = []
worlds.append(world.World("../../worlds/carCircuit.world"))
worlds.append(world.World("../../worlds/pjl_round.world"))
worlds.append(world.World("../../worlds/pjl_long.world"))
worlds.append(world.World("../../worlds/pjl_chick.world"))


# use a control to activate the car.
control=pods.Control()

"""
Constants are defined here
"""
previousFit = 0
"""
GA Constants
"""
POPSIZE         = 100      # Population size
MAXITER         = 1000        # Maximum number of iterations
CROSSOVERPROB   = 0.5       # Probability that we create by crossover breeding
MUTATEPROB      = 0.1       # Mutation rate of crossover offspring
ELITEPERCENT    = 0.1      # Percentage of ranked population to keep
SELECTPERCENT   = 0.3       # Top 30% available for selection
targetFitness   = 400

NELITE  = int(POPSIZE*ELITEPERCENT)     # top of population survive
NSELECT = int(POPSIZE*SELECTPERCENT)    # how many are bred
"""
NN Constants
"""
inputNeurons = 7
hiddenNeurons = 5
outputNeurons = 4
layerVector = [inputNeurons, hiddenNeurons, outputNeurons]


if TEST:
    popR=loadPop()
    pop,weights=zip(*popR)
else:
    # Initialise an instance of the GA and generate a random
    # starting population
    GA = GeneticAlgorithm(layerVector, POPSIZE)
    #pop=GA.generatePop()
    popR=loadPop()
    popJ,weights=zip(*popR)
    pop=[]
    while len(pop)<(len(popJ)):
        pop.append(popJ[0])

# Evaluate the population and pair fitnesses and weights
pairedPop = pairPop(pop,inputNeurons,hiddenNeurons,outputNeurons)
# Rank the population in order of fitness
rankedPop = sorted(pairedPop, key=itemgetter(-1), reverse = True)

savePop(rankedPop)
count = 0
oldPop=[0]*POPSIZE
oldFitness=[0]*POPSIZE
oldRankedPop=zip(oldPop,oldFitness)

mutateFlag=False
mutateCount=0

while count < MAXITER:
    # If we have a new highest fitness then save the population
    if rankedPop[0][-1] > previousFit:
        previousFit = rankedPop[0][-1]
        savePop(rankedPop)
        print "Saved new Population with fitness: " + repr(rankedPop[0][-1]) + " Iteration: " + repr(count)

    # If we have reached our goal then stop
    if rankedPop[0][-1] >= targetFitness:
        break

    # Generate the new population
    pop = GA.nextGen(rankedPop, mutateFlag, mutateCount)

    #DEBUG
    #for i in range(len(pop)):
    #   if oldPop[0] == pop[i]:
    #        print "Hit", i

    # Evaluate the population and pair fitnesses and weights
    pairedPop = pairPop(pop,inputNeurons,hiddenNeurons,outputNeurons)

    # Rank the population in order of fitness
    rankedPop = sorted(pairedPop, key=itemgetter(-1), reverse = True)

    #print rankedPop[0][-1], oldFitness[0]

    if rankedPop[0][-1] == oldFitness[0]:
        mutateFlag=True
        mutateCount+=1
    else:
        mutateCount=0

    if mutateCount>5:
        mutateCount=0

    # If the new mutated highest fitness is less than
    # the previously calculated then discard the new
    # generation and replace with the old generation
    if rankedPop[0][-1] < oldFitness[0]:
        rankedPop = oldRankedPop

    oldRankedPop = rankedPop

    oldPop, oldFitness = zip(*oldRankedPop)

    print "================================================="

    count += 1