from operator import itemgetter
import math,random, copy, pickle

# Link to the main PodWorld directory
import sys
sys.path.append("../../")
from pod import world,gui,pods

# If the user would like to test a generated population then change the
# TEST flag to true
TEST=False
if TEST:
    GUI=True
else:
    GUI=False

# Define angles used for sensor positioning on the car
FORWARDS_ANGLE = math.radians(0)
LEFT_ANGLE = math.radians(90)
RIGHT_ANGLE = math.radians(-90)

def equip_car(pod):
    """
    Initialisation of the car, setting the sensors and starting
    parameters. The car shape and color can also be set here.
    """
    sensors=[ pods.Sensor(angle=FORWARDS_ANGLE,name="Forwards"),
              pods.Sensor(angle=LEFT_ANGLE/5,name="Left-/5"),
              pods.Sensor(angle=LEFT_ANGLE/3,name="Left-/3"),
              pods.Sensor(angle=RIGHT_ANGLE/5,name="Right-/5"),
              pods.Sensor(angle=RIGHT_ANGLE/3,name="Right-/3"),
            ]
    pod.addSensors(sensors)

    pod.col=(204,255,225)
    # Generate a random shape for the car on each run
    pod.poly=[(-20*random.random(),-20*random.random()),(-20*random.random(),20*random.random()),(20*random.random(),20*random.random()),(20*random.random(),-20*random.random())]
    # Initialise the pod control parameters to zero to ensure
    # repeatable results are produced.
    pod.data= [0,0,0,0]

def controller(pod,control,NN):
    """
    Simple control system for controlling the car. The NN calculates
    values for acceleration, brake, left and right for any given sensor
    values.
    """

    # Inputs to the neural net include all the sensors defined in
    # equip_car() along with the current speed of the car and the
    # current value of slip
    inputValues = [ pod.sensors[0].val,
                    pod.sensors[1].val,
                    pod.sensors[2].val,
                    pod.sensors[3].val,
                    pod.sensors[4].val,
                    pod.state.vel,
                    pod.state.slip,
                  ]

    # The Neural Net is used to generate new outputs based on the
    # current inputValues
    pod.data = NN.update(inputValues)

    # Launch control is partly hard-coded here. If the car is less than
    # 0.45s into a race then accelerate max and brake zero, whilst
    # allowing the Neural Net to control the steering.
    #
    # Let the Neural Net control the car fully after this.
    if pod.state.age < 0.45:
        control.up=1.0
        control.down=0.0
        control.left=pod.data[2]
        control.right=pod.data[3]
    else:
        control.up=pod.data[0]
        control.down=pod.data[1]
        control.left=pod.data[2]
        control.right=pod.data[3]

def evaluate(pod,NN,simple_gui):
    """
    Evaluate the performance of the car for a given neural net
    and set of weights on a given track. This function returns
    a fitness value which is used to rank genes within a population.
    """

    # Set the maximum simulation run time
    TIME_LIMIT=40.0
    # Set the desired number of checkpoints to cross
    # before classifying the run as a success.
    TRIPS_LIMIT=60.0

    # Reset the state of the car before starting
    pod.reset()

    while True:

        if GUI:
            mess=str(pod.state)
            simple_gui.set_message(mess)
            simple_gui.display()

            if simple_gui.check_for_quit():
                break

        # If the car crashes before the TIME_LIMIT is reached then the
        # fitness is based purely on the distance travelled
        if pod.state.collide:
            dist=pod.state.pos_trips-pod.state.neg_trips+pod.state.seg_pos
            age=pod.state.age
            return dist

        # If the car does not complete the track within the TIME_LIMIT
        # but has not crashed then the fitness is based on the distance
        # travelled
        if pod.state.age > TIME_LIMIT:
            dist=pod.state.pos_trips-pod.state.neg_trips
            return dist

        # If the car has travelled past more checkpoints than
        # TRIPS_LIMIT then consider the run a success and generate a
        # fitness based on the distance travelled and time taken
        if pod.state.pos_trips - pod.state.neg_trips > TRIPS_LIMIT:
            return TRIPS_LIMIT + (TIME_LIMIT-pod.state.age)

        # Update the output control parameters for the car at the
        # current timestep and then step the car forward
        controller(pod,control,NN)
        pod.step(control)
    return

class NeuralNet(object):
    """
    Generalised class for generating a Neural Net with a single hidden
    layer. Each layer has a user defined number of neurons. Weightings
    are not defined within this class.
    """

    def __init__(self, layerVector, weights):
        # Copy across the number of neurons in each layer
        self.layerVector = copy.deepcopy(layerVector)
        # Adding in the bias
        self.layerVector[0] = self.layerVector[0]+1

        self.ni = self.layerVector[0]              # Input layer size
        self.nh = self.layerVector[1]              # Hidden layer size
        self.no = self.layerVector[2]              # Output layer size


        # Set the input and output weights
        self.weights = weights
        self.wi = self.weights[0]
        self.wo = self.weights[1]

        # Allocate memory for the output of each neuron
        self.out = []

        for i in range(len(self.layerVector)):
            # Create an array for each 'layer' of output values i.e.
            # hidden layer values and final output values
            self.out.append([])
            # Starting at the hidden layer, append zeros representing
            # temporary output values
            for j in range(self.layerVector[i]):
                self.out[i].append(0.0)


    def update(self, inputValues):
        """
        Return the output values from the Neural Net for a set of new
        input values. This function allows the neural net to be used as
        a real-time controller.
        """
        self.inputValues = inputValues
        self.inputValues.append(1)                 # Add the bias

        # Check the number of inputs is correct. This check is useful
        # to catch initialisation errors when moving between tracks
        if len(self.inputValues) != self.ni:
            print "Incorrect number of inputs"

        # Assign content to input layer
        for i in range(self.layerVector[0]):
            self.out[0][i] = self.inputValues[i]

        # For each layer of the Neural Net
        for i in range(len(self.layerVector)-1):
            # For each neuron in the next layer
            for j in range(self.layerVector[i+1]):
                sum = 0.0;
                # Sum the results for each neuron of the ith *
                # weighting to the current neuron in the ith+1 layer)
                for k in range(self.layerVector[i]):
                    sum += self.out[i][k] * self.weights[i][k][j]

                # Apply transfer function of choice to the sum
                self.out[i+1][j] = sigmoid(sum)

        # Return the outputs
        return self.out[len(self.layerVector)-1]

class GeneticAlgorithm(object):
    """
    Generalised class for Genetic Algorithm functions specifically
    for use with Neural Nets.
    """
    def __init__(self, layerVector, popSize):
        # Copy across the number of neurons in each layer
        self.layerVector = copy.deepcopy(layerVector)
        # Adding in the bias
        self.layerVector[0] = self.layerVector[0]+1

        self.ni = self.layerVector[0]              # Input layer size
        self.nh = self.layerVector[1]              # Hidden layer size
        self.no = self.layerVector[2]              # Output layer size

        self.popSize = popSize

    def generatePop(self):
        """
        Generate a population of random genes which represent sets
        of input and output weights of a Neural Net. The gene structure
        is as follows:
        Gene: [ [Input weights] , [Output weights] ]

        Input weights: [Input neuron 1 weights to each hidden layer neuron],
                       [Input neuron 1 weights to each hidden layer neuron],
                       ...                                                 ,
                       [Input neuron i weights to each hidden layer neuron]

        Output weights: [Hidden layer neuron 1 weights to each output neuron],
                        [Hidden layer neuron 2 weights to each output neuron],
                        ...                                                  ,
                        [Hidden layer neuron i weights to each output neuron]

        For a neural net with 3 input neurons and 2 hidden layer neurons
        an example set of input weights would look as follows:
        [ [-0.1,0.4], [0.2,-0.7], [0.3,0.9] ]
        """

        # Initialise population
        pop = []
        # Generate a random population of genes
        for i in range(self.popSize):
            # Initialise an empty gene each loop
            gene = []
            # Allocate memory for the input weights
            wi = [ [0.0]*self.nh for i in range(self.ni) ]
            # Allocate memory for the output weights
            wo = [ [0.0]*self.no for j in range(self.nh) ]
            # Randomize the weight values
            randomizeMatrix(wi, -0.2, 0.2)         # +-0.2 arbitrary
            randomizeMatrix(wo, -2.0, 2.0)         # +-2.0 arbitrary
            # Construct a gene from the weights
            gene = [wi, wo]
            # Add the gene to the population
            pop.append(gene)

        return pop

    def mutate(self, weights):
        """
        Mutate MUTATEPROB of the individual weights within a gene
        by generating a new value between +-1 for that weight
        """
        for i in range(len(weights)):
            for j in range(len(weights[i])):
                for k in range(len(weights[i][j])):
                    if random.random() < MUTATEPROB:
                        weights[i][j][k] = random.uniform(-1.0, 1.0)
        return

    def mutateSmallNudge(self, weights):
        """
        Mutate MUTATEPROB of the individual weights within a gene
        by up to +-0.05 of the current value of that weight
        """
        for i in range(len(weights)):
            for j in range(len(weights[i])):
                for k in range(len(weights[i][j])):
                    if random.random() < MUTATEPROB:
                        weights[i][j][k] += random.uniform(-1.0, 1.0)*0.05
        return

    def mutateBigNudge(self, weights):
        """
        Mutate MUTATEPROB*2 of the individual weights within a gene
        by up to +-1 of the current value of that weight
        """
        for i in range(len(weights)):
            for j in range(len(weights[i])):
                for k in range(len(weights[i][j])):
                    if random.random() < MUTATEPROB*2:
                        weights[i][j][k] += random.uniform(-1.0, 1.0)
        return

    def crossover(self, mother, father):
        """
        Generate a child gene from a pair of different adult genes
        """
        mother=copy.deepcopy(mother)
        father=copy.deepcopy(father)
        # Get the size of the parent genes. This will differ depending
        # on whether the gene represents an input or output weight to
        # the neural net
        size = len(mother)
        # Decide a point at which to combine the parent genes into a
        # child gene
        i=(random.randint(1,size-1) + 1)/2
        child = mother[0:i]+father[i:size]

        return child

    def clone(self, gene):
        """
        Clone the input gene
        """
        return copy.deepcopy(gene)

    def nextGen(self, rankedPop, stuckFlag):
        """
        This function generates the next generation of the population
        based on a mixture of mutation, breeding/crossover and elitism.
        """
        # Initialise variables
        newPop, wiOld, woOld, wiNew, woNew = [], [], [], [], []
        # Separate the weights and fitnesses from the old ranked
        # population
        oldRankedPop, fitnesses = zip(*rankedPop)

        # Preserve the elite in the new population
        for m in oldRankedPop[0:NELITE]:
            newPop.append(m)

        # The rest of the genes in the population will be generated
        # using the top n genes from the old population. The input and
        # output weights are separated here so that they can be
        # genetically modified independent of one another
        for n in range(len(oldRankedPop[NELITE:POPSIZE])):
            wiOld.append(oldRankedPop[n][0])
            woOld.append(oldRankedPop[n][1])

        while len(newPop) < POPSIZE:
            # Generate a random integer used to select weights from the
            # top NSELECT genes
            i = random.randint(0,NSELECT-1)

            # Crossover/breeding occurs here
            if random.random() < CROSSOVERPROB:
                # Generate another random integer used to select weights
                # from the top NSELECT genes
                j = random.randint(0,NSELECT-1)
                # Ensure parents are different for breeding
                while i==j:
                    j = random.randint(0,NSELECT-1)
                # Independently generate new sets of input and output
                # weights by using the crossover function
                wiNew = self.crossover(wiOld[i],wiOld[j])
                woNew = self.crossover(woOld[i],woOld[j])

                # If the population has stagnated then increase the
                # amount of mutation that takes place and vary the type
                # of mutation taking place
                if stuckFlag:
                    if random.random() < MUTATEPROB:
                        self.mutateBigNudge([wiNew,woNew])
                    else:
                        self.mutateSmallNudge([wiNew,woNew])
                else:
                    # Mutate a small proportion of children generated
                    # from crossover to try and slightly improve the
                    # gene. Only a small change is needed on some of
                    # the genes as these are already naturally good
                    # genes and shouldn't be altered too much.
                    if random.random() < MUTATEPROB:
                        self.mutateSmallNudge([wiNew,woNew])
            else:
                # Cloning and mutation occurs here
                wiNew = self.clone(wiOld[i])
                woNew = self.clone(woOld[i])
                self.mutate([wiNew,woNew])
            # Add generated gene to the new population
            newPop.append([wiNew,woNew])

        return newPop

def sigmoid(value):
    """
    Calculates the sigmoid

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



def randomizeMatrix(matrix, a, b):
    """
    This function will fill a given matrix with random values.
    """

    for i in range ( len (matrix) ):
        for j in range ( len (matrix[0]) ):
            matrix[i][j] = random.uniform(a,b)
    return

def savePop(pop,filename="popTest.dat"):
    """
    Saves a population
    """

    fout=open(filename,'w')
    pickle.dump(pop,fout)
    fout.close()
    return

def loadPop(filename="popTest.dat"):
    """
    Loads and returns a saved population
    """
    fin=open(filename,'r')
    b=pickle.load(fin)
    fin.close()
    return b

def pairPop(pop,layerVector):
    """
    This function evaluates each gene within a population and returns
    the genes paired with their corresponding fitnesses
    """
    weights, fitnesses = [], []
    for i in range(len(pop)):
        # Select a set of weights an initialise the Neural Net
        weight = pop[i]
        NN = NeuralNet(layerVector, weight)

        # Reset the fitness on each iteration
        fitness=0
        # Calculate a fitness for each track selected
        for each in worlds:
            # Create  the world
            world=each
            dt=world.dt

            pod=pods.CarPod(world)
            equip_car(pod)

            if GUI:
                frames_per_sec=int(1/dt)
                simple_gui=gui.SimpleGui( frames_per_sec=frames_per_sec,
                                          world=world,pods=[pod] )
                # Sum the fitnesses together on each iteration for the
                # different trakcs
                fitness+=evaluate(pod, NN, simple_gui)
            else:
                # Sum the fitnesses together on each iteration for the
                # different trakcs
                fitness+=evaluate(pod, NN, 0)

        # If in test mode then print the fitness
        if TEST:
            print "Finished with a fitness of: " + repr(fitness)

        # Add the current set of weights and corresponding fitness to
        # lists
        weights.append(weight)
        fitnesses.append(fitness)

    # Return a combined tuple of weights and their fitnesses
    return zip(weights, fitnesses)


"""
Main code is placed below. This code will set the genetic algorithm
continuously running until a desired fitness level is reach or until
the user intervenes.
"""

# Create an array of the tracks for which to evaluate the car on
worlds = []
worlds.append(world.World("../../worlds/carCircuit.world"))
worlds.append(world.World("../../worlds/pjl_round.world"))
worlds.append(world.World("../../worlds/pjl_long.world"))
worlds.append(world.World("../../worlds/pjl_chick.world"))
worlds.append(world.World("../../worlds/bravenew.world"))
worlds.append(world.World("../../worlds/unseen2.world"))

# Use a control to activate the car.
control=pods.Control()

"""
Genetic Algorithm Constants
"""
# Population size
POPSIZE         = 100
# Maximum number of generations
MAXITER         = 1000
# Probability that a gene is created by crossover breeding
CROSSOVERPROB   = 0.5
# Mutate 10% of all newly created genes
MUTATEPROB      = 0.1
# Keep top 10% of the old ranked population for the next generation,
# untouched
ELITEPERCENT    = 0.1
# Top 30% of the old ranked population are used to generated the new
# population
SELECTPERCENT   = 0.3
# The genetic algorithm will continue running until this fitness value
# is reached. 600 is based on evaluating the fitness across 6 tracks
# and represents an unattainable value
targetFitness   = 600

NELITE  = int(POPSIZE*ELITEPERCENT)     # Top of the population survive
NSELECT = int(POPSIZE*SELECTPERCENT)    # Number of genes to breed from
"""
Neural Net Constants
"""
inputNeurons  = 7
hiddenNeurons = 5
outputNeurons = 4
# Combine the neurons into a vector to simplify use throughout code
layerVector = [inputNeurons, hiddenNeurons, outputNeurons]
"""
General Variables
"""
# Store the highest fitness value achieved so far. Initialise to 0
maxFitness          = 0
# Count the number of generations that have been bred.
count               = 0
# Counts the number of generations since the best fitness last improved
noImprovementCount  = 0

# A flag used to notify the Genetic Algorithm that the best fitness is
# not improving and therefore to change the type of mutation being
# carried out to try and improve the best fitness
stuckFlag  = False
# Initialise an old population of all zeros. oldRankedPop is used to
# ensure that t
oldFitness = 0

if TEST:
    # If in test mode then load the saved population for testing
    popR=loadPop()
    pop,fitness=zip(*popR)
else:
    # Initialise an instance of the GA and generate a random
    # starting population
    GA = GeneticAlgorithm(layerVector, POPSIZE)
    pop=GA.generatePop()

# Evaluate the population and pair fitnesses and weights
pairedPop = pairPop(pop,layerVector)
# Rank the population in order of fitness
rankedPop = sorted(pairedPop, key=itemgetter(-1), reverse = True)
# Update the oldFitness container
oldFitness = rankedPop[0][-1]

# Save the inital population
savePop(rankedPop)

while count < MAXITER:

    # Generate the next generation
    pop = GA.nextGen(rankedPop, stuckFlag)

    # Evaluate the population and pair fitnesses and weights
    pairedPop = pairPop(pop,layerVector)

    # Rank the population in order of fitness
    rankedPop = sorted(pairedPop, key=itemgetter(-1), reverse = True)

    # If there is a new highest fitness then save the population
    if rankedPop[0][-1] > maxFitness:
        maxFitness = rankedPop[0][-1]
        savePop(rankedPop)
        print "Saved new Population with fitness: " + repr(rankedPop[0][-1]) + " Generation: " + repr(count)
    else:
        print "No improvements with generation " + repr(count)

    # If the goal has been reached then stop running
    if rankedPop[0][-1] >= targetFitness:
        break

    # If there has been no improvement in the maximum fitness of the
    # current generation and old generation then increase the count
    if rankedPop[0][-1] == oldFitness:
        noImprovementCount+=1
    else:
        stuckFlag=False
        noImprovementCount=0

    # If the count reaches a certain level then the Genetic Algorithm
    # needs to know that we are stuck. As a result the type of breeding
    # can be mixed up to try and force a change. The flag is switched
    # on and off in order to force maximum randmness in the output of
    # the Genetic Algorithm
    if noImprovementCount>10:
        stuckFlag=True
    if noImprovementCount>20:
        stuckFlag=False
        noImprovementCount=0

    # The oldFitness is used to monitor the variation/stagnation in the
    # Genetic Algorithm output
    oldFitness = rankedPop[0][-1]

    count += 1