from operator import itemgetter, attrgetter
import math,random, copy, pickle
from pod.pods import Sensor,gui


FORWARDS_ANGLE = math.radians(0)
LEFT_ANGLE = math.radians(90)
RIGHT_ANGLE = math.radians(-90)

class MyData:

    def __init__(self):
        """
        NN Constants
        """
        self.inputNeurons = 7
        self.hiddenNeurons = 5
        self.outputNeurons = 4
        self.layerVector = [self.inputNeurons, self.hiddenNeurons, self.outputNeurons]
        # Select an example weight to process through the neural
        # net. In practice this would loop through the population.
        self.weight = [[[0.017778289668618275, 0.9533231184481528, 0.41976422829814997, -0.8607994397726295, -0.18218545212449633],
                        [-0.020832360519291236, 0.8248212256668508, -0.01520143224062287, -0.08806616618079072, 0.9898064504665314],
                        [-0.2788335347251807, -0.6176053132737875, 0.10476598318049368, -0.043363941782328594, 0.8987053474110993],
                        [0.7139177318330789, 0.032583872499533456, -0.11034816050616736, 0.8945059170256191, 0.16391383918186708],
                        [-0.8947118220023427, -0.7423609777586484, -0.05721381744177598, -0.03859548842287741, -0.8476772183512125],
                        [-0.24222522998176216, 0.8190624689263206, -0.8332771729090709, 0.031352566910217705, 0.18873483076032516],
                        [0.7651030949286468, 0.884377994541315, -0.5229332622028304, 0.4356921380267271, 0.4260300470034277],
                        [-0.6231155091271667, -0.1126392412527717, 0.004378904691721974, -0.24634736298935067, -0.09136557876280449]],
                        [[0.578844569974907, -0.9310368236515731, -0.6416738726010356, 0.383530684408661],
                        [0.9089312272284296, -0.045508124206933426, -0.1216854922082804, 0.06484663542685198],
                        [0.8104525820899298, -0.7927235481140718, 0.5987747250538167, -0.40638587706646123],
                        [0.5501878701556744, 0.08000524245140195, -0.8627505171857919, 0.8285991750252195],
                        [-0.05568797047323062, 0.6721012010789854, 0.6316960864329721, -0.41823177477729434]]]
        self.NN = NeuralNet(self.layerVector, self.weight)

    def update(self, inputValues):
        return self.NN.update(inputValues)


def equip_car(pod):


    sensors=[ Sensor(angle=FORWARDS_ANGLE,name="Forwards"),
              Sensor(angle=LEFT_ANGLE/5,name="Eigth-Left"),
              Sensor(angle=LEFT_ANGLE/3,name="Quarter-Left"),
              Sensor(angle=RIGHT_ANGLE/5,name="Eigth-Right"),
              Sensor(angle=RIGHT_ANGLE/3,name="QuarterRight"),
            ]

    pod.addSensors(sensors)
    pod.col=(204,255,225)
    pod.data= [0,0,0,0,0,0,0]
    pod.poly=[(-20*random.random(),-20*random.random()),(-20*random.random(),20*random.random()),(20*random.random(),20*random.random()),(20*random.random(),-20*random.random())]


def controller(pod,control):
    pod.poly=[(-20*random.random(),-20*random.random()),(-20*random.random(),20*random.random()),(20*random.random(),20*random.random()),(20*random.random(),-20*random.random())]

    Data = MyData()

    inputValues = [ pod.sensors[0].val,
                    pod.sensors[1].val,
                    pod.sensors[2].val,
                    pod.sensors[3].val,
                    pod.sensors[4].val,
                    pod.state.vel,
                    pod.state.slip,
                  ]
    pod.data = Data.update(inputValues)

    """
    Simplest possible control system. The NN calculates
    values for acceleration, braking and left/right for
    any given sensor values
    """

    control.up=pod.data[0]
    control.down=pod.data[1]
    control.left=pod.data[2]
    control.right=pod.data[3]

    # Launch control
    if pod.state.age < 0.2:
        control.up=1.0
        control.down=0.0


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
                frames_per_sec=200
                simple_gui=gui.SimpleGui(frames_per_sec=frames_per_sec,world=world,pods=[pod])

                fitness+=evaluate(pod, NN, simple_gui)
            else:
                fitness+=evaluate(pod, NN, 0)
        weights.append(weight)
        fitnesses.append(fitness)

    return zip(weights, fitnesses)