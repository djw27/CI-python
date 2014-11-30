import math,random, copy, pickle
import numpy

class NeuralNet(object):
    """
    Generalised class for
    generating a neural net
    """
    def __init__(self, layerVector, weights):
        # Number of input, hidden and output nodes
        self.layerVector = layerVector
        self.layerVector[0] = self.layerVector[0]+1 # Adding in the bias
        self.ni = self.layerVector[0]
        self.nh = self.layerVector[1]
        self.no = self.layerVector[2]
        # Set the input and output weights
        self.weights = weights
        self.wi = weights[0]
        self.wo = weights[1]
        # activations for nodes
        #self.ai = [1.0]*self.ni
        #self.ah = [1.0]*self.nh
        #self.ao = [1.0]*self.no


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
            randomizeMatrix(wi, -0.2, 0.2)
            randomizeMatrix(wo, -2.0, 2.0)
            # combine weights into a gene
            gene = [wi, wo]
            # add the gene to the population
            pop.append(gene)

        return pop

def randomizeMatrix(matrix, a, b):
    """
    This function will fill a given matrix with random values.
    """
    for i in range ( len (matrix) ):
        for j in range ( len (matrix[0]) ):
            matrix[i][j] = random.uniform(a,b)

def sigmoid(value):
    """
    Calculates the sigmoid .

    """

    try:
        value = 1.0 / (1.0 + math.exp(-value))
    except OverflowError:
        value = 0.0

    return value

"""
GA Constants
"""
popSize = 100
"""
NN Constants
"""
inputNeurons = 5
hiddenNeurons = 6
outputNeurons = 4
layerVector = [inputNeurons, hiddenNeurons, outputNeurons]

# Initialise an instance of the GA and generate a random
# starting population
GA = GeneticAlgorithm(layerVector, popSize)
pop=GA.generatePop()
#print pop[0][1]

# Select an example weight to process through the neural
# net. In practice this would loop through the population.
weights = pop[0] # This is a [[6x(5+1)],[6x4]] matrix
NN = NeuralNet(layerVector, weights)

inputValues=[1,2,3,4,5]
outputs = NN.update(inputValues)

