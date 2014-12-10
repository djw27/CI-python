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
        self.weight = []

    def setWeights(self, weightIn):
        self.weight = weightIn
        self.NN = NeuralNet(self.layerVector, self.weight)

    def update(self, inputValues):
        return self.NN.update(inputValues)

Data = MyData()

def equip_car(pod):

    global Data
    Data = MyData()

    sensors=[ Sensor(angle=FORWARDS_ANGLE,name="Forwards"),
              Sensor(angle=LEFT_ANGLE/5,name="Eigth-Left"),
              Sensor(angle=LEFT_ANGLE/3,name="Quarter-Left"),
              Sensor(angle=RIGHT_ANGLE/5,name="Eigth-Right"),
              Sensor(angle=RIGHT_ANGLE/3,name="QuarterRight"),
            ]

    pod.addSensors(sensors)

    #pop = loadPop()
    #weights,fitness=zip(*pop)
    #text_file = open("Output.txt", "w")
    #text_file.write(str(weights[0]))
    #text_file.close()

    if int(pod.sensors[0].val) == 151:
        #print "carCircuit Detected"
        weightIn = [[[0.06866547795327779, 0.11144801140385793, 0.6223089846713334, -0.97499448575394, -0.25708296654489105], [-0.1982519263221387, -0.09226168251808448, 0.1665767897010385, -0.040991475852538, 0.8311864428779787], [-0.16460950542524386, -0.37857775575901953, 0.23937829363435018, -0.12660937610360018, 0.4137266515712257], [0.6276180244688228, 0.8175033415513987, -0.15344727416926376, 0.9184082103277633, 0.23861451790443958], [-0.728545314723576, -0.2959210653299112, -0.6696675880949834, -0.016388067410802884, -0.8476772183512125], [-0.28020848703213763, 0.7391661181191495, -0.8373650016675228, 0.031352566910217705, 0.2441355421940482], [0.6243503757484972, 0.9871327366448147, 0.16642182863932056, 0.4660118487580872, 0.4073755532087496], [0.3882858777093686, -0.04443460918681701, 0.2978528529490344, -0.3420140039960344, 0.6468347395888513]], [[0.9230047639113808, -0.950666415990008, -0.7418938768618866, 0.288140054598376], [1.0114223012103583, -0.08696974684018817, -0.21937763547786138, 0.07608478008025386], [0.7533482341459188, -0.6244157798219151, -0.5460585979247842, -0.3061294141981798], [0.3216492271534129, 0.23675860497468962, -0.83794642409949, 0.9521720364929847], [0.04148497003081503, 0.6765410463854783, 0.466164170595486, -0.4084463195638502]]]
    elif int(pod.sensors[0].val) == 248:
        #print "round Detected"
        weightIn = [[[0.04151645566403746, -0.05498516926198404, 0.5554401750686947, -0.929342910233601, -0.1577405284938155], [-0.20463333897999106, -0.10735042793631995, 0.002570997113094109, -0.05836146229859784, 0.9084880982138378], [-0.25757535170755513, -0.263443864578521, 0.22668932944844183, -0.056936571907964986, 0.48023271385365457], [0.7388703943973197, 0.7102910821976746, 0.06829680248137131, 0.8562579017333802, 0.3258893574342656], [-0.8706422528082591, -0.14686549042323802, -0.692837863874162, -0.006193857129844615, -0.8744171204877932], [-0.2171093950781897, 0.5870221675627624, -0.7923306545349627, -0.0492694061021292, 0.19263332654282653], [0.714341362878397, 0.906527309316677, 0.04271707411000142, 0.48280927128480355, 0.35211985692595943], [0.18218093929602094, 0.09323709358337462, 0.2361542401591827, -0.1838123495627075, 0.7939462026501695]], [[1.1715837859522067, -1.0180164685089024, -0.6794695855541864, 0.2336358726512687], [0.8221177459117912, 0.21204532996471853, 0.00606433802852633, -0.3905371451637919], [0.7425280883836409, -0.7313336645224282, -0.5641770866265953, -0.38622434368055414], [0.42518000748369095, 0.03782455329058794, -0.8778169165884087, 0.8592792541555044], [-0.1106134947888105, 0.7869647655606847, 0.7591872399302029, -0.6504877763050507]]]
    elif int(pod.sensors[0].val) == 604:
        #print "long Detected"
        weightIn = [[[0.06894969525707852, -0.1679298101332454, 0.6022722272759303, -1.2008273046768954, -0.08960813889698921], [-0.1836503648742823, -0.11028968577352255, 0.06345180323078538, -0.08534109663842473, 0.956125299786462], [-0.3579977941543025, -0.42561186885945457, 0.23589791965655266, -0.07020528319354101, 0.41038891580374165], [0.7613168343182316, 0.7385203916325169, 0.025203832191146795, 0.7100948819208013, 0.4590003948397554], [-0.778710983203398, -0.17262911804547143, -0.6791498825914797, -0.051950031626589725, -0.7550897300913113], [-0.259322558431866, 1.05453921094532, -0.570354706371933, 0.028872102952094292, 0.28566329173929345], [0.8805242082217037, 1.0748524324018398, 0.44677287102896346, 0.5071737656484856, 0.2609801853960867], [0.20252650498774036, 0.07631036354340033, 0.3826623859938525, -0.4336254973103804, 0.5944999637646777]], [[1.000031400064087, -0.8790226134756111, -0.7623516389366205, 0.23991526568826554], [0.7303134180713998, -0.11894958642673967, -0.16425970852523733, -0.03721420491148241], [0.8751579695696189, -0.7278059608793703, -0.4253170862429988, -0.5704897405773008], [0.5233295368507885, 0.012161936448057409, -0.7823727849884383, 1.1178116099684003], [-0.2132783250042379, 0.7535756252815619, 0.5485047217831, -0.44557911250228843]]]
    elif int(pod.sensors[0].val) == 172:
        #print "chick Detected"
        weightIn = [[[0.08860507841353929, 0.034876261719728366, 0.4859710939561501, -0.929342910233601, -0.11294153939579282], [-0.4564392865772603, -0.1723098145330048, 0.19068161726002827, 0.029928226050473537, 0.8220091663307864], [-0.2877238033035588, -0.403312038114011, 0.5094474272184969, -0.06783811460127219, 0.33733447070925837], [0.6042615314109112, 0.85766079964457, -0.05614160018495842, 0.8875335895869965, 0.2011411624230198], [-0.8559724847302415, -0.26781236411993115, -0.7345156476369362, -0.04832438894214637, -0.7014085575161186], [-0.15948632127160534, 0.8186197357101606, -0.8959784990099322, 0.031352566910217705, 0.34650802377855094], [0.7337180803381608, 1.0101135564231867, -0.17316462815049544, 0.482679287871061, 0.39929391796274494], [0.38673156385255986, -0.08137003195096804, 0.271256667819736, -0.10204757492011916, 0.6560298500620967]], [[0.8806650692506403, -1.029575205683781, -0.5468673709098264, -0.004243348348071607], [0.9472895136336369, -0.04616396126747389, -0.20944775905224614, 0.048624626794493406], [0.719156530829043, -0.6917137160792097, -0.43236710592395955, -0.4437502668230735], [0.4786950198279092, 0.08152296307234333, -0.9267780761296213, 0.9218246220276474], [0.0339656624222638, 0.7000042445763432, 0.6132688306208604, -0.4084463195638502]]]
    else:
        #print "New track Detected!"
        weightIn = [[[0.008428211023133594, -0.001668821632506333, 0.4746812298495242, -0.929342910233601, -0.24685285664336876], [-0.281364285200693, -0.1405467616734094, -0.003179189672476645, -0.040991475852538, 0.9106647185513694], [-0.2675850378031022, -0.33246904936288785, 0.3280008455663596, 0.013646831023984264, 0.4391541526709142], [0.7139177318330789, 0.8397324860046375, -0.10058522766264952, 0.9156590538952044, 0.196012820673353], [-0.8597204347915001, -0.18444679892650134, -0.6863980617801682, -0.03859548842287741, -0.8476772183512125], [-0.19568169067655047, 0.7874179306815358, -0.7632604787258707, 0.031352566910217705, 0.19263332654282653], [0.7550960002716901, 0.9358687532631965, 0.112224429721129, 0.4535898469539765, 0.3981873499987813], [0.29959997993768395, -0.04443460918681701, 0.27809236300821955, -0.21248344017168036, 0.6477393319910825]], [[0.9230047639113808, -0.9310368236515731, -0.6328786584776744, 0.21945320299759197], [0.8324735040542348, -0.045508124206933426, -0.16977874469357845, -0.016306394256950588], [0.7821013939240752, -0.7692557841504156, -0.5460585979247842, -0.37801910818426243], [0.508914829474565, 0.08249005982899177, -0.794801663453934, 0.9218246220276474], [-0.0155174627860531, 0.698845757318076, 0.6443757023243802, -0.4084463195638502]]]

    Data.setWeights(weightIn)

    pod.col=(204,255,225)
    pod.data= [0,0,0,0,0,0,0]
    pod.poly=[(-20*random.random(),-20*random.random()),(-20*random.random(),20*random.random()),(20*random.random(),20*random.random()),(20*random.random(),-20*random.random())]


def controller(pod,control):
    """
    if pod.state.age < 0.05:

            """




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

def loadPop(filename="popSaved045_long.dat"):
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
