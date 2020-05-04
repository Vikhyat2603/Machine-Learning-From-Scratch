import numpy as np
from BasicNeuralNetwork import NeuralNetwork
import activationFunctions as aF
import matplotlib.pyplot as plt
from time import sleep
from time import time as t

'''
make random agents
loop for generations:
    for each agent:
        while game is not over:
            feedforward processed state into agent NN, perform predicted action and get rewards
        record agent fitness
        
    crossover - recreate population with each weight & bias chosen from previous agents probabilistically according to fitnesses
    mutation - add random weights & biases to agent to help exploration
'''

# =============================================================================
# Neural Network + Genetic Algorithm Functions
# =============================================================================

def softmax(fitnesses, scalefactor=1):
    ''' Given array of fitnesses (and scaling factor to avoid np.exp inaccuracies), scales values to range 0 to 1, 
    with total array sum being 1, to be used as probabilities'''
    # Inputs scaled by softmaxFactor term to avoid inaccuracies
    return np.exp(fitnesses*scalefactor) / np.exp(fitnesses*scalefactor).sum()

def createNN(stateDim, actionDim, hiddenLayerLengths = [], hiddenActivator = aF.ReLu, outputActivator = aF.identity, weightsScaleFactor=1):
    ''' Creates Neural Network with specified architecture, and initial weights scaled by weightsScaleFactor'''
    layerLengths = [stateDim] + hiddenLayerLengths + [actionDim]
    depth = len(layerLengths) - 1
    activators = [hiddenActivator]*(depth-1) + [outputActivator]
    
    return NeuralNetwork(layerLengths, activators, weightsScaleFactor)

def matToArr(mat):
    ''' Converts list of matrices(weights or biases) to one flat array '''
    return np.hstack([node for layer in mat for node in layer])

def arrToMat(arr, zipInfo):
    ''' Converts flat array to list of matrices according to given information '''
    mat = []
    i = 0
    for length, shape in zipInfo:
        mat.append(arr[i: i + length].reshape(shape))
        i += length
    return mat

def breed(agents, fitnesses, bestAgent):
    ''' Given agents and their fitnesses, recreates the population by choosing weights and biases from
    agents probabilistically according to fitnesses; the best agent is always chosen for the next generation'''
    
    # List of weights and biases for all agents
    agentWeights = [a.weights for a in agents]
    agentBiases = [a.biases for a in agents]
    
    # Array of flattened weights and biases for all agents
    agentWeightsArr = np.array(list(map(matToArr, agentWeights)))
    agentBiasesArr = np.array(list(map(matToArr, agentBiases)))
    
    # Lengths of an agent's flattened weights and biases
    weightLength = agentWeightsArr.shape[1]
    biasLength = agentBiasesArr.shape[1]
    
    # Calculate probability of choosing parameters from an agent, based on its fitness
    chosenProbability = softmax(fitnesses, softmaxFactor) 
    
    # Adds the best agent to the future population
    agents[0].weights = bestAgentWeights
    agents[0].biases = bestAgentBiases
    
    for agent in agents[1:]:
        # Crossover:
        # Probabilistically choose random indices to decide which agent to inherit each weight and bias from
        randomIndicesW = np.random.choice(populationSize, size=weightLength, p=chosenProbability)
        randomIndicesB = np.random.choice(populationSize, size=biasLength, p=chosenProbability)
        
        # Copy weights and biases from parent agents
        newWeightsArr = agentWeightsArr[randomIndicesW, range(weightLength)]
        newBiasesArr = agentBiasesArr[randomIndicesB, range(biasLength)]
    
        # Set agent's new weights and biases
        agent.weights = arrToMat(newWeightsArr, zipWeightInfo)
        agent.biases = arrToMat(newBiasesArr, zipBiasInfo)

def mutate(agent, mutationFactor):
    ''' Mutate agent's weights and biases by a normal distribution with stdev of mutationFactor'''
    # Mutate biases by adding random normal biases with stdev = mutationFactor
    delBiases = [np.random.normal(0, mutationFactor, biasShape) for biasShape in agent.biasShapes]
    # Mutate weights by adding weights in Xavier initialisation form, scaled by mutationFactor
    delWeights = [np.random.normal(0, mutationFactor*np.sqrt(2/(r+c)), (r, c)) for r, c in agent.weightShapes]
    
    agent.updateParameters(delBiases, delWeights)
  
# =============================================================================
# Training Hyperparameters and Game Setup
# =============================================================================

# =============================================================================
# generations = 100 # Generations of genetic algorithm
# populationSize = 20 # Number of agents
# initialWeightsScale = 5 # Scaling factor for agents weights during intialisation
# mutationFactor = 2 # Stdev of intial mutation, similar to initial exploration factor
# mutationFactorDecay = 0.97 # Decay constant for mutationFactor (decay for exploration factor)
# agentHiddenLayers = [] # Hidden layer lengths for agents
# 
# gameLength = 500 # Maximum steps in a game
# softmaxFactor = 200 # Scale softmax according to fitnesses to avoid inaccuracies in np.exp()
# 
# from CartPoleGame import CartPoleGame
# game = CartPoleGame(gameLength) # Creates game object
# =============================================================================

generations = 75 # Generations of genetic algorithm
populationSize = 10 # Number of agents
initialWeightsScale = 5 # Scaling factor for agents weights during intialisation
mutationFactor = 2 # Stdev of intial mutation, similar to initial exploration factor
mutationFactorDecay = 0.9 # Decay constant for mutationFactor (decay for exploration factor)
agentHiddenLayers = [3] # Hidden layer lengths for agents

gameLength = 200 # Maximum steps in a game
softmaxFactor = 120 # Scale softmax according to fitnesses to avoid inaccuracies in np.exp()

from MountainCarGame import MountainCarGame
game = MountainCarGame(gameLength) # Creates game object

# =============================================================================
# Agents Setup
# =============================================================================
      
stateDim, actionDim = game.stateDim, game.actionDim # get dimensions of states and actions

# Create Neural Network agents, input dimensions being stateDim, and output dimensions being actionDim
agents = []
for i in range(populationSize):
    agent = createNN(stateDim, actionDim, agentHiddenLayers, aF.atanScaled, aF.atanScaled, weightsScaleFactor=initialWeightsScale)
    agents.append(agent)

# Shapes of weights and biases of each layer
weightShapes = agent.weightShapes
biasShapes = agent.biasShapes

# Lengths of weights and biases of each layer   
weightLengths = [x[0]*x[1] for x in weightShapes]
biasLengths = agent.layerLengths[1:]

# Lengths and shapes of weights & biases of each layer, used to recreate them from flat arrays
zipWeightInfo = list(zip(weightLengths, weightShapes))
zipBiasInfo = list(zip(biasLengths, biasShapes))

# =============================================================================
# Training
# =============================================================================

recordFitnesses = [] # Store all agents' fitnesses over generations
start_time = t()
reportInterval = generations//5 # Interval for printing generation fitness

for generation in range(generations):
    # Get fitnesses of agents
    fitnesses = np.array([game.evaluateFitness(agent, seed=generation) for agent in agents]) 
        
    recordFitnesses.append(fitnesses)
    
    bestAgent = agents[fitnesses.argmax()] # Get agent with best fitness
    # Store best agent's parameters
    bestAgentWeights = bestAgent.weights 
    bestAgentBiases = bestAgent.biases
    
    breed(agents, fitnesses, bestAgent) # Re-create agent population
    
    # Mutate all agents by the  mutation factor
    for agent in agents:
        mutate(agent, mutationFactor)
        
    mutationFactor *= mutationFactorDecay # Decay the mutation factor

    # Print mean and max fitness every <reportInterval> generations
    if not (generation % reportInterval):
        meanFitness = fitnesses.mean()
        maxFitness = fitnesses.max()
        print(f'Generation: {generation:3d} - Fitness - Mean: {meanFitness:.3f} | Max: {maxFitness:.3f}')

# =============================================================================
# Performance
# =============================================================================

print(f'Time taken: {(t() - start_time):.2f} seconds')

# Calculate and plot mean and maximum fitnesses over generations

meanFitnesses = [r.mean() for r in recordFitnesses]
maxFitnesses = [r.max() for r in recordFitnesses]

plt.close('all')

plt.plot(meanFitnesses, c='r', label = 'Mean')
plt.plot(maxFitnesses, c='g', label = 'Max')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.ylim(-0.1, 1.1)
plt.title("Population Fitness")

plt.legend()
plt.show()

# Set best agent parameters to parameters before mutation
bestAgent.weights = bestAgentWeights
bestAgent.biases = bestAgentBiases

# Replay game for bestAgent in the last generation
fitness = game.evaluateFitness(bestAgent, render=True)
sleep(2)
game.game.close()

print(f'Best Agent Fitness: {fitness:.3f}') # Print best agent's fitness
