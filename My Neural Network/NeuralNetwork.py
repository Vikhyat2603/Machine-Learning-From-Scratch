'''Contains Neural Network class'''
import numpy as np

class NeuralNetwork:
    '''The Neural Network class'''
    np.random.seed(0)
    def __init__(self, layerLengths, activators):
        '''layerLengths, activator names'''
        self.layerLengths = layerLengths
        self.depth = len(layerLengths)
        self.activators = [0] + activators # Extra blank activator to adjust indices
        self.weightGen = list(zip(layerLengths[1:], layerLengths[:-1])) # shape of weights[i]: (L[i],L[i-1])
        
        self.zeroBiases = np.array([np.zeros((l, 1)) for l in layerLengths[1:]]) # 0s in shape of biases
        self.zeroWeights = np.array([np.zeros((r, c)) for r, c in self.weightGen]) # 0s in shape of weights
        
        self.biases = self.zeroBiases
        # Normal weight distribution with Xavier intialisation
        self.weights = np.array([np.random.normal(0, 0.0625, (r, c))* np.sqrt(2/(r+c)) for r, c in self.weightGen]) 
        
        ## (Initially zero weights and biases from previous iteration for Momentum
        self.prevDeltaB = self.zeroBiases
        self.prevDeltaW = self.zeroWeights  

    def feedForward(self, i):
        '''feeds forward i, stores individual layerInputs and z values as attributes and
        returns the output; give input of shape: (n_inputs, any batchSize)'''
        self.zValues = [0] # element put to adjust indices (input layer(layer 0) has no z values)
        self.layerInputs = [i] # element put to adjust indices (input layer(layer 0) has no input values)
        for activator, w, b in zip(self.activators[1:], self.weights, self.biases):
            self.layerInputs.append(i)
            i = np.dot(w, i)+b
            self.zValues.append(i)
            i = activator(i)
        return i

    def backPropogate(self, i, target):
        '''Calculates bias and weight changes needed layer-wise, for input i, target and learning rate alpha'''
        output = self.feedForward(i.T).T
        error = (output-target).T / self.layerLengths[-1]
        deltaB = []
        deltaW = []
        for layerID in range(self.depth-1, 0, -1):
            activPrime = self.activators[layerID](self.zValues[layerID], True)
            delta = error * activPrime 
            deltaB.append(np.sum(delta, axis = 1).reshape(-1,1))
            deltaW.append(np.dot(delta, self.layerInputs[layerID].T))
            error = np.dot(self.weights[layerID-1].T, delta) / self.layerLengths[layerID-1]
        return np.array(deltaB[::-1]), np.array(deltaW[::-1])
    
    def updateParameters(self, b_delta, w_delta, alpha):
        '''uses b_delta and w_delta and updates biases and weights'''
        self.biases -= b_delta * alpha / self.batchSize
        self.weights -= w_delta * alpha / self.batchSize
        
    def setTrainingData(self, inputs, targets, batchSize):
        '''Stores training data; give inputs, targets, batchSize '''
        assert len(inputs) == len(targets), 'Data inputs length should be equal to data outputs length'
        batches = len(inputs) // batchSize
        batchesInputs  = np.array(np.array_split(inputs, batches))
        batchesTargets = np.array(np.array_split(targets, batches))

        self.batchSize = batchSize
        self.zipData = list(zip(batchesInputs, batchesTargets))
    
    def batchEvaluation(self, alpha, momentumEnabled = False, beta = 0.9):
        '''runs backpropogation for every training batch; give learning rate, momentum enabled (True/False), momentum factor'''
        totalDeltaB = self.zeroBiases.copy() # Bias list to store total desired bias change across batches
        totalDeltaW = self.zeroWeights.copy() # Weights list to store total desired weights change across batches

        for batchInput, batchTarget in self.zipData:
            deltaValues, deltaW = self.backPropogate(batchInput, batchTarget)
            totalDeltaB += deltaValues
            totalDeltaW += deltaW
                
        if momentumEnabled:
            # Add previous gradients if momentum is enabled
            totalDeltaB = (1-beta) * totalDeltaB + beta * self.prevDeltaB
            totalDeltaW = (1-beta) * totalDeltaW + beta * self.prevDeltaW

            self.prevDeltaB = totalDeltaB
            self.prevDeltaW = totalDeltaW

        self.updateParameters(totalDeltaB,  totalDeltaW, alpha)
