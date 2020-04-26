'''Contains Neural Network class'''
import numpy as np

class NeuralNetwork:
    '''The Neural Network class'''
    def __init__(self, layerLengths, activators, initialWeightsScale=1):
        '''layerLengths, activator names, scaling term for weights'' stdev'''
        self.layerLengths = layerLengths
        self.depth = len(layerLengths) # Network depth (including input layer)
        self.activators = [0] + activators # Extra blank activator to adjust indices
       
        self.biasShapes = [(l, 1) for l in layerLengths[1:]] # shape of biases: (L[i+1],1)
        self.weightShapes = list(zip(layerLengths[1:], layerLengths[:-1])) # shape of weights[i]: (L[i+1],L[i])

        self.zeroBiases = [np.zeros(biasShape) for biasShape in self.biasShapes] # 0s in shape of biases
        self.zeroWeights = [np.zeros((r, c)) for r, c in self.weightShapes] # 0s in shape of weights
        
        # Initialise biases as 0
        self.biases = self.zeroBiases 
        
        # Xavier Normal intialisation for weights, scaled by the 'initialWeightsScale' parameter
        self.weights = [np.random.normal(0, initialWeightsScale * np.sqrt(2/(r+c)), (r, c)) for r, c in self.weightShapes]
 
    def setTrainingData(self, inputs, targets, miniBatchSize):
        '''Stores training data; give inputs (dataSetLength, networkInputSize),
        targets (dataSetLength, networkOutputSize), miniBatchSize '''
        
        assert len(inputs) == len(targets), 'Data inputs length should be equal to data outputs length'
        self.inputs = inputs
        self.targets = targets
        self.miniBatchSize = miniBatchSize
        self.shuffleData()
        
    def shuffleData(self):
        batchSize = len(self.inputs)
        
        self.indices = np.arange(batchSize)
        np.random.shuffle(self.indices)
        
        miniBatches = batchSize // self.miniBatchSize # Number of mini-batches
        
        miniBatchInputs  = np.array_split(self.inputs[self.indices].T, miniBatches, axis = 1) # Split inputs into mini-batches
        miniBatchTargets = np.array_split(self.targets[self.indices].T, miniBatches, axis = 1) # Split targets into mini-batches

        self.zipData = list(zip(miniBatchInputs, miniBatchTargets))
        
    def feedForward(self, i):
        '''feeds forward i, stores individual layerInputs and z values as attributes and
        returns the output; give input of shape: (n_inputs, any batchSize)'''
        
        self.zValues = [0] # element put to adjust indices (input layer(layer 0) has no z values)
        self.layerInputs = [i] # element put to adjust indices (input layer(layer 0) has no input values)
        for activator, w, b in zip(self.activators[1:], self.weights, self.biases):
            self.layerInputs.append(i) # Cache previous layer activations
            i = np.dot(w, i)+b
            self.zValues.append(i) # Cache linear activation z = Wx + b
            i = activator(i)
        return i
    
    def lossFunction(self, output, target):
        '''Returns loss and d(loss)/d(output) given output, target'''
        return 0.5*(output-target)**2, (output-target)
    
    def backPropogate(self, i, target):
        '''Calculates bias and weight changes needed layer-wise, given input i, target (y) and miniBatchSize'''
        output = self.feedForward(i)
        miniBatchSize = len(i[0])
        # 'gradient' stores d(loss)/d(a[i]) throughout the function
        # Calculates loss and d(loss)/d(a[L])
        loss, gradient = self.lossFunction(output, target) 
        gradient /= self.layerLengths[-1] # Divide by layer length to distribute gradient across nodes
        
        deltaB = []
        deltaW = []
        for layerID in range(self.depth-1, 0, -1):
            activPrime = self.activators[layerID](self.zValues[layerID], True) # Finds d(a[i])/d(z[i])
            delta = (gradient * activPrime) # d(loss)/d(b[i]) = d(loss)/d(a[i]) * d(a[i])/d(b[i]) : Finds d(loss)/d(z[i])
            deltaB.append(np.mean(delta, axis = 1, keepdims = True)) # Gradient mean calculated across mini-batches
            deltaW.append(np.dot(delta, self.layerInputs[layerID].T) / miniBatchSize) # d(loss)/d(w[i]), takes mean across mini-batches
            gradient = np.dot(self.weights[layerID-1].T, delta) # Propagates gradient to previous layer, based on weights
            gradient /= self.layerLengths[layerID-1] # Divide by previous layer length to distribute gradient across nodes
            
        return deltaB[::-1], deltaW[::-1]
    
    def updateParameters(self, deltaB, deltaW, alpha, gradientUpdateType, beta, iteration):
        '''uses b_delta and w_delta and updates biases and weights, also takes alpha, updateType, beta, iteration'''
        
        if gradientUpdateType == 'ewa':
            # Exponentially Weighted Averages of gradients
            # prevTheta = (1-beta) * theta + beta * theta
            
            deltaB = [(1 - beta) * dB + beta * pdB for dB, pdB in zip(deltaB, self.prevDeltaB)]
            deltaW = [(1 - beta) * dW + beta * pdW for dW, pdW in zip(deltaW, self.prevDeltaW)]

            self.prevDeltaB = deltaB
            self.prevDeltaW = deltaW
            
            #Bias correction : 
            alpha /= 1 - beta**iteration
            
        elif gradientUpdateType == 'momentum':
            # Classic Momentum
            # prevTheta = theta + beta * prevTheta
            
            deltaB = [dB + beta * pdB for dB, pdB in zip(deltaB, self.prevDeltaB)]
            deltaW = [dW + beta * pdW for dW, pdW in zip(deltaW, self.prevDeltaW)]

            self.prevDeltaB = deltaB
            self.prevDeltaW = deltaW
        
        # Gradient Descent update
        # thetaPrime = theta - learning_rate * deltaTheta
        
        self.biases = [b - alpha * dB for b, dB in zip(self.biases, deltaB)]
        self.weights = [w - alpha * dW for w, dW in zip(self.weights, deltaW)]
           
    def miniBatchGD(self, alpha, gradientUpdateType=None, beta=0.9, shuffleData=True):
        '''runs backpropogation for every training mini-batch: learning rate, gradient update type-  
        'None for classic or 'momentum' or 'ewa'(Exponentially Weighted Averages), beta, shuffle mini-batches (True/False)'''     
        if shuffleData:
            self.shuffleData()
        
        ## Initially previous weights and biases as zero (for Momentum)
        self.prevDeltaB = self.zeroBiases
        self.prevDeltaW = self.zeroWeights
        
        for miniBatch, (miniBatchInput, miniBatchTarget) in enumerate(self.zipData, 1):
            deltaB, deltaW = self.backPropogate(miniBatchInput, miniBatchTarget)
            self.updateParameters(deltaB,  deltaW, alpha, str(gradientUpdateType).lower(), beta, miniBatch)
