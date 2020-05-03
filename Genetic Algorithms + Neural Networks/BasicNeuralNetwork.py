'''Contains Neural Network class'''
import numpy as np

class NeuralNetwork:
    '''The Neural Network class'''
    def __init__(self, layerLengths, activators, weightsScaleFactor=0.1, seed=0):
        np.random.seed(seed)
        '''layerLengths, activator names, scale factor for weights, random seed'''
        self.layerLengths = layerLengths
        self.depth = len(layerLengths) # Network depth (including input layer)
        self.activators = [0] + activators # Extra blank activator to adjust indices
        
        self.biasShapes = [(l, 1) for l in layerLengths[1:]] # shape of biases: (L[i+1],1)
        self.weightShapes = list(zip(layerLengths[1:], layerLengths[:-1])) # shape of weights[i]: (L[i+1],L[i])
        
        self.biases = [np.zeros(biasShape) for biasShape in self.biasShapes] # 0s in shape of biases
        # Normal weight distribution with Xavier intialisation scaled by initialWeightsStdev
        self.weights = [np.random.normal(0, weightsScaleFactor*np.sqrt(2/(r+c)), (r, c)) for r, c in self.weightShapes]
                
    def feedForward(self, i):
        '''feeds forward i, returns neural network output'''
        for activator, w, b in zip(self.activators[1:], self.weights, self.biases):
            i = activator(np.dot(w, i)+b)
        return i
    
    def updateParameters(self, deltaB, deltaW):                        
        self.biases = [b - dB for b, dB in zip(self.biases, deltaB)]
        self.weights = [w - dW for w, dW in zip(self.weights, deltaW)]