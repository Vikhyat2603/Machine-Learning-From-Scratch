import numpy as np
import datetime
from time import time as t
import matplotlib.pyplot as plt
from NeuralNetwork import NeuralNetwork
import activationFunctions as aF

# =============================================================================
# ############################# Hyperparameters ###############################
# =============================================================================

np.random.seed(0)
layerLengths = (1,5,1)
hiddenActivator = aF.tanh
outputActivator = aF.tanh
activators = [hiddenActivator]*(len(layerLengths)-2)+[outputActivator] # Create activators list

epochs = 2000
alpha = 0.01 # Learning rate
beta = 0.9 # Momentum factor
miniBatchSize = 25
gradientUpdateType = 'ewa'

NN = NeuralNetwork(layerLengths, activators, 0.05) # Create Neural Network Object

# =============================================================================
# ################################# Get Data ##################################
# =============================================================================

def f(x): return np.sin(x/10)#2*x - x*x

totalLength = 500
trainLength = 200
validationLength = 50

inputs = np.linspace(-50, 30, totalLength).reshape(-1,1)
np.random.shuffle(inputs)
targets = f(inputs)

dataRange = targets.max() - targets.min()

trainInputs = inputs[:trainLength]
trainTargets = targets[:trainLength]

validationInputs = inputs[trainLength: trainLength + validationLength]
validationTargets = targets[trainLength: trainLength + validationLength]

# Send training data to neural network object (& split according to batchSize)
NN.setTrainingData(trainInputs, trainTargets, miniBatchSize) 

# =============================================================================
# ######################## Data Processing Functions ##########################
# =============================================================================

livePlot = True # Live Plot slows down training
plotData = True

plt.figure(0)

def getNRMSE(pred, targ):
    return float(((pred-targ.T)**2).mean()**0.5 / dataRange)

def plot_live(x, y):
    plt.clf()
    plt.scatter(x, y, c = 'r')
    plt.scatter(inputs, targets, c='g') # make dotted line plot
    plt.legend(['Predicted', 'Data'])
    plt.pause(1e-5)

# =============================================================================
# ############################### Train Network ###############################
# =============================================================================

trainLosses = [] # train data loss
validationLosses = [] # validation data loss
reportInterval = epochs // 20 # Prints performance statistics at intervals and plots if livePlot is on (decrease for continous plotting)

print('Date & Time:',str(datetime.datetime.now()).split('.')[0])
print('Starting training...\n')
startTime = t()

# Try-Finally exception handling used to allow keyboardInterrupt to interrupt training and still show final performance
try:   
    for epoch in range(epochs):
        NN.miniBatchGD(alpha, gradientUpdateType, beta) # Runs training for one epoch
        
        if (epoch % reportInterval) == 0:
            # Print (and record) performance statistics
            print(f'Epoch: {epoch} | Date & Time: {str(datetime.datetime.now())[:-7]}')
            
            predictedTargets = NN.feedForward(trainInputs.T)
            trainNRMSE = getNRMSE(predictedTargets, trainTargets)        
            trainLosses.append(trainNRMSE)
            
            predictedTargets = NN.feedForward(validationInputs.T)
            validationNRMSE = getNRMSE(predictedTargets, validationTargets)                    
            validationLosses.append(validationNRMSE)

            print(f'\t NRMSE - Train: {trainNRMSE:.6f} | Validation: {validationNRMSE:.6f}\n')

            if livePlot:
                plot_live(inputs, NN.feedForward(inputs.T).T)
                
finally:
    print('\nTime taken:', round(t() - startTime, 6))

    # Print (and plot) performance statistics
    predictedTargets = NN.feedForward(inputs.T)
    NRMSE = round(getNRMSE(predictedTargets, targets), 6)
    print('Total Data NRMSE:', NRMSE)
    
    if gradientUpdateType:
        print(f'Epochs: {epoch+1} - GD with {gradientUpdateType} | Alpha: {alpha} | Beta: {beta} | Mini-Batch Size: {miniBatchSize}')
    else:
        print(f'Epochs: {epoch+1} - Classic GD  | Alpha: {alpha} | Mini-Batch Size: {miniBatchSize}')
    
    print(f'Architecture: {layerLengths} | Hidden Activator: {hiddenActivator.__name__} | Output Activator: {outputActivator.__name__}')

    if plotData:    
        plt.figure(1)
        
        plt.plot(trainLosses, '-o', c = 'r', linewidth = 0.4, markersize = 1.5)
        plt.plot(validationLosses, '-o', c = 'b', linewidth = 0.4, markersize = 1.5)
        plt.legend(['Train RMSE', 'Validation RMSE'])
        plt.show()
    
####################################################################
