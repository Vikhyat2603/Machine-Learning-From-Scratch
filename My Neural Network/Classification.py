import numpy as np
import datetime
from time import time as t
import matplotlib.pyplot as plt
from itertools import product
import dataCreator
from NeuralNetwork import NeuralNetwork
import activationFunctions as aF

# =============================================================================
# ############################# Hyperparameters ###############################
# =============================================================================

np.random.seed(1)
layerLengths = (2,4,1)
hiddenActivator = aF.ReLu
outputActivator = aF.atanScaled
activators = [hiddenActivator] * (len(layerLengths)-2) + [outputActivator] # Create activators list

epochs = 100
alpha = 0.02 # Learning rate
beta = 0.9 # Momentum factor
miniBatchSize = 8
gradientUpdateType = 'momentum'

NN = NeuralNetwork(layerLengths, activators) # Create Neural Network Object

# =============================================================================
# ################################# Get Data ##################################
# =============================================================================

dataCreator.pointCount = 2000
dataCreator.noise = 5
dataCreator.pointSeperation = 4.5

inputs, labels = dataCreator.createData('circular')
labels = labels.reshape(-1, 1)

trainLength = 750
validationLength = 250
totalLength = len(inputs)

trainInputs = inputs[:trainLength]
trainLabels = labels[:trainLength]

validationInputs = inputs[trainLength: trainLength + validationLength]
validationLabels = labels[trainLength: trainLength + validationLength]

NN.setTrainingData(trainInputs, trainLabels, miniBatchSize) # Send training data to neural network object
# =============================================================================
# ######################## Data Processing Functions ##########################
# =============================================================================

livePlot = True # Live Plot binary classification progress
plotDataOn = True

# Prepare data plot grid
minX, minY = inputs.min(axis = 0)
maxX, maxY = inputs.max(axis = 0)
grid = np.array(list(product(np.linspace(minX, maxX, 40), np.linspace(minY, maxY, 40))))

if livePlot:
    fig = plt.figure(figsize = (4,4))
    liveAx = fig.add_subplot(1, 1, 1)

def plotData():
    plt.close('all')
    
    dataCreator.plotPoints(inputs, labels.reshape(totalLength), False)
    dataCreator.plotPoints(inputs, predictedLabels.flatten().round(), False)
    dataCreator.plotPoints(inputs, predictedLabels.flatten(), False)
    dataCreator.plotPoints(grid, NN.feedForward(grid.T).flatten(), False)

    fig = plt.figure(figsize = (8,4))
    ax1 = plt.subplot(121)
        
    ax1.plot(trainLosses, '-o', c = 'r', linewidth = 0.4, markersize = 1.5)
    ax1.plot(validationLosses, '-o', c = 'b', linewidth = 0.4, markersize = 1.5)
    ax1.legend(['Train RMSE', 'Validation RMSE'], loc = "upper right")
    ax1.set_ylim(0,1)

    ax2 = plt.subplot(122)
    
    ax2.plot(trainAccuracies, '-o', c = 'r', linewidth = 0.4, markersize = 1.5)
    ax2.plot(validationAccuracies, '-o', c = 'b', linewidth = 0.4, markersize = 1.5)
    ax2.legend(['Train Accuracy', 'Validation Accuracy'], loc = "lower right")
    ax2.set_ylim(0,100)

    plt.suptitle('RMSE and Accuracy')
    plt.show()
    
def plotLive(inputs, predicted):
    liveAx.scatter(inputs[:,0], inputs[:,1], c = dataCreator.cpick.to_rgba(predicted))
    plt.pause(1e-6)

def getAccuracy(predictedLabels, actualLabels):
    ''' return % of correct predictions given (predicted, actual)'''
    return (predictedLabels.round() == actualLabels).mean()*100

def getRMSE(x, y):
    return float(np.sqrt(((x - y)**2).mean()))

# =============================================================================
# ############################### Train Network ###############################
# =============================================================================

trainLosses = [] # train data loss
validationLosses = [] # validation data loss
trainAccuracies = [] # number of correct matches with train data
validationAccuracies = [] # number of correct matches with validation data
reportInterval = epochs // 10 # Prints performance statistics at intervals and plots if livePlot is on (decrease for continous plotting)

print('Date & Time:',str(datetime.datetime.now()).split('.')[0])
print('Starting training...\n')
startTime = t()

# Try-Finally exception handling used to allow keyboardInterrupt to interrupt training and still show final performance
try:
    for epoch in range(epochs):
        if (epoch % reportInterval) == 0:
            # Print (and record) performance statistics
            print(f'Epoch: {epoch} | Date & Time: {str(datetime.datetime.now())[:-7]}')
            
            predictedLabels = NN.feedForward(trainInputs.T).T
            trainRMSE = getRMSE(predictedLabels, trainLabels)
            trainAccuracy = getAccuracy(predictedLabels, trainLabels)
            
            trainLosses.append(trainRMSE)
            trainAccuracies.append(trainAccuracy)
            
            predictedLabels = NN.feedForward(validationInputs.T).T
            validationRMSE = getRMSE(predictedLabels, validationLabels)
            validationAccuracy = getAccuracy(predictedLabels, validationLabels)
            
            validationLosses.append(validationRMSE)
            validationAccuracies.append(validationAccuracy) 
            
            print(f'\tRMSE - Train: {trainRMSE:.6f} | Validation: {validationRMSE:.6f}')
            print(f'\tAccuracy - Train: {trainAccuracy:.2f}% | Validation: {validationAccuracy:.2f}%')
            
            if livePlot:
                plotLive(grid, NN.feedForward(grid.T).flatten())    
        
        NN.miniBatchGD(alpha, gradientUpdateType, beta)  # Runs training for one epoch
                               
finally:
    print('\nTime taken:', round(t() - startTime, 6))

    # Print (and plot) performance statistics
    predictedLabels = NN.feedForward(inputs.T).T
    RMSE = getRMSE(predictedLabels, labels)
    accuracy = getAccuracy(predictedLabels, labels)
    
    print(f'Complete Data - RMSE:{RMSE:.6f} | Accuracy: {accuracy:.2f}%')

    if gradientUpdateType:
        print(f'Epochs: {epoch+1} - GD with {gradientUpdateType} | Alpha: {alpha} | Beta: {beta} | Mini-Batch Size: {miniBatchSize}')
    else:
        print(f'Epochs: {epoch+1} - Classic GD  | Alpha: {alpha} | Mini-Batch Size: {miniBatchSize}')
    
    print(f'Architecture: {layerLengths} | Hidden Activator: {hiddenActivator.__name__} | Output Activator: {outputActivator.__name__}')
    
    if plotDataOn:
        plotData()
####################################################################
