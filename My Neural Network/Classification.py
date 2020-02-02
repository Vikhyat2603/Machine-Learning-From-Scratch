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
layerLengths = (2,4,3,1)
hiddenActivator = aF.ReLu
outputActivator = aF.atanScaled
activators = [hiddenActivator] * (len(layerLengths)-2) + [outputActivator] # Create activators list

epochs = 1000
alpha = 0.02 # Learning rate
beta = 0.9 # Momentum factor
batchSize = 7
momentumEnabled = True

NN = NeuralNetwork(layerLengths, activators) # Create Neural Network Object

# =============================================================================
# ################################# Get Data ##################################
# =============================================================================

dataCreator.pointCount = 2000
dataCreator.noise = 12
dataCreator.pointSeperation = 20

inputs, labels = dataCreator.createData('moons')
labels = labels.reshape(-1, 1)

trainLength = 750
validationLength = 250
totalLength = len(inputs)

trainInputs = inputs[:trainLength]
trainLabels = labels[:trainLength]

validationInputs = inputs[trainLength: trainLength + validationLength]
validationLabels = labels[trainLength: trainLength + validationLength]

NN.setTrainingData(trainInputs, trainLabels, batchSize) # Send training data to neural network object

# =============================================================================
# ######################## Data Processing Functions ##########################
# =============================================================================

livePlot = True # Live Plot binary classification progress
plotData = True

# Prepare live plot grid
if livePlot or plotData:
    
    minX, minY = inputs.min(axis = 0)
    maxX, maxY = inputs.max(axis = 0)
    grid = np.array(list(product(np.linspace(minX, maxX, 40), np.linspace(minY, maxY, 40))))
    
    fig = plt.figure(figsize = (4,4))
    liveAx = fig.add_subplot(1, 1, 1)

##def adjustPlotColour(x): return aF.atan(5*(x-0.5))/2.4+0.5
def adjustPlotColour(x): return x

def plotLive(inputs, predicted):
    liveAx.scatter(inputs[:,0], inputs[:,1], c = dataCreator.cpick.to_rgba(adjustPlotColour(predicted)))

def getCorrectCount(predictedLabels, actualLabels):
    ''' return number of correct predictions given (predicted, actual)'''
    return (predictedLabels.round() == actualLabels).sum()

def getRMSE(x, y):
    return float(((x-y)**2).mean()**0.5)

# =============================================================================
# ############################### Train Network ###############################
# =============================================================================

trainLosses = [] # train data loss
validationLosses = [] # validation data loss
correctTrain = [] # number of correct matches with train data
correctValidation = [] # number of correct matches with validation data
reportInterval = epochs // 25 # Prints performance statistics at intervals and plots if livePlot is on (decrease for continous plotting)

print('Date & Time:',str(datetime.datetime.now()).split('.')[0])
print('Starting training...\n')
startTime = t()

# Try-Finally exception handling used to allow keyboardInterrupt to interrupt training and still show final performance
try:
    for epoch in range(epochs):
        NN.batchEvaluation(alpha, momentumEnabled, beta)  # Runs training for one epoch
        
        if (epoch % reportInterval) == 0:
            # Print (and record) performance statistics
            print('Epoch:', epoch, '| Date & Time:', str(datetime.datetime.now()).split('.')[0])
            
            predictedLabels = NN.feedForward(trainInputs.T).T
            RMSE = getRMSE(predictedLabels, trainLabels)
            correctResultCt = getCorrectCount(predictedLabels, trainLabels)
            correctPct = round(correctResultCt/trainLength*100, 2)
            
            trainLosses.append(RMSE)
            correctTrain.append(correctPct)
            
            print('\tTrain Data      - RMSE:{0:.6f}'.format(round(RMSE, 6)), '| Correct:', correctResultCt, '/', trainLength, ' - ', correctPct,'%')
    
            predictedLabels = NN.feedForward(validationInputs.T).T
            RMSE = getRMSE(predictedLabels, validationLabels)       
            correctResultCt = getCorrectCount(predictedLabels, validationLabels)
            correctPct = round(correctResultCt/validationLength*100, 2)
            
            validationLosses.append(RMSE)
            correctValidation.append(correctPct) 
            
            print('\tValidation Data - RMSE:{0:.6f}'.format(round(RMSE, 6)), '| Correct:', correctResultCt, '/', validationLength, ' - ', correctPct,'%')
              
            if livePlot:
                plotLive(grid, NN.feedForward(grid.T).flatten())
                plt.pause(1e-8)
                               
finally:
    print('\nTime taken:', round(t() - startTime, 6))

    # Print (and plot) performance statistics
    predictedLabels = NN.feedForward(inputs.T).T
    RMSE = getRMSE(predictedLabels, labels)
    correctResultCt = getCorrectCount(predictedLabels, labels)
    correctPct = round(correctResultCt/totalLength*100, 2)
    
    print('Complete Data - RMSE:{0:.6f}'.format(round(RMSE,6)), '| Correct:', correctResultCt, '/', totalLength, ' - ', correctPct,'%')

    if momentumEnabled:
        print('Epochs:', epoch+1, ' |With Momentum|', '  Alpha:', alpha, '  Beta:', round(beta,6), '  Batch Size:', batchSize)
    else:
        print('Epochs:', epoch+1, ' |No Momentum|', '  Alpha:', alpha, '  Batch Size:', batchSize)
    
    print('Architecture:', layerLengths, '   Hidden Activator:', hiddenActivator.__name__, '   Output Activator:', outputActivator.__name__)
    
    if plotData:
        dataCreator.plotPoints(inputs, labels.reshape(totalLength), False)
        dataCreator.plotPoints(inputs, predictedLabels.flatten().round(), False)
        dataCreator.plotPoints(inputs, predictedLabels.flatten(), False)
        dataCreator.plotPoints(grid, NN.feedForward(grid.T).flatten(), True)
        
        plt.plot(trainLosses, '-o', c = 'r', linewidth = 0.4, markersize = 1.5, label = 'Train RMSE')
        plt.plot(validationLosses, '-o', c = 'b', linewidth = 0.4, markersize = 1.5, label = 'Validation RMSE')
        plt.legend(loc = "upper right")
        plt.show()
            
        plt.plot(correctTrain, '-o', c = 'r', linewidth = 0.4, markersize = 1.5, label = 'Train Correct %')
        plt.plot(correctValidation, '-o', c = 'b', linewidth = 0.4, markersize = 1.5, label = 'Validation Correct %')
        plt.legend(loc = "lower right")
        plt.show()
        
####################################################################
