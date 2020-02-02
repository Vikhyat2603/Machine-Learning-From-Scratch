import numpy as np
import datetime
from time import time as t
import matplotlib.pyplot as plt
from NeuralNetwork import NeuralNetwork
import activationFunctions as aF

# =============================================================================
# ############################# Hyperparameters ###############################
# =============================================================================

np.random.seed(1)
layerLengths = (1,3,4,1)
hiddenActivator = aF.ReLu
outputActivator = aF.identity
activators = [hiddenActivator]*(len(layerLengths)-2)+[outputActivator] # Create activators list

epochs = 7500
alpha = 0.001 # Learning rate
beta = 0.9 # Momentum factor
batchSize = 10
momentumEnabled = True

NN = NeuralNetwork(layerLengths, activators) # Create Neural Network Object

# =============================================================================
# ################################# Get Data ##################################
# =============================================================================

def f(x): return 2*x - x*x

totalLength = 200

inputs = np.linspace(-8, 9, totalLength).reshape(-1,1)
np.random.shuffle(inputs)

targets = f(inputs)

dataRange = targets.max() - targets.min()

trainLength = 50
validationLength = 50

trainInputs = inputs[:trainLength]
trainTargets = targets[:trainLength]

validationInputs = inputs[trainLength: trainLength + validationLength]
validationTargets = targets[trainLength: trainLength + validationLength]

# Send training data to neural network object (& split according to batchSize)
NN.setTrainingData(trainInputs, trainTargets, batchSize) 

# =============================================================================
# ######################## Data Processing Functions ##########################
# =============================================================================

livePlot = True # Live Plot slows down training
plotData = True

# Prepare live plot grid
if livePlot:
    fig = plt.figure()
    liveAx = fig.add_subplot(1, 1, 1)

def getNRMSE(pred, targ):
    return float(((pred-targ.T)**2).mean()**0.5 / dataRange)

def plot_live(x, y):
    liveAx.clear()
    liveAx.scatter(x, y, c = 'r', label = 'Predicted')
    liveAx.scatter(inputs, targets, c = 'g', marker='.', label = 'Data')

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    liveAx.legend(by_label.values(), by_label.keys())

# =============================================================================
# ############################### Train Network ###############################
# =============================================================================

trainLosses = [] # train data loss
validationLosses = [] # validation data loss
reportInterval = epochs // 150 # Prints performance statistics at intervals and plots if livePlot is on (decrease for continous plotting)

print('Date & Time:',str(datetime.datetime.now()).split('.')[0])
print('Starting training...\n')
startTime = t()

# Try-Finally exception handling used to allow keyboardInterrupt to interrupt training and still show final performance
try:   
    for epoch in range(epochs):
        NN.batchEvaluation(alpha, momentumEnabled, beta) # Runs training for one epoch
        
        if (epoch % reportInterval) == 0:
            # Print (and record) performance statistics
            print('Epoch:', epoch, '| Date & Time:', str(datetime.datetime.now()).split('.')[0])
            
            predictedTargets = NN.feedForward(trainInputs.T)
            NRMSE = getNRMSE(predictedTargets, trainTargets)        
            trainLosses.append(NRMSE)

            print('\tTrain Data      - NRMSE:{0:.6f}'.format(round(NRMSE, 6)))
           
            predictedTargets = NN.feedForward(validationInputs.T)
            NRMSE = getNRMSE(predictedTargets, validationTargets)                    
            validationLosses.append(NRMSE)
            
            print('\tValidation Data - NRMSE:{0:.6f}'.format(round(NRMSE, 6)))

            if livePlot:
                plot_live(inputs.T, NN.feedForward(inputs.T).T)
                plt.pause(1e-8)
      
finally:
    print('\nTime taken:', round(t() - startTime, 6))

    # Print (and plot) performance statistics
    predictedTargets = NN.feedForward(inputs.T)
    NRMSE = round(getNRMSE(predictedTargets, targets), 6)
    print('Final NRMSE:', NRMSE)
    
    if momentumEnabled:
        print('Epochs:', epoch+1, ' |With Momentum|', '  Alpha:', alpha, '  Beta:', round(beta,6), '  Batch Size:', batchSize)
    else:
        print('Epochs:', epoch+1, ' |No Momentum|', '  Alpha:', alpha, '  Batch Size:', batchSize)
    
    print('Architecture:', layerLengths, '   Hidden Activator:', hiddenActivator.__name__, '   Output Activator:', outputActivator.__name__)

    plt.clf()
    if plotData:
        plt.plot(inputs, predictedTargets.T, 'o', c = 'r', label = 'Predicted')
        plt.plot(inputs, targets, 'o', c = 'g', marker='.', label = 'Data')
        plt.legend()
        plt.show()
        
        plt.plot(trainLosses, '-o', c = 'r', linewidth = 0.4, markersize = 1.5, label = 'Train RMSE')
        plt.plot(validationLosses, '-o', c = 'b', linewidth = 0.4, markersize = 1.5, label = 'Validation RMSE')
        plt.legend()
        plt.show()
    
####################################################################
