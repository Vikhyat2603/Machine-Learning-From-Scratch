import numpy as np
import datetime
from time import time as t
import activationFunctions as aF
import matplotlib.pyplot as plt
from mnist import MNIST
from NeuralNetwork import NeuralNetwork

# =============================================================================
# ############################# Hyperparameters ###############################
# =============================================================================

np.random.seed(1)
layerLengths = (784,100,10)
hiddenActivator = aF.ReLu
outputActivator = aF.atanScaled
activators = [hiddenActivator]*(len(layerLengths)-2)+[outputActivator] # Create activators list

epochs = 1000
alpha = 0.001 # Learning rate
beta = 0.9 # Momentum factor
batchSize = 30 
momentumEnabled = True

NN = NeuralNetwork(layerLengths, activators) # Create Neural Network Object

# =============================================================================
# ######################## Data Processing Functions ##########################
# =============================================================================

def displayAndCheck(index):
    '''Takes index, displays image at index and NN-predicted label'''
    img = inputs[index]
    
    imgStr = mndata.display(img).split('\n')
    for i in imgStr: print(i)
    
    res = NN.feedForward(img.reshape(784,1))
    print(res.argmax())

def getCorrectCount(predDataLabels, dataClasses):
    '''return number of correct predictions given labels(shape: (dataLength, 10)), "classes"(not one-hot encoded)'''
    return (predDataLabels.argmax(axis = 1) == dataClasses).sum()

def getRMSE(x, y):
    return float(np.sqrt(((x - y)**2).mean()))

# =============================================================================
# ################################# Get Data ##################################
# =============================================================================

mndata = MNIST(r'.\mnist-files')
MNISTimages, MNISTclasses = mndata.load_training()

trainLength = 2000
validationLength = 1000
totalLength = len(MNISTimages)

inputs = np.array(MNISTimages)
classes = np.array(MNISTclasses)

# One Hot Encoding
indices = np.vstack([range(totalLength),MNISTclasses])
labels = np.zeros((totalLength, 10))
labels[indices[0], indices[1]] = 1

trainInputs = inputs[:trainLength]
trainLabels = labels[:trainLength]
trainClasses = classes[:trainLength]

validationInputs = inputs[trainLength: trainLength + validationLength]
validationLabels = labels[trainLength: trainLength + validationLength]
validationClasses = classes[trainLength: trainLength + validationLength]

NN.setTrainingData(trainInputs, trainLabels, batchSize) # Send training data to neural network object

# =============================================================================
# ############################### Train Network ###############################
# =============================================================================

trainLosses = [] # train data loss
validationLosses = [] # validation data loss
correctTrain = [] # number of correct matches with train data
correctValidation = [] # number of correct matches with validation data
reportInterval = epochs // 10 # Prints performance statistics at intervals

print('Date & Time:',str(datetime.datetime.now()).split('.')[0])
print('Starting training...\n')
startTime = t()

# Try-Finally exception handling used to allow keyboardInterrupt to interrupt training and still show final performance
try:
    for epoch in range(epochs+1):
        NN.batchEvaluation(alpha, momentumEnabled, beta) # Runs training for one epoch

        if (epoch % reportInterval) == 0:
            # Print (and record) performance statistics
            print('Epoch:', epoch, '| Date & Time:', str(datetime.datetime.now()).split('.')[0])
            
            predictedLabels = NN.feedForward(trainInputs.T).T # Predict labels for training data
            RMSE = getRMSE(predictedLabels, trainLabels)   
            correctResultCt = getCorrectCount(predictedLabels, trainClasses)
            correctPct = round(correctResultCt/trainLength*100, 2)
            
            trainLosses.append(RMSE)
            correctTrain.append(correctPct)
            
            print('\tTrain Data      - RMSE:{0:.6f}'.format(round(RMSE, 6)), '| Correct:', correctResultCt, '/', trainLength, ' - ', correctPct,'%')
    
            predictedLabels = NN.feedForward(validationInputs.T).T # Predict labels for validation data
            RMSE = getRMSE(predictedLabels, validationLabels)        
            correctResultCt = getCorrectCount(predictedLabels, validationClasses)
            correctPct = round(correctResultCt/validationLength*100, 2)
            
            validationLosses.append(RMSE)
            correctValidation.append(correctPct) 
            
            print('\tValidation Data - RMSE:{0:.6f}'.format(round(RMSE, 6)), '| Correct:', correctResultCt, '/', validationLength, ' - ', correctPct,'%')   
            
finally:
    print('\nTime taken:', t() - startTime)

    # Print (and plot) performance statistics
    predictedLabels = NN.feedForward(inputs.T).T
    RMSE = getRMSE(predictedLabels, labels)
    correctResultCt = getCorrectCount(predictedLabels, classes)
    correctPct = round(correctResultCt/totalLength*100, 2)
    
    print('Complete Data - RMSE:{0:.6f}'.format(round(RMSE,6)), '| Correct:', correctResultCt, '/', totalLength, ' - ', correctPct,'%')
    
    if momentumEnabled:
        print('Epochs:', epochs, ' |With Momentum|', '  Alpha:', alpha, '  Beta:', round(beta,6), '  Batch Size:', batchSize)
    else:
        print('Epochs:', epochs, ' |No Momentum|', '  Alpha:', alpha, '  Batch Size:', batchSize)

    print('Architecture:', layerLengths, '   Hidden Activator:', hiddenActivator.__name__, '   Output Activator:', outputActivator.__name__)
    
    plt.plot(trainLosses, '-o', c = 'r', linewidth = 0.5, markersize = 1.5, label = 'Train RMSE')
    plt.plot(validationLosses, '-o', c = 'b', linewidth = 0.5, markersize = 1.5, label = 'Validation RMSE')
    plt.legend(loc = "upper right")
    plt.show()
        
    plt.plot(correctTrain, '-o', c = 'r', linewidth = 0.5, markersize = 1.5, label = 'Train Correct %')
    plt.plot(correctValidation, '-o', c = 'b', linewidth = 0.5, markersize = 1.5, label = 'Validation Correct %')
    plt.legend(loc = "lower right")
    plt.show()
    
################################################################
