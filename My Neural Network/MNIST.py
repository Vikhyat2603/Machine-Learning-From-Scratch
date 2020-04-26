import numpy as np
import datetime
from time import time as t
import activationFunctions as aF
import matplotlib.pyplot as plt
from mnist import MNIST
from NeuralNetwork import NeuralNetwork

# =============================================================================
# ################################# Get Data ##################################
# =============================================================================

mndata = MNIST(r'.\mnist-files')
MNISTimages, MNISTclasses = mndata.load_training()

trainLength = 2500
validationLength = 500
totalLength = len(MNISTimages)

inputs = np.array(MNISTimages)
classes = np.array(MNISTclasses)

# One Hot Encoding
indices = np.vstack([range(totalLength), MNISTclasses])
labels = np.zeros((totalLength, 10))
labels[indices[0], indices[1]] = 1

trainInputs = inputs[:trainLength]
trainLabels = labels[:trainLength]
trainClasses = classes[:trainLength]

validationInputs = inputs[trainLength: trainLength + validationLength]
validationLabels = labels[trainLength: trainLength + validationLength]
validationClasses = classes[trainLength: trainLength + validationLength]

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

def getAccuracy(predictedLabels, actualLabels):
    ''' return % of correct predictions given (predicted, actual)'''
    return (predictedLabels.round() == actualLabels).mean()*100

def getRMSE(x, y):
    return float(np.sqrt(((x - y)**2).mean()))

# =============================================================================
# ############################# Hyperparameters ###############################
# =============================================================================

np.random.seed(1)
layerLengths = (784,100,10)
hiddenActivator = aF.ReLu
outputActivator = aF.atanScaled
activators = [hiddenActivator]*(len(layerLengths)-2)+[outputActivator] # Create activators list

epochs = 50
alpha = 0.01 # Learning rate
beta = 0.9 # Momentum factor
miniBatchSize = 32
gradientUpdateType = 'ewa'

NN = NeuralNetwork(layerLengths, activators, 0.1) # Create Neural Network Object

# =============================================================================
# ############################### Train Network ###############################
# =============================================================================

trainLosses = [] # train data loss
validationLosses = [] # validation data loss
trainAccuracies = [] # number of correct matches with train data
validationAccuracies = [] # number of correct matches with validation data
reportInterval = epochs // 10 # Prints performance statistics at intervals

NN.setTrainingData(trainInputs, trainLabels, miniBatchSize) # Send training data to neural network object

print(f'Date & Time: {str(datetime.datetime.now())[:-7]}')
print('Starting training...\n')
startTime = t()

# Try-Finally exception handling used to allow keyboardInterrupt to interrupt training and still show final performance
try:
    for epoch in range(epochs+1):
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
        
        NN.miniBatchGD(alpha, gradientUpdateType, beta) # Runs training for one epoch
            
finally:
    print(f'\nTime taken: {t() - startTime}')

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
    
    plt.close('all')

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
    
################################################################
