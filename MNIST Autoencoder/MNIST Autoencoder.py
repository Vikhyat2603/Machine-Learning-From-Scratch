import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import mnist
import matplotlib.pyplot as plt
from time import time as t

np.random.seed(0)

print('Getting dataset')
#Get test images, labels(not needed) and ignore the test data
(X, y_labels), _ = mnist.load_data()

X = X.reshape(-1, 28*28) # Flatten each image
X = X/255 # Normalise to make values between 0 to 1
X_train = X[:20000] # Take first 20000 images for training data
y = X_train # Target data is same as the X data

# Make model - ANN Architecture : (784, 300, 10, 300, 784); Activation Function : relu
model = Sequential([Dense(300, activation='relu', input_dim=784),
                    Dense(10, activation='relu'),
                    Dense(300, activation='relu'),
                    Dense(784, activation='sigmoid')])

# Compile model - Optimizer : Adam; Loss : Binary Cross Entropy
model.compile(optimizer='adam', loss='binary_crossentropy')

print('Started training')
start = t()

# Train model - Train : validation split is 1:1; 15 epochs with batch_size of 250
history = model.fit(X_train, y, validation_split=0.5, epochs=15, batch_size=250, verbose=1)
print('Training over after', t()-start, 'seconds')

# Plot training & validation loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()


def displayImage(idx):
    '''Display an image along with the network's prediction for it'''
    testImg = X[idx].reshape(1,-1)
    predImg = model.predict(testImg)
    
    fig, axes = plt.subplots(2,1)
    axes[0].imshow(testImg.reshape(28,28))
    axes[1].imshow(predImg.reshape(28,28))
    
displayImage(4) # Change to any index from 0 to 59999
