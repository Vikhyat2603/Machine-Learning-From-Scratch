# A flexible Neural Network Library based on NumPy

This is one of the first ML projects I worked on and it gave me a great insight into Neural Networks as I designed the complete structure and learning algorithm myself. (After understanding the concepts from 3Blue1Brown's Youtube Playlist)

## Files:
#### - NeuralNetwork.py
    Contains NeuralNetwork class; you can set the architecture and activators during initialisation, then give
    training data with desired batch size, which it will split the data into. To train the network on the data,
    call batchEvaluation which runs gradient descent with the training data, and provide the learning rate, whether
    or not you want Momentum to be implemented, and momentum factor.
    
#### - activationFunctions.py
    Contains activation functions in the form f(x, deriv) where x is the input and deriv is a boolean noting whether
    to return the derivative of the function f'(x) or just f(x).  
      
    Included functions:  
    - atan  
    - tanh  
    - sigmoid(x) = 1/(1+e^-x)  
    - ReLu(x) = max(0,x)  
    - identity(x) = x  
    - leakyReLu(x) = (x+0.95*abs(x))/2  
    - atanScaled(x) = atan(x)/π + 0.5 (scaled to give output from 0 to 1)  
    - ReXu(x) = 0.5 - sqrt(0.25 - x) if x<0 else x
    - createScaledFunction which takes a function and scales it by a given value (converts f(x) -> f(ax)), and can  
      be used to avoid overflow errors

#### - dataCreator.py
    Creates data for binary classification; 2 datasets are available: 'circular' - data seperable by a circle
    (inside/outside circle) and 'moons' - data in the shape of two arcs from scikit.datasets.make_moons. Points
    are created, labelled, and shuffled. Noise levels and number of data points can be given for both datasets

#### - Regression.py
    Uses the Neural Network for a regression task - example given: trying to fit data on the curve y = 2x - x^2.
    Prints the NRMSE(Normalsied Root Mean Square Error) for both training and validation data. If livePlot is on,
    it shows the network's progress in trying to fit the data.
    (livePlot slows down training and doesn't work on notebooks)

#### - Classification.py  
    Uses the Neural Network for a classification task. While training, prints the RMSE(Root Mean Square Error) for
    both training and validation data, along with accuracy percentage. If livePlot is turned on, it shows the
    network's progress with the binary classification by showing the network's output for a grid that is
    approximately the input space, in a coloured format from red to blue representing the network's confidence of
    whether it falls into the first class. (livePlot slows down training and doesn't work on notebooks)

#### - MNIST.py 
    Uses the Neural Network (with architecture 784,100,10) on the classic MNIST dataset.
    On training for ~1 minute with Momentum and 2k training images, the network acheives 85% accuracy on the
    whole 60k image dataset.

#### - mnist-files.rar
    Contains MNIST data in 4 ubyte files. Extract to ./mnist-files
    
### Dependencies:
    - NumPy for calculations
    - matplotlib, for visualisation
    - scikit to make moons dataset for classification
    - python-mnist to parse MNIST data files (https://github.com/sorki/python-mnist)
    
### Performance:
    View the performance folder for performance visualisations on classification and regression 

### Credits:
    3Blue1Brown's Neural Networks playlist with great intuitive and mathematical explanations on Youtube
