# Using Genetic Algorithms to train Neural Networks - built from scratch on NumPy

## Files:
#### - activationFunctions.py
    Contains various activation functions for the agent Neural Networks.
    
#### - BasicNeuralNetwork.py
    Contains NeuralNetwork class; you can set the architecture and activators during initialisation, along with
    a scaling factor for the weights. The function feedForward returns the network's prediction given inputs, and
    update parameters can be used to add delta values to the weights and biases.
    
#### - GymTrial.py
    Plays different OpenAI Gym game environments, with random action sampling
    
#### - CartPoleGame.py
    Encapsulates the CartPole game environment, and can be modified to create a custom fitness function for the game
    (here fitness function is defined simply as sum of rewards divided by max game length) Main method is
    evaluateFitness that takes an agent (Neural Network) and game environment seed as parameters and returns the
    agent's fitness for one game.
   
#### - MountainCarGame.py
    Encapsulates the MountainCar game environment, and can be modified to create a custom fitness function for the game
    (here fitness function is defined to incorporate max rightward position reached). Main method is evaluateFitness 
    that takes an agent (Neural Network) and game environment seed as parameters and returns the agent's fitness for
    one game.

#### - GeneticAlgo + NeuralNetwork.py
    Main code involving the genetic algorithm's functions and set up of the agents (Neural Networks). Everything that
    needs to be altered for different environments is under 'Training Hyperparameters and Game Setup'. Uncomment one of
    the two blocks of code to run the Genetic Algorithm for CartPole or MountainCar.    
   
### Dependencies:
    - NumPy for calculations
    - matplotlib, for visualisation
    - OpenAI gym for game environments
    
### Performance:
    View the performance folder for graphs of fitness along generations
