import gym
import numpy as np

class MountainCarGame:       
    def __init__(self, gameLength=200):
        ''' Makes the OpenAI Gym MountainCar-v0 game environment'''
        self.game = gym.make("MountainCar-v0")
        self.stateDim, self.actionDim = self.game.observation_space.shape[0], 3
        self.game._max_episode_steps = gameLength 
        self.reset()
        
        # Get maximum possible state values (Not effective for environments with high but unlikely state values)
        self.stateExtreme = np.maximum(abs(self.game.observation_space.high), abs(self.game.observation_space.low))
            
    def reset(self, seed=0):
        self.game.seed(seed)
        self.game.action_space.seed(seed)
        self.state = self.game.reset()
        self.isOver = False
        
        # Additional State info needed for custom fitness function:
        self.maxPosition = -1 # Stores max position reached by the car
    
    def getState(self):
        ''' Gets current game state '''
        return self.state
    
    def recordState(self):
        ''' Records information about the state, used later in custom fitness function '''
        self.maxPosition = max(self.maxPosition, self.state[0]) # Max position (rightward) reached by car is kept track of
    
    def normaliseState(self, state):
        ''' Normalises state according to expecting maximum state values in game '''
        return state/self.stateExtreme
    
    def getReward(self, action, render=False):
        ''' Carries out action and returns reward'''
        if render: 
            self.game.render()
        
        observation, reward, done, info = self.game.step(action)       
        
        self.state = observation
        self.isOver = done
                
        return reward
    
    def customReward(self, reward):
        ''' Adjusts the reward of each time step '''
        return reward
    
    def customFitness(self, fitness):
        ''' Adjusts the final agent fitness '''

        fitness /= self.game._max_episode_steps # Fitness range is now : {-1 to 0] - higher if game ends early
         # Adds a term between 0 to 0.15 that's higher if maxPosition is higher (position range : {-1.2 to 0.6})
        fitness += (self.maxPosition+1.2)/12
        
        # Final Fitness Range : {0 to 1} (although 1 is not acheivable as it must take some time to reach further right)   
        return (fitness+1)/1.15
    
    @staticmethod
    def softmax(fitnesses, scalefactor=1):
        ''' Given array of fitnesses (and scaling factor to avoid np.exp inaccuracies), scales values to range 0 to 1, 
        with total array sum being 1, to be used as probabilities'''
        # Inputs scaled by softmaxFactor term to avoid inaccuracies
        return np.exp(fitnesses*scalefactor) / np.exp(fitnesses*scalefactor).sum()
    
    def evaluateFitness(self, agent, seed=0, render=False):
        ''' Calculate agent's fitness in one game, given agent, seed, render(T/F)'''
        fitness = 0
        self.reset(seed) # Reset game to initial state, with a new seed for every generation  
    
        while not self.isOver:
            state = self.getState() # Get game state
            self.recordState() # Records information about the state, used later in custom fitness function
            
            processedState = self.normaliseState(state).reshape(-1, 1) # Normalise and reshape state
            actionProbs = agent.feedForward(processedState) # Get agent's action activations for actions in current state
            
            actionProbs = self.softmax(actionProbs.flatten(), 12.5) # Converts agent's activations to probabilities
            action = np.random.choice(np.arange(3), 1, p=actionProbs)[0] # Chooses action according to probability output by NN
            
            reward = self.getReward(action, render) # Get reward for action taken by agent
            reward = self.customReward(reward) # Adjusts reward according to custom reward function
            
            fitness += reward
                    
        fitness = self.customFitness(fitness) # Adjusts fitness according to custom fitness function
        
        return fitness
