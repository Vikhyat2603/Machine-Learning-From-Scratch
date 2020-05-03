import gym
import numpy as np

class CartPoleGame:       
    def __init__(self, gameLength=500):
        ''' Makes the OpenAI Gym CartPole-v0 game environment'''
        self.game = gym.make("CartPole-v0")
        self.stateDim, self.actionDim = self.game.observation_space.shape[0], 1
        self.game._max_episode_steps = gameLength 
        self.reset()
        
        # Extreme absolute values chosen by experimentation
        self.stateExtreme = np.array([5, 5, 24, 5])
            
    def reset(self, seed=0):
        self.game.seed(seed)
        self.game.action_space.seed(seed)
        self.state = self.game.reset()
        self.isOver = False
            
    def getState(self):
        ''' Gets current game state '''
        return self.state
    
    def recordState(self):
        ''' Records information about the state, used later in custom fitness function '''
        pass
    
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
        return fitness/self.game._max_episode_steps # Fitness range : {0 to 1} (higher reward for balancing cartPole longer)
    
    def evaluateFitness(self, agent, seed=0, render=False):
        ''' Calculate agent's fitness in one game, given agent, seed, render(T/F)'''
        fitness = 0
        self.reset(seed) # Reset game to initial state, with a new seed for every generation  
    
        while not self.isOver:
            state = self.getState() # Get game state
            self.recordState() # Records information about the state, according to custom fitness function
            
            processedState = self.normaliseState(state).reshape(-1, 1) # Normalise and reshape state
            action = agent.feedForward(processedState) # Get agent's action activation for current state
            action = round(float(action)) # Rounds agent's activation (to 0 or 1(left or right))
    
            reward = self.getReward(action, render) # Get reward for action taken by agent
            reward = self.customReward(reward) # Adjusts reward according to custom reward function
            
            fitness += reward
                                
        fitness = self.customFitness(fitness) # Adjusts fitness according to custom fitness function
        
        return fitness  