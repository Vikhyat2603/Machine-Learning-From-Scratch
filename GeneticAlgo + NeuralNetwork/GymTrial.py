import gym
import numpy as np
##MountainCar-v0
##MountainCarContinuous-v0
##Acrobot-v1
##Pendulum-v0
##CartPole-v0
##FlappyBird-v0
game = gym.make("CartPole-v0")
game._max_episode_steps=200
try:
    # for i_episode in range(20):
    observation = game.reset()
    for t in range(game._max_episode_steps):
        game.render()
#            action = game.action_space[0]
        action = game.env.action_space.sample()
        observation, reward, done, info = game.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
except Exception as e:
    print(e)
except KeyboardInterrupt:
    pass
finally:
    game.close()
