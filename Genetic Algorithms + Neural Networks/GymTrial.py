import gym

##MountainCar-v0
##MountainCarContinuous-v0
##Acrobot-v1
##Pendulum-v0
##CartPole-v0

game = gym.make("MountainCar-v0")
game._max_episode_steps=200

observation = game.reset()
for t in range(game._max_episode_steps):
    game.render()
    
    action = game.env.action_space.sample()
    observation, reward, done, info = game.step(action)
    if done:
        print("Episode finished after {} timesteps".format(t+1))
        break
    
game.close()
