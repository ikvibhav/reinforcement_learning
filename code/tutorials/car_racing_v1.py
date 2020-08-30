import gym
import numpy as np

render = True
n_episodes = 1
env = gym.make('CarRacing-v0')
#env = gym.make("MountainCar-v0")

print(env.observation_space)
print(env.action_space)
print(env.action_space.high)
print(env.action_space.low)
#print(env.observation_space.high)
#print(env.observation_space.low)
#exit()
'''
rewards = []
for i_episode in range(n_episodes):
    observation = env.reset()
    sum_reward = 0
    #for t in range(1000):
    done = False
    while not done:
        if render:
            env.render()
        # [steering, gas, brake]
        action = env.action_space.sample()
        # observation is 96x96x3
        observation, reward, done, _ = env.step(action)
        #print(len(observation))
        print(np.shape(observation))
        print(reward)
        #print(len(observation[0]))
        #print(len(observation[0][0]))
        #print(observation)
        # break
        #sum_reward += reward
        #if(t % 100 == 0):
        #    print(t)
        #if done or t == 999:
        #    print("Episode {} finished after {} timesteps".format(i_episode, t+1))
        #    print("Reward: {}".format(sum_reward))
        #    rewards.append(sum_reward)
        #if done:
        #    break
'''