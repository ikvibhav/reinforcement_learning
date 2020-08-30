import gym
import numpy as np

env = gym.make("MountainCar-v0")
#print(env.action_space.n) #To print out the size of the action space
# we can pass a 0, 1, or 2 as our "action" for each step.
'''
env.reset()

done = False
while not done:
    action = 2  # always go right!
    env.step(action)
    env.render()
'''
#In the case of this gym environment,
#the observations are returned from resets and steps
#print(env.reset())
''''
state = env.reset()

done = False
i =1
while not done:
    action = 2
    new_state, reward, done, _ = env.step(action)
    i=i+1
    print(i)	#our limit is 200 steps
    print(reward, new_state)
'''

#To print the range of the state values
#print(env.observation_space.high)
#print(env.observation_space.low)
#Number of Q-values is too large, so we reduce the range to 20 bckets/groups:
DISCRETE_OS_SIZE = [20]*len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low)/DISCRETE_OS_SIZE

#print(discrete_os_win_size) #size of each bucket

#For a combination of states, the agent 
#performs the action which has the highest Q value

q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))
#low and high are dependent on the environments

print(np.shape(q_table))