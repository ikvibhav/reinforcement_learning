import gym
import numpy as np
import math
import matplotlib.pyplot as plt 

env = gym.make("CartPole-v0")

#Environment values
print(env.observation_space.high)	#[4.8000002e+00 3.4028235e+38 4.1887903e-01 3.4028235e+38]
print(env.observation_space.low)	#[-4.8000002e+00 -3.4028235e+38 -4.1887903e-01 -3.4028235e+38]
print(env.action_space.n)			#2

#Hyperparamters
DISCRETE_BUCKETS = 20
EPISODES = 30000
DISCOUNT = [0.5, 0.8, 0.95, 1]
EPISODE_DISPLAY = 500
LEARNING_RATE = 0.25
EPSILON = 0.1

#Q-Table of size theta_state_size*theta_dot_state_size*env.action_space.n
theta_minmax = env.observation_space.high[2] #math.radians(24)
theta_dot_minmax = math.radians(50)
theta_state_size = 50
theta_dot_state_size = 50
Q_TABLE = np.random.randn(theta_state_size,theta_dot_state_size,env.action_space.n)

# For stats
ep_rewards = []
ep_rewards_table = {'ep': [], 'zero': [], 'one': [], 'two': [], 'three': []}

def discretised_state(state):
	#state[2] -> theta
	#state[3] -> theta_dot
	discrete_state = np.array([0,0])		#Initialised discrete array

	theta_window =  ( theta_minmax - (-theta_minmax) ) / theta_state_size
	discrete_state[0] = ( state[2] - (-theta_minmax) ) // theta_window
	discrete_state[0] = min(theta_state_size-1, max(0,discrete_state[0]))

	theta_dot_window =  ( theta_dot_minmax - (-theta_dot_minmax) )/ theta_dot_state_size
	discrete_state[1] = ( state[3] - (-theta_dot_minmax) ) // theta_dot_window
	discrete_state[1] = min(theta_dot_state_size-1, max(0,discrete_state[1]))

	return tuple(discrete_state.astype(np.int))

for dis_iter in range(len(DISCOUNT)):
	Q_TABLE = np.random.randn(theta_state_size,theta_dot_state_size,env.action_space.n)
	ep_rewards = []
	for episode in range(EPISODES):
		episode_reward = 0
		curr_discrete_state = discretised_state(env.reset())
		done = False
		i = 0

		if episode % EPISODE_DISPLAY == 0:
			render_state = True
		else:
			render_state = False

		while not done:
			if np.random.random() > EPSILON:
				action = np.argmax(Q_TABLE[curr_discrete_state])
			else:
				action = np.random.randint(0, env.action_space.n)
			
			new_state, reward, done, _ = env.step(action)
			new_discrete_state = discretised_state(new_state)
			if render_state:
				env.render()

			if not done:
				max_future_q = np.max(Q_TABLE[new_discrete_state[0],new_discrete_state[1]])
				current_q = Q_TABLE[curr_discrete_state[0],curr_discrete_state[1], action]
				new_q = current_q + LEARNING_RATE*(reward + DISCOUNT[dis_iter]*max_future_q - current_q)
				Q_TABLE[curr_discrete_state[0],curr_discrete_state[1], action]=new_q

			i=i+1
			curr_discrete_state = new_discrete_state
			episode_reward += reward

		ep_rewards.append(episode_reward)

		if not episode % EPISODE_DISPLAY:
			avg_reward = sum(ep_rewards[-EPISODE_DISPLAY:])/len(ep_rewards[-EPISODE_DISPLAY:])
			if dis_iter == 0:
				ep_rewards_table['ep'].append(episode)
				ep_rewards_table['zero'].append(avg_reward)
			elif dis_iter == 1:
				ep_rewards_table['one'].append(avg_reward)
			elif dis_iter == 2:
				ep_rewards_table['two'].append(avg_reward)
			elif dis_iter == 3:
				ep_rewards_table['three'].append(avg_reward)

			print(f"Episode:{episode} avg:{avg_reward} min:{min(ep_rewards[-EPISODE_DISPLAY:])} max:{max(ep_rewards[-EPISODE_DISPLAY:])} disount factor:{DISCOUNT[dis_iter]}")

env.close()

plt.plot(ep_rewards_table['ep'], ep_rewards_table['zero'], label="0.5")
plt.plot(ep_rewards_table['ep'], ep_rewards_table['one'], label="0.8")
plt.plot(ep_rewards_table['ep'], ep_rewards_table['two'], label="0.95")
plt.plot(ep_rewards_table['ep'], ep_rewards_table['three'], label="1")
plt.legend(loc=4) #bottom right
plt.title('CartPole Q-Learning discount factor sensitivity')
plt.ylabel('Average reward/Episode')
plt.xlabel('Episodes')
plt.show()
