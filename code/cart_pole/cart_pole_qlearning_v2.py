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
EPISODES = 20000
DISCOUNT = 0.95
EPISODE_DISPLAY = 500
LEARNING_RATE = 0.25
EPSILON = 0.6
EPSILON_MIN = 0.2
EPSILON_DECREMENTER = (EPSILON - EPSILON_MIN)//EPISODES


#Q-Table of size theta_state_size*theta_dot_state_size*env.action_space.n
theta_minmax = 15
theta_dot_minmax = 50
theta_state_size = 7
theta_dot_state_size = 13
Q_TABLE = np.random.randn(theta_state_size,theta_dot_state_size,env.action_space.n)

# For stats
ep_rewards = []
ep_rewards_table = {'ep': [], 'avg': [], 'min': [], 'max': []}

def discretised_state(state):
	#state[2] -> theta
	#state[3] -> theta_dot
	discrete_state = np.array([0,0])		#Initialised discrete array
	if state[2] < math.radians(-15):
		discrete_state[0] = 0
	elif state[2] < math.radians(-10):
		discrete_state[0] = 1
	elif state[2] < math.radians(-5):
		discrete_state[0] = 2
	elif state[2] < math.radians(0):
		discrete_state[0] = 3
	elif state[2] < math.radians(5):
		discrete_state[0] = 4
	elif state[2] < math.radians(10):
		discrete_state[0] = 5
	else:
		discrete_state[0] = 6

	if state[3] < math.radians(-48):
		discrete_state[1] = 0
	elif state[3] < math.radians(-40):
		discrete_state[1] = 1
	elif state[3] < math.radians(-32):
		discrete_state[1] = 2
	elif state[3] < math.radians(-24):
		discrete_state[1] = 3
	elif state[3] < math.radians(-16):
		discrete_state[1] = 4
	elif state[3] < math.radians(-8):
		discrete_state[1] = 5
	elif state[3] < math.radians(0):
		discrete_state[1] = 6
	elif state[3] < math.radians(8):
		discrete_state[1] = 7
	elif state[3] < math.radians(16):
		discrete_state[1] = 8
	elif state[3] < math.radians(24):
		discrete_state[1] = 9
	elif state[3] < math.radians(32):
		discrete_state[1] = 10
	elif state[3] < math.radians(40):
		discrete_state[1] = 11
	else:
		discrete_state[1] = 12
	

	'''
	#Check why not working
	if state[2] < -theta_minmax:
		discrete_state[0] = 0
	elif state[2] > theta_minmax:
		discrete_state[0] = theta_state_size-1
	else:
		ratio_theta =  (theta_state_size-1)/2*theta_minmax
		discrete_state[0] = np.floor((state[2]+theta_minmax)*ratio_theta)

	if state[3] < -theta_dot_minmax:
		discrete_state[1] = 0
	elif state[3] > theta_dot_minmax:
		discrete_state[1] = theta_dot_state_size-1
	else:
		ratio_theta_dot =  (theta_dot_state_size-1)/2*theta_dot_minmax
		discrete_state[1] = np.floor((state[3]+theta_dot_minmax)*ratio_theta_dot)
	'''
	return tuple(discrete_state.astype(np.int))

#print(discretised_state(env.reset()))

for episode in range(EPISODES):
	episode_reward = 0
	curr_discrete_state = discretised_state(env.reset())
	done = False

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
			max_future_q = np.max(Q_TABLE[new_discrete_state])
			current_q = Q_TABLE[curr_discrete_state+(action,)]
			new_q = current_q + LEARNING_RATE*(reward + DISCOUNT*max_future_q - current_q)
			Q_TABLE[curr_discrete_state+(action,)]=new_q
		#else:
		#	Q_TABLE[curr_discrete_state + (action,)] = 200
			#print(f"We made it on episode {episode}")

		curr_discrete_state = new_discrete_state
		episode_reward += reward

	#if EPSILON > EPSILON_MIN:
	#	EPSILON = EPSILON - EPSILON_DECREMENTER

	ep_rewards.append(episode_reward)

	if not episode % EPISODE_DISPLAY:
		avg_reward = sum(ep_rewards[-EPISODE_DISPLAY:])/len(ep_rewards[-EPISODE_DISPLAY:])
		ep_rewards_table['ep'].append(episode)
		ep_rewards_table['avg'].append(avg_reward)
		ep_rewards_table['min'].append(min(ep_rewards[-EPISODE_DISPLAY:]))
		ep_rewards_table['max'].append(max(ep_rewards[-EPISODE_DISPLAY:]))
		print(f"Episode:{episode} avg:{avg_reward} min:{min(ep_rewards[-EPISODE_DISPLAY:])} max:{max(ep_rewards[-EPISODE_DISPLAY:])}")

env.close()

plt.plot(ep_rewards_table['ep'], ep_rewards_table['avg'], label="avg")
plt.plot(ep_rewards_table['ep'], ep_rewards_table['min'], label="min")
plt.plot(ep_rewards_table['ep'], ep_rewards_table['max'], label="max")
plt.legend(loc=4) #bottom right
plt.show()