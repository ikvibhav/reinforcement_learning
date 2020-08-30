import gym
import numpy as np
import matplotlib.pyplot as plt 

env = gym.make("MountainCar-v0")

#Environment values
print(env.observation_space.high)	#[0.6  0.07]
print(env.observation_space.low)	#[-1.2  -0.07]
print(env.action_space.n)			#3

DISCRETE_BUCKETS = 20
EPISODES = 5000
DISCOUNT = 0.95
EPISODE_DISPLAY = 500
LEARNING_RATE = 0.1
EPSILON = 0.8
EPSILON_MIN = 0.1
EPSILON_DECREMENTER = 0.02

#Discretise each dimension of the observation space to size DISCRETE_BUCKETS
DISCRETE_STATE_SIZE = [DISCRETE_BUCKETS]*len(env.observation_space.high)
DISCRETE_WIN_SIZE = (env.observation_space.high-env.observation_space.low)/DISCRETE_STATE_SIZE
print(DISCRETE_WIN_SIZE)

#Q-Table of size DISCRETE_BUCKETS*DISCRETE_BUCKETS*env.action_space.n
Q_TABLE = np.random.randn(DISCRETE_BUCKETS,DISCRETE_BUCKETS,env.action_space.n)

# For stats
ep_rewards = []
ep_rewards_table = {'ep': [], 'avg': [], 'min': [], 'max': []}

#print(np.shape(Q_TABLE))

def discretised_state(state):
	discrete_state = (state-env.observation_space.low)//DISCRETE_WIN_SIZE
	return tuple(discrete_state.astype(np.int))		#integer tuple as we need to use it later on to extract Q table values

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
			new_q = (1-LEARNING_RATE)*current_q + LEARNING_RATE*(reward + DISCOUNT*max_future_q)
			Q_TABLE[curr_discrete_state+(action,)]=new_q
		elif new_state[0] >= env.goal_position:
			print(f"We made it on episode {episode}")
			Q_TABLE[curr_discrete_state + (action,)] = 0

		curr_discrete_state = new_discrete_state
		episode_reward += reward

	if EPSILON > EPSILON_MIN:
		EPSILON = EPSILON - EPSILON_DECREMENTER

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




