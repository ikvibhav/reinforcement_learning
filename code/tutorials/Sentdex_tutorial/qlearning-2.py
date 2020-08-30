import gym
import numpy as np

env = gym.make("MountainCar-v0")

LEARNING_RATE = 0.1
DISCOUNT = 0.95 			#Weighting for future actions
EPISODES = 25000

SHOW_EVERY = 2000			

DISCRETE_OS_SIZE = [20]*len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low)/DISCRETE_OS_SIZE

epsilon = 0.5				#Higher value, higher chance to explore
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES//2		#Always divides to an integer
epsilon_decay_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)

q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))


def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low)/discrete_os_win_size
    return tuple(discrete_state.astype(np.int))  # we use this tuple to look up the 2 Q values for the available actions in the q-table

#print(discrete_state)						#prints the discretised state value
#print(q_table[discrete_state])				#prints the Q-values for the initial state
#print(np.argmax(q_table[discrete_state]))	#prints the action corresponding to largest q value

for episode in range(EPISODES):
	discrete_state = get_discrete_state(env.reset())
	done = False

	if episode % SHOW_EVERY == 0:
		print(episode)
		render = True
	else:
		render = False
	
	while not done:
	    #action = np.argmax(q_table[discrete_state])

	    #action selection using epsilon. Fr mountaincar-v0, without epsilon aspect gives better learning
	    if np.random.random() > epsilon:
	    	action = np.argmax(q_table[discrete_state])			# Get action from Q table
	    else:
	    	action = np.random.randint(0, env.action_space.n)	# random action 0<= action < env.action_space.n

	    new_state, reward, done, _ = env.step(action)
	    new_discrete_state = get_discrete_state(new_state)
	    if render:
	    	env.render()
	    # if simulation did not end yet after last step, update Q table
	    if not done:
	    	# Maximum possible Q value in next step (for new state)
	    	max_future_q = np.max(q_table[new_discrete_state])

	    	# Current Q value (for current state and performed action)
	    	current_q = q_table[discrete_state+(action,)]

	    	# New Q value for current state and action
	    	new_q = (1-LEARNING_RATE)*current_q + LEARNING_RATE*(reward + DISCOUNT*max_future_q)
	    	
	    	# Update Q table with new Q value 
	    	q_table[discrete_state+(action,)]=new_q

	    # Simulation ended (for any reson) - if goal position is achived - update Q value with reward directly
	    elif new_state[0] >= env.goal_position:
	            #q_table[discrete_state + (action,)] = reward
	            print(f"We made it on episode {episode}")
	            q_table[discrete_state + (action,)] = 0

	    # Update the discrete_state variable
	    discrete_state = new_discrete_state

	# Decay done after every episode
	if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
		epsilon -= epsilon_decay_value

env.close()