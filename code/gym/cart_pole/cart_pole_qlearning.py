import logging
import os

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Setup Logging
log_file_name = "cart_pole_qlearning.log"
if os.path.exists(log_file_name):
    os.remove(log_file_name)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)
file_handler = logging.FileHandler(log_file_name)
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

env = gym.make("CartPole-v1")  # , render_mode="human")

# Environment values
# Observation Space
# - [Cart Position, Cart Velocity, Pole Angle, Pole Velocity]
# - Max: [4.8000002e+00 3.4028235e+38 4.1887903e-01 3.4028235e+38]
# - Min: [-4.8000002e+00 -3.4028235e+38 -4.1887903e-01 -3.4028235e+38]
# Action Space
# - [Push Cart to Left, Push Cart to Right]
# - 0: Push Cart to Left
# - 1: Push Cart to Right
print(env.observation_space.high)
print(env.observation_space.low)
print(env.action_space.n)

# Hyperparamters
EPISODES = 30000  # Max number of episodes = 500 in CartPole-v1
DISCOUNT = 0.95
EPISODE_DISPLAY = 1000
LEARNING_RATE = 0.25
EPSILON = 0.2

# Pole Angle and Pole Velocity are considered in this example.
# Pole Angle is called theta and Pole Velocity is called theta_dot
# Q-Table of size theta_state_size*theta_dot_state_size*env.action_space.n
theta_minmax = env.observation_space.high[2] / 2
theta_dot_minmax = env.observation_space.high[3]
theta_state_size = 50
theta_dot_state_size = 50
STATE_BINS = [
    np.linspace(-theta_minmax, theta_minmax, theta_state_size),
    np.linspace(-theta_dot_minmax, theta_dot_minmax, theta_dot_state_size),
]
# Q_TABLE = np.random.randn(theta_state_size, theta_dot_state_size, env.action_space.n)
Q_TABLE = np.random.uniform(
    low=0, high=1, size=(theta_state_size, theta_dot_state_size, env.action_space.n)
)
import pdb

pdb.set_trace()
# For stats
episode_rewards_list = []
summarised_dictionary = {"ep": [], "avg": [], "min": [], "max": []}


def discretised_state(state):
    # state[2] -> theta
    # state[3] -> theta_dot
    discrete_state = np.array([0, 0])  # Initialised discrete array

    theta_window = (theta_minmax - (-theta_minmax)) / theta_state_size
    discrete_state[0] = (state[0][2] - (-theta_minmax)) // theta_window
    discrete_state[0] = min(theta_state_size - 1, max(0, discrete_state[0]))

    theta_dot_window = (theta_dot_minmax - (-theta_dot_minmax)) / theta_dot_state_size
    discrete_state[1] = (state[0][3] - (-theta_dot_minmax)) // theta_dot_window
    discrete_state[1] = min(theta_dot_state_size - 1, max(0, discrete_state[1]))

    return tuple(discrete_state.astype(int))


def test_discretize_state(state, bins):
    # return tuple(np.digitize(s, b) for s, b in zip(state, bins))
    zipped = zip(state, bins)
    discretized = []
    for s, b in zipped:
        digitized = np.digitize(s, b)
        digitized = max(min(digitized, len(b) - 1), 0)
        discretized.append(digitized)
    return tuple(discretized)


for episode in tqdm(range(EPISODES)):
    logger.debug(f"Episode: {episode}")
    episode_reward = 0
    curr_state, _ = env.reset()
    curr_discrete_state = test_discretize_state(
        [curr_state[2], curr_state[3]], STATE_BINS
    )
    terminated, truncated = False, False

    episode_length = 0
    while not terminated:
        episode_length += 1
        if np.random.random() > EPSILON:
            action = np.argmax(Q_TABLE[curr_discrete_state])
        else:
            action = np.random.randint(0, env.action_space.n)
        new_state, reward, terminated, truncated, _ = env.step(action)
        new_discrete_state = test_discretize_state(
            [new_state[2], new_state[3]], STATE_BINS
        )
        logger.debug(
            (
                f"curr_state: {curr_state[2], curr_state[3]}, curr_discrete_state: {curr_discrete_state}, "
                f"new_state: {new_state[2], new_state[3]}, new_discrete_state: {new_discrete_state}, "
                f"action: {action}, reward: {reward}, "
            )
        )
        # Q-Learning
        max_future_q = np.max(Q_TABLE[new_discrete_state[0], new_discrete_state[1]])
        current_q = Q_TABLE[curr_discrete_state[0], curr_discrete_state[1], action]

        new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (
            reward + DISCOUNT * max_future_q
        )

        Q_TABLE[curr_discrete_state[0], curr_discrete_state[1], action] = new_q

        curr_discrete_state = new_discrete_state
        episode_reward += reward

    episode_rewards_list.append(episode_reward)

    if not episode % EPISODE_DISPLAY:
        avg_reward = sum(episode_rewards_list[-EPISODE_DISPLAY:]) / len(
            episode_rewards_list[-EPISODE_DISPLAY:]
        )
        summarised_dictionary["ep"].append(episode)
        summarised_dictionary["avg"].append(avg_reward)
        summarised_dictionary["min"].append(
            min(episode_rewards_list[-EPISODE_DISPLAY:])
        )
        summarised_dictionary["max"].append(
            max(episode_rewards_list[-EPISODE_DISPLAY:])
        )
        logger.info(
            (
                f"Episode:{episode}, avg:{avg_reward} "
                f"min:{min(episode_rewards_list[-EPISODE_DISPLAY:])} "
                f"max:{max(episode_rewards_list[-EPISODE_DISPLAY:])}, "
                f"episode_length = {episode_length}, "
                f"truncated = {truncated}, terminated = {terminated}"
            )
        )

env.close()
plt.plot(summarised_dictionary["ep"], summarised_dictionary["avg"], label="avg")
plt.plot(summarised_dictionary["ep"], summarised_dictionary["min"], label="min")
plt.plot(summarised_dictionary["ep"], summarised_dictionary["max"], label="max")
plt.legend(loc=4)  # bottom right
plt.title("CartPole Q-Learning")
plt.ylabel("Average reward/Episode")
plt.xlabel("Episodes")
plt.savefig("cartpole_qlearning.png")
