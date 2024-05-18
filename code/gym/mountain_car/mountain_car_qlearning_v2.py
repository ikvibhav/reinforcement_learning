import logging
import os

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Random Number Generator
default_rng = np.random.default_rng(seed=100)

# Setup Logging
log_file_name = "mountain_car_qlearning_v2.log"
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
env = gym.make("MountainCar-v0")  # , render_mode="human")

# Environment values
print(env.observation_space.high)  # [0.6  0.07]
print(env.observation_space.low)  # [-1.2  -0.07]
print(env.action_space.n)  # 3

# Hyperparamters
DISCRETE_BIN_SIZE = 20
DISCRETE_OS_SIZE = [DISCRETE_BIN_SIZE] * len(env.observation_space.high)
discrete_window_size = (
    env.observation_space.high - env.observation_space.low
) / DISCRETE_OS_SIZE
Q_TABLE = default_rng.uniform(
    low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n])
)
LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 2000

# Exploration settings
EPSILON = 1
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2
epsilon_decay_value = EPSILON / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

# For stats
EPISODE_STATS = 100
episode_rewards_list = []
summarised_dictionary = {"ep": [], "avg": [], "min": [], "max": []}


def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_window_size
    return tuple(discrete_state.astype(int))


for episode in tqdm(range(EPISODES)):
    episode_reward = 0
    terminated, truncated = False, False
    curr_state, _ = env.reset()
    curr_discrete_state = get_discrete_state(curr_state)

    steps = 0
    while not terminated:
        steps += 1
        if default_rng.random() > EPSILON:
            action = np.argmax(Q_TABLE[curr_discrete_state])
        else:
            action = default_rng.integers(env.action_space.n)

        new_state, reward, terminated, truncated, _ = env.step(action)
        new_discrete_state = get_discrete_state(new_state)

        if not terminated:
            max_future_q = np.max(Q_TABLE[new_discrete_state])
            current_q = Q_TABLE[curr_discrete_state + (action,)]
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (
                reward + DISCOUNT * max_future_q
            )
            Q_TABLE[curr_discrete_state + (action,)] = new_q
        elif new_state[0] >= env.unwrapped.goal_position:
            Q_TABLE[curr_discrete_state + (action,)] = 0

        curr_discrete_state = new_discrete_state
        episode_reward += reward

    episode_rewards_list.append(episode_reward)

    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        EPSILON -= epsilon_decay_value

    if episode % EPISODE_STATS == 0:
        avg_reward = sum(episode_rewards_list[-EPISODE_STATS:]) / len(
            episode_rewards_list[-EPISODE_STATS:]
        )
        summarised_dictionary["ep"].append(episode)
        summarised_dictionary["avg"].append(avg_reward)
        summarised_dictionary["min"].append(min(episode_rewards_list[-EPISODE_STATS:]))
        summarised_dictionary["max"].append(max(episode_rewards_list[-EPISODE_STATS:]))
        logger.info(
            f"Episode:{episode}, avg:{avg_reward}, "
            f"min:{min(episode_rewards_list[-EPISODE_STATS:])}, "
            f"max:{max(episode_rewards_list[-EPISODE_STATS:])}, "
            f"steps = {steps}, truncated = {truncated}, terminated = {terminated}"
        )

env.close()
plt.plot(summarised_dictionary["ep"], summarised_dictionary["avg"], label="avg")
plt.plot(summarised_dictionary["ep"], summarised_dictionary["min"], label="min")
plt.plot(summarised_dictionary["ep"], summarised_dictionary["max"], label="max")
plt.legend(loc=4)  # bottom right
plt.title("Mountain Car Q-Learning")
plt.ylabel("Average reward/Episode")
plt.xlabel("Episodes")
plt.savefig("mountain_car_qlearning_v2.png")
