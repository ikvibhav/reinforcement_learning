import logging
import os

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

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
DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
discrete_window_size = (
    env.observation_space.high - env.observation_space.low
) / DISCRETE_OS_SIZE
Q_TABLE = np.random.uniform(
    low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n])
)
LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 25000


def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_window_size
    return tuple(discrete_state.astype(int))


for i in tqdm(range(EPISODES)):
    terminated, truncated = False, False
    curr_state, _ = env.reset()
    curr_discrete_state = get_discrete_state(curr_state)

    while not terminated:
        action = np.argmax(Q_TABLE[curr_discrete_state])
        new_state, reward, terminated, truncated, _ = env.step(action)
        new_discrete_state = get_discrete_state(new_state)

        if not terminated:
            max_future_q = np.max(Q_TABLE[new_discrete_state])
            current_q = Q_TABLE[curr_discrete_state + (action,)]
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (
                reward + DISCOUNT * max_future_q
            )
            Q_TABLE[curr_discrete_state + (action,)] = new_q

        elif new_state[0] >= env.goal_position:
            logger.info(f"Goal reached at episode {i}")
            terminated = True
