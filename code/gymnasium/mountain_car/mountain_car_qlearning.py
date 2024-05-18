import logging
import os
from dataclasses import dataclass

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


@dataclass
class Hyperparameters:
    RANDOM_SEED: int
    DISCRETE_BIN_SIZE: int
    LEARNING_RATE: float
    DISCOUNT: float
    EPISODES: int
    EPSILON: float
    START_EPSILON_DECAYING: int
    END_EPSILON_DECAYING: int
    EPISODE_STATS: int


class RLEnvironment:
    def __init__(self, hyper: Hyperparameters):
        self.env = gym.make("MountainCar-v0")  # , render_mode="human")
        self.default_rng = np.random.default_rng(seed=hyper.RANDOM_SEED)
        self.observation_space_high = self.env.observation_space.high
        self.observation_space_low = self.env.observation_space.low
        self.discrete_window_size = (
            (self.observation_space_high - self.observation_space_low)
            / [hyper.DISCRETE_BIN_SIZE]
            * len(self.env.observation_space.high)
        )

    def get_action_space_size(self):
        return self.env.action_space.n

    def get_discrete_state(self, state):
        discrete_state = (
            state - self.env.observation_space.low
        ) / self.discrete_window_size
        return tuple(discrete_state.astype(int))

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def close(self):
        self.env.close()


class QLearning:
    def __init__(self, env: RLEnvironment, hyper: Hyperparameters):
        self.env = env
        self.hyper = hyper
        self.episode_rewards_list = []
        self.summarised_dictionary = {"ep": [], "avg": [], "min": [], "max": []}
        self.epsilon_decay_value = 1 / (
            hyper.END_EPSILON_DECAYING - hyper.START_EPSILON_DECAYING
        )
        self.Q_TABLE = self.env.default_rng.uniform(
            low=-2,
            high=0,
            size=(
                [hyper.DISCRETE_BIN_SIZE] * len(self.env.observation_space_high)
                + [self.env.get_action_space_size()]
            ),
        )

    def train(self):
        for episode in tqdm(range(self.hyper.EPISODES)):
            episode_reward = 0
            terminated, _ = False, False
            curr_state, _ = self.env.reset()
            curr_discrete_state = self.env.get_discrete_state(curr_state)

            steps = 0
            while not terminated:
                steps += 1
                if self.env.default_rng.random() > self.hyper.EPSILON:
                    action = np.argmax(self.Q_TABLE[curr_discrete_state])
                else:
                    action = self.env.default_rng.integers(
                        self.env.get_action_space_size()
                    )

                new_state, reward, terminated, _, _ = self.env.env.step(action)
                new_discrete_state = self.env.get_discrete_state(new_state)

                if not terminated:
                    max_future_q = np.max(self.Q_TABLE[new_discrete_state])
                    current_q = self.Q_TABLE[curr_discrete_state + (action,)]
                    new_q = (
                        1 - self.hyper.LEARNING_RATE
                    ) * current_q + self.hyper.LEARNING_RATE * (
                        reward + self.hyper.DISCOUNT * max_future_q
                    )
                    self.Q_TABLE[curr_discrete_state + (action,)] = new_q

                curr_discrete_state = new_discrete_state
                episode_reward += reward

            self.episode_rewards_list.append(episode_reward)

            if (
                self.hyper.END_EPSILON_DECAYING
                >= episode
                >= self.hyper.START_EPSILON_DECAYING
            ):
                self.hyper.EPSILON -= self.epsilon_decay_value

            if episode % self.hyper.EPISODE_STATS == 0:
                average_reward = np.mean(
                    self.episode_rewards_list[-self.hyper.EPISODE_STATS :]
                )
                self.summarised_dictionary["ep"].append(episode)
                self.summarised_dictionary["avg"].append(average_reward)
                self.summarised_dictionary["min"].append(
                    np.min(self.episode_rewards_list[-self.hyper.EPISODE_STATS :])
                )
                self.summarised_dictionary["max"].append(
                    np.max(self.episode_rewards_list[-self.hyper.EPISODE_STATS :])
                )
                logger.info(
                    f"Episode:{episode}, avg:{average_reward}, "
                    f"min:{np.min(self.episode_rewards_list[-self.hyper.EPISODE_STATS:])}, "
                    f"max:{np.max(self.episode_rewards_list[-self.hyper.EPISODE_STATS:])}, "
                    f"steps = {steps}, terminated = {terminated}"
                )


def setup_logger(log_file_name, log_level=logging.INFO):
    if os.path.exists(log_file_name):
        os.remove(log_file_name)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logger = logging.getLogger(__name__)
    file_handler = logging.FileHandler(log_file_name)
    file_handler.setLevel(log_level)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def save_stats_figure(summarised_dictionary):
    plt.plot(
        summarised_dictionary["ep"],
        summarised_dictionary["avg"],
        label="avg",
    )
    plt.plot(
        summarised_dictionary["ep"],
        summarised_dictionary["min"],
        label="min",
    )
    plt.plot(
        summarised_dictionary["ep"],
        summarised_dictionary["max"],
        label="max",
    )
    plt.legend(loc=4)
    plt.title("Mountain Car Q-Learning")
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.savefig("mountain_car_qlearning.png")


if __name__ == "__main__":
    logger = setup_logger("mountain_car_qlearning.log")
    logger.info("Mountain Car Q-Learning")

    # Setup Hyperparameters
    hyper = Hyperparameters(
        RANDOM_SEED=100,
        DISCRETE_BIN_SIZE=20,
        LEARNING_RATE=0.1,
        DISCOUNT=0.95,
        EPISODES=2000,
        EPSILON=1,
        START_EPSILON_DECAYING=1,
        END_EPSILON_DECAYING=2000 // 2,
        EPISODE_STATS=100,
    )

    # Initialise the Environment
    rl_env = RLEnvironment(hyper)

    # The Q-Learning Algorithm
    q_learning = QLearning(rl_env, hyper)
    q_learning.train()

    # Close the Environment
    rl_env.close()

    # Save the Stats Figure
    save_stats_figure(q_learning.summarised_dictionary)
