import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

env = gym.make("CartPole-v1", render_mode="human")

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
EPISODES = 100  # Max number of episodes = 500 in CartPole-v1
DISCOUNT = 0.95
EPISODE_DISPLAY = 10
LEARNING_RATE = 0.25
EPSILON = 0.2

# Pole Angle and Pole Velocity are considered in this example.
# Pole Angle is called theta and Pole Velocity is called theta_dot
# Q-Table of size theta_state_size*theta_dot_state_size*env.action_space.n
theta_minmax = env.observation_space.high[2]
theta_dot_minmax = env.observation_space.high[3]
theta_state_size = 50
theta_dot_state_size = 50
Q_TABLE = np.random.randn(theta_state_size, theta_dot_state_size, env.action_space.n)

# For stats
ep_rewards = []
ep_rewards_table = {"ep": [], "avg": [], "min": [], "max": []}


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
    return tuple(np.digitize(s, b) for s, b in zip(state, bins))


for episode in range(EPISODES):
    episode_reward = 0
    curr_state, _ = env.reset()
    curr_discrete_state = test_discretize_state(
        [curr_state[2], curr_state[3]],
        [
            np.linspace(-theta_minmax, theta_minmax, theta_state_size),
            np.linspace(-theta_dot_minmax, theta_dot_minmax, theta_dot_state_size),
        ],
    )
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

        # new_state, reward, done, _ = env.step(action)
        new_state, reward, done, _, _ = env.step(action)
        new_discrete_state = test_discretize_state(
            new_state,
            [
                np.linspace(-theta_minmax, theta_minmax, theta_state_size),
                np.linspace(-theta_dot_minmax, theta_dot_minmax, theta_dot_state_size),
            ],
        )
        if render_state:
            env.render()

        if not done:
            max_future_q = np.max(Q_TABLE[new_discrete_state[0], new_discrete_state[1]])
            current_q = Q_TABLE[curr_discrete_state[0], curr_discrete_state[1], action]
            new_q = current_q + LEARNING_RATE * (
                reward + DISCOUNT * max_future_q - current_q
            )
            Q_TABLE[curr_discrete_state[0], curr_discrete_state[1], action] = new_q

        curr_discrete_state = new_discrete_state
        episode_reward += reward

    ep_rewards.append(episode_reward)

    if not episode % EPISODE_DISPLAY:
        avg_reward = sum(ep_rewards[-EPISODE_DISPLAY:]) / len(
            ep_rewards[-EPISODE_DISPLAY:]
        )
        ep_rewards_table["ep"].append(episode)
        ep_rewards_table["avg"].append(avg_reward)
        ep_rewards_table["min"].append(min(ep_rewards[-EPISODE_DISPLAY:]))
        ep_rewards_table["max"].append(max(ep_rewards[-EPISODE_DISPLAY:]))
        print(
            f"Episode:{episode} avg:{avg_reward} min:{min(ep_rewards[-EPISODE_DISPLAY:])} max:{max(ep_rewards[-EPISODE_DISPLAY:])}, done = {done}"
        )

env.close()
plt.plot(ep_rewards_table["ep"], ep_rewards_table["avg"], label="avg")
plt.plot(ep_rewards_table["ep"], ep_rewards_table["min"], label="min")
plt.plot(ep_rewards_table["ep"], ep_rewards_table["max"], label="max")
plt.legend(loc=4)  # bottom right
plt.title("CartPole Q-Learning")
plt.ylabel("Average reward/Episode")
plt.xlabel("Episodes")
plt.savefig("cartpole_qlearning.png")
