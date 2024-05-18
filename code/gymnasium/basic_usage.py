# Source - https://gymnasium.farama.org/content/basic_usage/
import gymnasium as gym

# Prints all environments that are available
# print(gym.envs.registry.keys())


# Creates an environment
env = gym.make("CartPole-v1", render_mode="human")

# The "agent-environment loop"
obs, info = env.reset()
for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()
