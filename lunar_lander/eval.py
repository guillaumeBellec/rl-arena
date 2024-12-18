import gymnasium as gym
import numpy as np
from heuristic.agent import Agent as HeuristicAgent
from agent import Agent


def eval(env, agent_cls):
    # Reset the environment to generate the first observation
    observation, info = env.reset(seed=42)

    total_rewards = []
    reward_list = []

    for episode in range(10):
        total_reward = 0
        observation, *_ = env.reset()
        agent = agent_cls(env)
        while True:
            action = agent.choose_action(observation)

            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            reward_list.append(reward)

            if terminated or truncated:
                break
                #observation, info = env.reset()

        total_rewards += [total_reward]

    return total_rewards, float(np.mean(reward_list))


if __name__ == "__main__":

    env = gym.make("LunarLander-v2", render_mode="human")
    total_rewards, rewards_per_step = eval(env, Agent)

    print(f"total rewards: {np.mean(total_rewards)} +- {np.std(total_rewards)}")
    print(f"rewards per step: {rewards_per_step}")

    env.close()