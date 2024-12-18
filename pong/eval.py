import gymnasium as gym
import numpy as np
from agent import Agent
import matplotlib.pyplot as plt

from pettingzoo.atari import pong_v3

import ale_py
gym.register_envs(ale_py)

def eval(agent_cls):
    # Reset the environment to generate the first observation
    env = gym.make("ALE/Pong-v5", render_mode="human")
    observation, info = env.reset(seed=42)


    total_rewards = []
    for episode in range(10):
        total_reward = 0
        observation, *_ = env.reset()
        agent = agent_cls(env)

        while True:
            action = agent.choose_action(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if terminated or truncated:
                break
                #observation, info = env.reset()

        total_rewards += [total_reward]

    return total_rewards

def eval_versus_random(agent_cls):

    env = pong_v3.env(render_mode="human")
    env.reset(seed=42)
    print("env agents:", env.agents)

    cum_reward = {agent: 0 for agent in env.agents}

    # Instantiate your agent (assuming first agent is controlled by your agent)
    agent_dict = {
        env.agents[0]: agent_cls(env, env.agents[0]), # create agent object
        env.agents[1]: agent_cls(env, env.agents[1]), # random agent
    }

    for player_name in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        cum_reward[player_name] += reward
        if termination or truncation:
            break

        current_agent = agent_dict[player_name]

        if current_agent is None:
            action = env.action_space(player_name).sample()
        else:
            action = current_agent.choose_action(observation)

        # env.step
        env.step(action)

    env.close()
    print(cum_reward)

if __name__ == "__main__":

    total_rewards = eval_versus_random(Agent)
    print(total_rewards)

    print(f"total rewards: {np.mean(total_rewards)} +- {np.std(total_rewards)}")