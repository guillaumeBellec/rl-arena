from copy import deepcopy
import torch
import torch.nn as nn
import numpy as np
from time import time, time_ns
from rl_commons import MyBatchNorm, get_mlp, future_returns, generalized_advantage, RLModel, Simulation, deepshape
import os
from utils import EmaVal, to_torch
import gymnasium as gym


def run_many(envs, agent, num_steps, last_obs_list, last_done_list, agent_states):

    batch_size = len(envs)
    env = envs[0]
    observations = np.zeros([batch_size, num_steps, *env.observation_space.shape], dtype=np.float32)
    next_observations = np.zeros([batch_size, num_steps, *env.observation_space.shape], dtype=np.float32)
    rewards = np.zeros([batch_size, num_steps], dtype=np.float32)
    td_errors = np.zeros([batch_size, num_steps])
    values = np.zeros([batch_size, num_steps], dtype=np.float32)
    logits = np.zeros([batch_size, num_steps, env.action_space.n], dtype=np.float32)
    dones = np.zeros([batch_size, num_steps], dtype=bool)
    actions = np.zeros([batch_size, num_steps], dtype=int)

    assert len(last_obs_list) == len(last_done_list) == len(envs)

    for t in range(num_steps):

            chosen_actions, l, predicted_values, agent_states = agent.model.action_selection(last_obs_list, last_done_list, agent_states)

            next_obs_list = []
            next_done_list = []

            for b, env in enumerate(envs):
                action = int(chosen_actions[b])
                value = float(predicted_values[b])

                next_obs, reward, terminated, truncated, info = env.step(action)
                done = truncated or terminated

                if done:
                    next_obs, *_ = env.reset()


                # stop values
                observations[b][t] = last_obs_list[b]
                rewards[b][t] = reward
                values[b][t] = value
                logits[b][t] = l[b].detach().cpu().numpy()
                dones[b][t] = done
                actions[b][t] = action
                next_observations[b][t] = next_obs

                # for the next step
                next_done_list += [done]
                next_obs_list += [next_obs]
            last_obs_list = next_obs_list
            last_done_list = next_done_list

    return (observations, rewards, values, logits, dones, actions), last_obs_list, last_done_list, agent_states


class LunarEnv(gym.Env):

    def __init__(self, seed, **kwargs):
        self._env = gym.make("LunarLander-v2", autoreset=True)
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        self.seed = seed
        self.reset(seed=self.seed, burn_in=100)

    def close(self):
        return self._env.close()

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        done = truncated or terminated
        if done:
            obs, info = self.reset()

        return obs, reward, terminated, truncated, info

    def reset(self, burn_in=0, **kwargs):
        # random noise generation
        obs, info = self._env.reset(**kwargs)

        if burn_in > 0:
            for _ in range(np.random.randint(burn_in)):
                obs, reward, terminated, truncated, info = self.step(np.random.randint(self._env.action_space.n))

        return obs, info

def train(optimizer, env_groups, agent, num_steps, num_training_steps, n_target_updates=20, n_prints=100):
    n_prints = max(n_target_updates * np.ceil(n_prints / n_target_updates), n_prints)
    n_prints = max(n_target_updates, n_prints)

    agent.model.train()
    n_actors = len(env_groups)
    envs0 = env_groups[0]
    n_envs = envs0.num_envs if not isinstance(envs0, list) else len(envs0)

    best_rewards_per_step = 0.5 # threshold
    rewards_ema = EmaVal()

    simulations = [Simulation(envs=envs, agent=agent) for envs in env_groups]

    for i in range(num_training_steps):
        optimizer.zero_grad()

        if i % n_target_updates == 0:
            # update target networks
            agent.copy_target_network()

        # ACTOR Part
        t0 = time()
        simu = simulations[i  % len(simulations)]
        if i == 0 or np.random.randint(10) == 0:
            # regularly reset the whole state (@ runtime need to be ready for the first step)
            last_obs = simu.agent_state.observation_buffer.last()
            simu.agent_state = agent.zero_state(n_envs)
            simu.agent_state.observation_buffer.add(last_obs)

        run_tensors, init_agent_states = simu.run_many(num_steps)
        rewards = run_tensors[2] # should be the second here
        dt_run = time() - t0

        # LEARNER:
        tensors_stacked, agent_state_stacked = Simulation.get_stacked_torch_tensors(simulations)

        t0 = time()
        assert agent.model.training
        losses = agent.model.losses(agent_state_stacked, *tensors_stacked)
        loss = losses["policy"] + losses["value"] + 1e-2 * losses["entropy"]
        loss.backward()
        dt_step = time() - t0

        reward_per_step = float(rewards.mean().item())
        rewards_ema(reward_per_step)

        if i % n_prints == 0:
            assert n_prints % n_target_updates == 0, "expected multiple of target network updates, then rewards should be"
            if reward_per_step > best_rewards_per_step and i > 1_000:
                    best_rewards_per_step = reward_per_step
                    agent.model.save()
                    print("saved. ", end="")
            loss_string = " ".join([f" {key} {loss.item():0.03f} " for key, loss in losses.items()])
            print(f"rewards @ {i} : {reward_per_step:0.03f} running mean {rewards_ema.read():0.03f} \t loss {loss_string} \t step {dt_step:0.02f}s run {dt_run:0.02f}s")

        # perform optimizer step (after save to ensure that we save when target_network == network.eval()
        optimizer.step()
        agent.model._step_count.data += 1 # step count

if __name__ == "__main__":
    import gymnasium as gym
    from agent import Agent

    n_parallel_agents = 16 # lower than that, run time saturates. Reduce number of steps instead.
    n_groups = 2 # training batch size = n_groups x n_parallel_agents,
    num_steps = 100
    n_target_updates = 20 * n_groups # number of training steps between target network updates
    num_training_steps = 100_000

    #envs = [[gym.make("LunarLander-v2") for _ in range(n_parallel_agents)] for _ in range(n_groups)]
    envs = [[LunarEnv(seed=i + g * n_parallel_agents) for i in range(n_parallel_agents)] for g in range(n_groups)]
    #envs = [gym.make_vec("LunarLander-v2", num_envs=n_parallel_agents, vectorization_mode="sync") for _ in range(n_groups)]
    #envs = [gym.vector.AsyncVectorEnv([lambda : LunarEnv() for i in range(n_parallel_agents)], shared_memory=False) for g in range(n_groups)]

    agent =  Agent(None)
    agent.model.train()
    optimizer = torch.optim.AdamW(agent.model.parameters(), lr=1e-4)
    train(optimizer, envs, agent, num_steps, num_training_steps=num_training_steps, n_target_updates=n_target_updates)
    agent.model.save()
    print(f"saved agent for {num_training_steps}")