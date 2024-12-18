import torch
import numpy as np
from time import time, time_ns
from rl_commons import train
import gymnasium as gym
import json


class LunarEnv(gym.Env):

    def __init__(self, seed):
        super().__init__()
        self._env = gym.make("LunarLander-v2")
        self.seed = seed
        self.burn_in = 10
        self.verbose = False

        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        self.metadata = self._env.metadata
        self.spec = self._env.spec
        self.reward_range = self._env.reward_range
        self.render_mode = self._env.render_mode
        self.just_had_reset = False
        self.reset(seed=self.seed)

    def reset(self, **kwargs):

        obs, info = self._env.reset(**kwargs)
        self.step_count = 0

        b = self.burn_in
        n = (np.random.randint(b) + time_ns() + self.seed * 7) % b # use time_ns() but the number generator is synced on the threads
        if self.verbose:
            print(f"(seed = {self.seed}) reset burn out n={n}")
        for i in range(n):
            obs, reward, terminated, truncated, info = self._env.step(np.random.randint(self.action_space.n))

        self.just_had_reset = True
        return obs, info

    def close(self):
        return self._env.close()

    def step(self, action):
        self.just_had_reset = False
        total_rewards = 0
        self.step_count += 1

        obs, reward, terminated, truncated, info = self._env.step(action)
        total_rewards += reward
        done = terminated or truncated

        if done:
            #obs, info = self.reset() # already called
            if self.verbose:
                print(f"(seed = {self.seed}) done @ step: {self.step_count} total rewards {total_rewards}")

        #if self.verbose and total_rewards != 0:
        #    print(f"(seed = {self.seed}) step: {self.step_count} total rewards {total_rewards}")

        return obs, total_rewards, terminated, truncated, info

if __name__ == "__main__":
    import gymnasium as gym
    from agent import Agent
    import argparse
    from datetime import datetime

    from torch.utils.tensorboard import SummaryWriter

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_parallel_actors', type=int, default=16)
    parser.add_argument('--n_actor_steps', type=int, default=500)
    parser.add_argument("--date", type=str, default=datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))

    parser.add_argument('--resample_factor', type=int, default=1)
    parser.add_argument('--n_target_updates', type=int, default=20)
    parser.add_argument('--n_print', type=int, default=20) # number of target net updates

    parser.add_argument('--n_training_steps', type=int, default=100_000)
    parser.add_argument('--rps_saving_thr', type=float, default=1.0)
    parser.add_argument('--returns_saving_thr', type=float, default=200.)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lbda', type=float, default=0.93) # for ppo
    parser.add_argument('--alpha', type=float, default=0.01) #
    parser.add_argument('--lr', type=float, default=1e-4) #

    parser.add_argument('--n_learner_steps', type=int, default=10)
    parser.add_argument('--learner_batch_size', type=int, default=64)


    args= parser.parse_args()

    #envs = [[gym.make("LunarLander-v2") for _ in range(n_parallel_agents)] for _ in range(n_groups)]
    #envs = [[LunarEnv(seed=i + g * n_parallel_agents) for i in range(n_parallel_agents)] for g in range(n_groups)]
    #def make_env(seed):
    #    return lambda : LunarEnv(seed=seed)
    #envs = gym.vector.AsyncVectorEnv([make_env(seed=i) for i in  range(args.n_parallel_actors)])
    envs = gym.make_vec("LunarLander-v2", num_envs=args.n_parallel_actors, vectorization_mode="sync")

    agent = Agent(None)
    writer = SummaryWriter(f"runs/LunarLander_{args.date}")
    writer.add_text("args", json.dumps(args.__dict__,indent=4))
    optimizer = torch.optim.AdamW(agent.model.parameters(), lr=args.lr)
    train(optimizer, envs, agent, writer, args)