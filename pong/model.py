import os

from rl_commons import MyBatchNorm, RNNModel, train, MLPModel
import argparse

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"]="python"

import torch
import torch.nn as nn
import numpy as np
from time import time
import torch.nn.functional as F

from pong_env import PongEnv

from rl_commons import Simulation, RLModel, get_mlp, prepare_tensors
from utils import EmaVal, to_scalar, to_torch, deep_torch_stack
from torch.utils.tensorboard import SummaryWriter
import random

class PongCNN(nn.Module):
    def __init__(self, dim_out=128, n_channels=3):  # Pong has 6 possible actions
        super(PongCNN, self).__init__()

        # Convolutional layers with BatchNorm
        channels = [16, 32, 64] #[32, 64, 64]
        self.conv1 = nn.Conv2d(n_channels, channels[0], kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(channels[0]) #32)

        self.conv2 = nn.Conv2d(channels[0], channels[1], kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(channels[1]) #64)

        self.conv3 = nn.Conv2d(channels[1], channels[2], kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(channels[2])

        # Layer norm for the fully connected layer
        self.fc1 = nn.Linear(channels[2] * 7 * 7, 512)
        self.ln1 = MyBatchNorm(512)

        self.fc2 = nn.Linear(512, dim_out)
        self.ln2 = MyBatchNorm(dim_out)

    def forward(self, x):
        B, T, d, C, W, H = x.shape
        x = x.view(B * T, d * C, W, H)

        # Convolutional layers with BatchNorm and ReLU
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # Flatten
        x = x.view(B, T, -1) # dim 6 * 7 * 7

        # Fully connected layers with LayerNorm
        x = F.relu(self.ln1(self.fc1(x))) # dim 512
        x = F.relu(self.ln2(self.fc2(x))) # dim 128

        return x

class PongModel(RLModel):

    def define_models(self):
        #input_size = math.prod(self.observation_shape) * self.n_observation_buffer + 1
        self.embed_dim = 32
        self.value_model_dim = 1 #RegressionAsClassificationLoss().nbins if args.racl else 1

        self.player_context_embedding = nn.Embedding(1, self.embed_dim)
        self.is_done_embedding = nn.Embedding(1, self.embed_dim)
        self.action_embedding = nn.Embedding((self.num_actions+1) * self.n_observation_buffer, self.embed_dim)

        self.value_model_dim=1
        dim_cnn = 128
        self.cnn = PongCNN(dim_cnn, self.n_observation_buffer)
        self.value_model = MLPModel((dim_cnn, self.embed_dim), self.value_model_dim, 128, 2, norm="batch")
        self.policy_model = MLPModel((dim_cnn, self.embed_dim), self.num_actions, 128, 2, norm="batch")
        self.model_list = [self.cnn, self.policy_model, self.value_model]
        return

    def forward(self, obs_array, step_counts, last_action, model_state):
        B, T, d, C, W, H = obs_array.shape

        obs_array = to_torch(obs_array) #.reshape(B, T, d, C, W, H) #.float() #
        context = self.context_embedding(step_counts, last_action)
        if self.player_index > 0:
            context += self.player_context_embedding.weight * torch.tensor(self.player_index) # TODO: add possibility for player 1,2

        # outputs

        x = self.cnn(obs_array)
        x = (x, context)
        # TODO: Check if it's better to detach the policy here? Maybe good to avoid gradient mixing but...
        logits, policy_model_state = self.policy_model(x, model_state["policy"])
        value, value_model_state = self.value_model(x, model_state["value"])
        final_states = {
            "policy": policy_model_state,
            "value": value_model_state
        }

        return logits, value, final_states


if __name__ == "__main__":
    import gymnasium as gym
    from agent import Agent
    from datetime import datetime

    import ale_py # only needed for single player training
    gym.register_envs(ale_py)


    parser = argparse.ArgumentParser()
    parser.add_argument('--n_parallel_actors', type=int, default=8)
    parser.add_argument('--n_actor_steps', type=int, default=300)
    parser.add_argument("--date", type=str, default=datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))

    parser.add_argument('--resample_factor', type=int, default=5)
    parser.add_argument('--n_target_updates', type=int, default=5)
    parser.add_argument('--n_print', type=int, default=5) # number of target net updates

    parser.add_argument('--n_training_steps', type=int, default=1_000_000)
    parser.add_argument('--returns_saving_thr', type=float, default=0.)
    parser.add_argument('--gamma', type=float, default=0.98)
    parser.add_argument('--lbda', type=float, default=0.93) # for ppo
    parser.add_argument('--alpha', type=float, default=0.01) #

    parser.add_argument('--n_learner_steps', type=int, default=10)
    parser.add_argument('--learner_batch_size', type=int, default=16)
    args= parser.parse_args()

    agent =  Agent(None)

    # This option goes with env.step in run many, faster on my thinkpad laptop that the default parallelized atari
    def make_env(seed):
        return lambda: PongEnv(num_steps=agent.n_action_repeat, seed=seed)

    #n_actor_groups = gym.vector.AsyncVectorEnv(
    #    [make_env(seed=i) for i in range(args.n_parallel_actors)],
    #    shared_memory=True,
    #    copy=True,)
    #env_groups = [[PongEnv(num_steps=agent.n_action_repeat, seed=i + numb_parallel_agents * i_g) for i in
    #     range(numb_parallel_agents)] for i_g in range(env_groups)]

    # This option goes with async_multi_step
    envs = gym.make_vec("ALE/Pong-v5", num_envs=args.n_parallel_actors, vectorization_mode="async")

    optimizer = torch.optim.AdamW(agent.model.parameters(), lr=2e-4)
    writer = SummaryWriter(f"runs/PongAgent_{agent.name}")

    train(optimizer, envs, agent, writer, args)

    #agent.model.save()
    print(f"saved agent for {args.n_training_steps}")