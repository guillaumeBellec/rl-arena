import gc
from copy import deepcopy

import gymnasium
import torch
import torch.nn as nn
import numpy as np
from time import time
import os
from dataclasses import dataclass

from fontTools.misc.bezierTools import namedtuple
from collections import deque
from pong_env import pre_process, PongEnv, roll_observation_buffer

from debugging_memory import memory_profile, get_object_size
from utils import make_vgg_layers, EmaVal, deepclone, deepdetach, RunStack, Simulation


def run_many(envs, agent, num_steps, last_obs_array, last_done_array, agent_states):
    profile = False

    # could be different?
    batch_size = envs.num_envs

    observations = np.zeros([batch_size, num_steps, *agent.model.downscaled_shape], dtype=np.uint8)
    #next_observations = np.zeros([batch_size, num_steps, *agent.model.downscaled_shape], dtype=np.float32)
    rewards = np.zeros([batch_size, num_steps], dtype=np.float32)
    values = np.zeros([batch_size, num_steps], dtype=np.float32)
    logits = np.zeros([batch_size, num_steps, agent.model.num_actions], dtype=np.float32)
    dones = np.zeros([batch_size, num_steps], dtype=bool)
    actions = np.zeros([batch_size, num_steps], dtype=np.int64)

    assert last_obs_array.shape[0] == last_done_array.shape[0] == batch_size, f"last_obs_array={last_obs_array.shape} last_done_array={last_done_array.shape} batch_size={batch_size}"

    dt_actions = []
    dt_envs = []
    dt_stacks = []

    for t in range(num_steps):
            t0 = time()
            chosen_actions, l, predicted_values, agent_states = agent.model.action_selection(last_obs_array, last_done_array, agent_states)
            t_action = time()

            #next_obs_array, reward, terminated, truncated, info = async_multi_step(envs, chosen_actions, agent.n_action_repeat) #, agent.n_action_repeat)
            new_I, reward, terminated, truncated, info = envs.step(chosen_actions) #, agent.n_action_repeat)
            next_done_array = np.logical_or(terminated, truncated)
            next_obs_array = roll_observation_buffer(pre_process(new_I), last_obs_array)

            t_envs = time()

            observations[:,t] = last_obs_array[:,-1] # last of buffer is most recent
            rewards[:,t] = reward
            values[:,t] = predicted_values.squeeze(-1)
            logits[:,t] = l.detach().cpu().numpy()
            dones[:,t] = next_done_array
            actions[:,t] = chosen_actions
            #next_observations[:,t] = next_obs_array

            # for the next step
            last_obs_array = next_obs_array
            last_done_array = next_done_array

            t_stack = time()

            dt_actions += [t_action - t0]
            dt_envs += [t_envs - t_action]
            dt_stacks += [t_stack - t_envs]

    if profile:
        print(f"dt_actions={np.sum(dt_actions):0.04f}s env={np.sum(dt_envs):0.04f} stack={np.sum(dt_stacks):0.04f}")
    return (observations, rewards, values, logits, dones, actions,), \
        last_obs_array, last_done_array, agent_states

def train(optimizer, envs_groups, agent, num_simu_steps, num_training_steps, seq_len=30, n_target_updates=25, n_prints=100):

    run_stack_size = seq_len // num_simu_steps # 60 will be the number of RNN steps
    #if n_target_updates is None: n_target_updates = len(env_groups) * run_stack_size
    n_prints = min(n_prints, num_training_steps)
    n_target_updates = min(n_target_updates, n_prints)

    agent.model.train()
    n_envs = envs_groups[0].num_envs

    best_rewards_per_step = 5 # threshold
    rewards_ema = EmaVal()
    losses_ema = [EmaVal() for _ in range(3)]

    # train off between training on the same batch and off-policy problems
    simulations = []
    for i_envs, envs in enumerate(envs_groups):

        # init the buffer
        I, _ = envs.reset(seed=[i_envs * n_envs + i for i in range(n_envs)])
        I = pre_process(I)
        last_obs_array = np.stack([I for _ in range(agent.model.n_observation_buffer)], 1)
        last_done_array = np.ones(n_envs, dtype=bool)

        run_stack = RunStack(run_stack_size) # 60 or 100 is a good number of lstm steps ?
        simu = Simulation(
            envs=envs,
            agent_states=agent.model.zero_state(n_envs),
            last_observations=last_obs_array,
            last_dones=last_done_array,
            run_stack=run_stack
        )

        simulations.append(simu)
    simulations = tuple(simulations)

    t_init = time()
    for i in range(num_training_steps):
        optimizer.zero_grad()

        simu : Simulation = simulations[i % len(simulations)]

        if i % n_target_updates == 0:
            # update target networks, for pong since we recycle recent runs, this is somewhat redundant.
            agent.copy_target_networks()

        t0 = time()
        with torch.inference_mode():
                agent_states = simu.agent_states
                if np.random.randint(500 // num_simu_steps) == 0:
                    agent_states = agent.model.zero_state(n_envs)  # sometimes reset the agent RNN states randomly

                new_run, last_obs_array, last_done_array, new_agent_states = \
                        run_many(simu.envs, agent, num_simu_steps, simu.last_observations, simu.last_dones, agent_states)

                # update, TODO: could be done in run_many?
                simu.run_stack.add(new_run)
                simu.last_observations = last_obs_array
                simu.last_dones = last_done_array
                simu.agent_states = new_agent_states

        t_run = time()

        # compute the loss
        observations, rewards, values, logits, dones, actions = simu.run_stack.get_stack_tensors()

        losses = agent.model.losses(observations, rewards, values, logits, dones, actions, deepclone(agent_states))
        loss = losses[0] + losses[1] + 1e-2 * losses[2]

        loss.backward()

        t_step = time()

        # Reporting with prints
        reward_per_step = float(rewards.mean().item())
        rewards_ema(reward_per_step)
        [ema(l) for ema,l in zip(losses_ema, losses)]

        if i % n_prints == 0:
            assert n_prints % n_target_updates == 0, f"expected multiple of target network updates, then rewards should be got n_prints={n_prints} and n_target={n_target_updates}"

            loss_string = " ".join([f" {ema.read():0.03f} " for ema in losses_ema])
            print(f"rewards @ {i}: \t {reward_per_step:0.03f} running mean {rewards_ema.read():0.03f} \t|\t loss {loss_string} \t|\t step {t_step - t_run:0.02f}s run {t_run - t0:0.02f}s ")
            if reward_per_step > best_rewards_per_step:
                best_rewards_per_step = reward_per_step
                agent.model.save()
                print("saved. ")

            #gc.collect()
            #next_memory = memory_profile()
            #memory_changes = {
            #    key: (next_memory[key] - current_memory[key]) // 1024 // 1024
            #    for key in current_memory
            #}
            #current_memory = next_memory

            #print("\tmemory diff: ", memory_changes, "\t total: ", current_memory["process_memory"] // 1024 // 1024)

        # perform optimizer step (after save to ensure that we save when target_network == network.eval()
        optimizer.step()

    print(f"Training finished {num_training_steps} steps in {(time() - t_init) / 3600} hours.")

def td_errors(rewards, dones, values, gamma):
    next_values = values[:,1:]

    next_values = torch.where(dones[:,:-1], torch.zeros_like(next_values), next_values)

    values = values[:,:-1]
    rewards = rewards[:,:-1]
    td_errors = rewards + gamma * next_values - values
    return td_errors

def generalized_advantage(rewards, dones, values, gamma, lbda):
    td_err = td_errors(rewards, dones, values, gamma)
    advantages = []
    gae = torch.zeros_like(td_err[:, -1])  # Initialize with zeros for last timestep

    # Calculate GAE backwards
    for t in reversed(range(td_err.shape[1])):
        gae = torch.where(dones[:,t], torch.zeros_like(gae), gae)
        gae = td_err[:, t] + gamma * lbda * gae
        advantages.append(gae)

    # Reverse the accumulated advantages and stack them
    advantages = torch.stack(advantages[::-1], dim=1)

    return advantages

def future_returns(rewards, dones, values, gamma):
    B, T = rewards.shape
    if len(values.shape) == 3:
        values = values.squeeze(2)

    # Initialize returns array
    Rs = torch.zeros_like(rewards)

    # Bootstrap from the last value unless the episode is done
    last_value = torch.where(dones[:, -1], torch.zeros_like(values[:, -1]), values[:, -1])
    Rs[:, -1] = rewards[:, -1] + gamma * last_value

    # Backwards value iteration
    for t in range(T - 2, -1, -1):  # Work backwards from T-2 to 0
        next_value = Rs[:, t + 1]
        # Zero out next value if current step leads to done
        next_value = torch.where(dones[:, t], torch.zeros_like(next_value), next_value)
        Rs[:, t] = rewards[:, t] + gamma * next_value

    return Rs

def future_returns_old(rewards, dones, values, gamma):
    B, T = rewards.shape
    if len(values.shape) == 3:
        values = values.squeeze(2)

    Rs = torch.zeros_like(rewards)
    Rs[:,-1] = torch.where(dones[:,-1], rewards[:,-1],values[:,-1])
    for i in range(1,T):
        t = T-i-1
        next_R = Rs[:,t+1]
        next_R = torch.where(dones[:,t], torch.zeros_like(next_R), next_R)
        R = gamma * next_R + rewards[:,t]
        Rs[:,t] = R

    return Rs

class MyBatchNorm(nn.Module):

    def __init__(self, n, **kwargs):
        super().__init__()
        self.bn = nn.BatchNorm1d(n, **kwargs)

    def forward(self, x):
        if len(x.shape) == 2: return self.bn(x)
        if len(x.shape) == 3:
            return self.bn(x.transpose(1, 2)).transpose(1, 2)

def get_mlp(n_in, n_out, n_hidden=64, n_hidden_layers=1, norm=True):

    layers = []
    for i in range(n_hidden_layers):
        layers += [nn.Linear(n_in if i == 0 else n_hidden, n_hidden)]
        if norm: layers += [MyBatchNorm(n_hidden)]
        layers += [nn.ReLU()]

    layers += [nn.Linear(n_hidden, n_out)]

    return nn.Sequential(*layers)

class MLPModel(nn.Module):

    def __init__(self, input_size, output_size, hidden_size=128):
        super(MLPModel, self).__init__()
        self.inputs = MyBatchNorm(input_size)
        self.mlp = get_mlp(input_size, output_size, hidden_size, 2, True)

    def zero_state(self, batch_size):
        return (None,)

    def forward(self, x, state):

        x = self.inputs(x)
        x = self.mlp(x)

        return x, state

class RNNModel(nn.Module):

    def __init__(self, dim, hidden_size=64, residual=True):
        super(RNNModel, self).__init__()
        self.inputs = MyBatchNorm(dim)
        self.residual = residual
        self.rnn = nn.LSTM(dim, hidden_size, num_layers=2, batch_first=True, bidirectional=False)
        self.out = nn.Sequential(MyBatchNorm(hidden_size),nn.Linear(hidden_size, dim)) #get_mlp(hidden_size, output_size)

    def zero_state(self, batch_size):
        h0 = torch.zeros([self.rnn.num_layers, batch_size, self.rnn.hidden_size])
        c0 = torch.zeros([self.rnn.num_layers, batch_size, self.rnn.hidden_size])
        return (h0,c0)


    def forward(self, inputs, state):

        if len(inputs.shape) == 2:
            x = inputs.unsqueeze(1)
            x, state = self.forward(x, state)
            return x.squeeze(1), state

        x = self.inputs(inputs)
        x, final_state = self.rnn(x, state)
        x = self.out(x)
        if self.residual: x = x + inputs
        return x, final_state

class RLModel(nn.Module):

    def __init__(self, get_target_network_fn, gamma=0.97, lbda=0.93):
        super(RLModel, self).__init__()

        self.get_target_network_fn = get_target_network_fn

        self.downscaled_shape = [52, 40 , 3]
        self.n_observation_buffer = 3
        self.num_actions = 6 # for simplicy
        self.cnn_dim = 384 #self.cnn(dummy_input).numel()
        self.hidden_dim = 128

        self.cnn = make_vgg_layers([8, "M", 16, "M", 32, "M", 64, "M"], in_channels=3 * self.n_observation_buffer, batch_norm=True)
        self.fc = nn.Sequential(nn.Linear(self.cnn_dim+1, self.hidden_dim), MyBatchNorm(self.hidden_dim))
        self.rnn = RNNModel(self.hidden_dim)
        
        self.policy_head = get_mlp(self.hidden_dim, self.num_actions)
        self.value_head = get_mlp(self.hidden_dim, 1)

        self.register_buffer("_reward_var", torch.zeros((1,), dtype=torch.float32))

        self.gamma = gamma
        self.lbda = lbda

    def model_file_name(self):
        return "model.pt"

    def normalize_rewards(self, rewards, epsilon=1e-8, momentum=0.99):
        if self._reward_var == 0:
            self._reward_var.data[0] = rewards.var()

        self._reward_var.data[0] = momentum * self._reward_var + (1-momentum) * rewards.var()

        return rewards / torch.clip(torch.sqrt(self._reward_var), min=epsilon)


    def zero_state(self, batch_size):
        return self.rnn.zero_state(batch_size)

    def action_selection(self, obs_array, done_array, states):
        logits, values, final_states = self.forward(obs_array, done_array, states, True)

        D = torch.distributions.categorical.Categorical(logits=logits)
        action = D.sample()
        return action, logits, values, final_states

    def make_buffer_dimension(self, observations):
        B, T, *_ = observations.shape
        d = self.n_observation_buffer
        assert T > d
        zz = torch.zeros_like(observations[:,0:1])
        buffer = []
        for i in range(d):
            o_delayed = torch.cat([observations[:,:T-i], zz.repeat([1, i, 1, 1, 1])], 1)
            buffer += [o_delayed]

        buffer.reverse() # in-place reverse list order
        return torch.stack(buffer, 2) # dimension 2 is for buffer

    def losses(self, observations, rewards, value_old, logits_old, dones, actions, states=None):

        observations_buffer = self.make_buffer_dimension(observations)
        logits, values_trainable, final_states = self.forward(observations_buffer, dones, states, False)

        # log small formatting
        log_probs = torch.log_softmax(logits, -1)
        log_probs_old = torch.log_softmax(logits_old, -1)
        values_trainable = values_trainable.squeeze(2) #* self.value_scale

        # normalize rewards
        rewards = self.normalize_rewards(rewards)

        # real RL losses
        Rs = future_returns(rewards, dones, value_old, gamma=self.gamma).detach()

        action_oh = torch.nn.functional.one_hot(actions, self.num_actions).float()

        log_pis = torch.einsum("bti, bti-> bt", log_probs, action_oh)
        log_pis_old = torch.einsum("bti, bti-> bt", log_probs_old, action_oh)

        #policy_loss = - ((Rs - value_old).detach() * log_pis).mean()
        A = generalized_advantage(rewards, dones, value_old, self.gamma, self.lbda)

        epsi = 0.2
        r = torch.exp(log_pis - log_pis_old.detach())[:,:-1]
        r_clipped = r.clip(min=1-epsi, max=epsi)
        #a2c_loss = - (A * log_pis).mean()
        ppo_loss = - (torch.minimum(r * A ,r_clipped * A)).mean()
        value_loss = (Rs.detach() - values_trainable).square().mean()

        entropy_loss = - log_pis.mean()

        return (ppo_loss, value_loss, entropy_loss)

    def save(self,):
        torch.save(self.state_dict(), self.model_file_name())

    def load(self):
        file_name = self.model_file_name()
        dirs = list(os.listdir("./"))
        if file_name in dirs:
            file_path = f"./{file_name}"
            self.load_state_dict(torch.load(file_path))
        else:
            print(f"Warnings: file {file_name} not found dir has: {dirs}")

    def forward(self, obs_array, done_array, states, use_target):
        if isinstance(obs_array, np.ndarray):
            obs_array = torch.from_numpy(obs_array)

        if isinstance(done_array, np.ndarray):
            done_array = torch.from_numpy(done_array)

        if obs_array.dtype == torch.uint8:
            obs_array = obs_array.float() / 256.0 # max is one.

        assert len(obs_array.shape) > 4, f"got shape {obs_array.shape}"
        assert [i for i in obs_array.shape[-3:]] == self.downscaled_shape, f"got shape {obs_array.shape} -> {obs_array.shape[-3:]} expected {self.downscaled_shape}"
        assert obs_array.shape[-4] == self.n_observation_buffer, f"got shape {obs_array.shape} but self.n_observation_buffer={self.n_observation_buffer}"

        if len(obs_array.shape) == 5: # B, d, W, H, C (no time dimension):
            assert len(done_array.shape) == 1, f"got obs_array={obs_array.shape} | done_array={done_array.shape}"
            obs_array = obs_array[:, None]
            done_array = done_array[:, None]
            logits, value, final_states = self.forward(obs_array, done_array, states, use_target)
            return logits[:,0], value[:,0], final_states

        if use_target:
            with torch.no_grad():
                return self.get_target_network_fn().forward(obs_array, done_array, states, False)

        # CNN
        B, T, d, W, H, C = obs_array.shape
        o = torch.cat([obs_array[:,:,i] for i in range(d)], -1) # buffer buffer on channel dimension
        o = o.view(B * T, W, H, C * d) # flatten time dimension
        o = self.cnn(o.permute([0,3,1,2])).reshape([B, T, self.cnn_dim])

        # Merge visual input and done flags
        d = done_array.float().unsqueeze(-1)
        x = torch.cat([o, d], -1)
        x = self.fc(x) #

        # RNN
        x, final_states = self.rnn(x, states) #

        # outputs
        logits = self.policy_head(x)
        value = self.value_head(x)

        return logits, value, final_states


if __name__ == "__main__":
    import gymnasium as gym
    from agent import Agent

    import ale_py # only needed for single player training
    gym.register_envs(ale_py)

    # parallelization parameters
    batch_size = 8 # batch-size (keep it low if cpu only)
    env_groups = 8 # separate group of environments to alternative and avoid overfitting one history, too high will be off-policy
    num_steps = 15 # num steps so simulate in one group between each gradient descent step

    num_training_steps = 100_000

    agent =  Agent(None)

    # This option goes with env.step in run many, faster on my thinkpad laptop
    env_groups = [gym.vector.AsyncVectorEnv([lambda : PongEnv(num_steps=agent.n_action_repeat) for i in range(batch_size)]) for _ in range(env_groups)]

    # This option goes with async_multi_step
    #envs = gym.make_vec("ALE/Pong-v5", num_envs=batch_size, vectorization_mode="async")

    optimizer = torch.optim.AdamW(agent.model.parameters(), lr=1e-4)
    train(optimizer, env_groups, agent, num_steps, num_training_steps=num_training_steps)

    agent.model.save()
    print(f"saved agent for {num_training_steps}")