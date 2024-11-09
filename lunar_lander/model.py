from copy import deepcopy
import torch
import torch.nn as nn
import numpy as np
from time import time
import os


def run_many(envs, agent, num_steps, last_obs_list, last_done_list, agent_states):

    batch_size = len(envs)
    env = envs[0]
    observations = np.zeros([batch_size, num_steps, *env.observation_space.shape])
    next_observations = np.zeros([batch_size, num_steps, *env.observation_space.shape])
    rewards = np.zeros([batch_size, num_steps])
    td_errors = np.zeros([batch_size, num_steps])
    values = np.zeros([batch_size, num_steps])
    logits = np.zeros([batch_size, num_steps, env.action_space.n])
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

    return observations, rewards, values, logits, dones, actions, next_observations, last_obs_list, last_done_list, agent_states


def train(optimizer, envs, agent, num_steps, num_training_steps, n_target_updates=20):

    agent.model.train()

    best_rewards_per_step = -1 # threshold
    rewards_ema = None
    ema_decay = 0.99

    agent_states = agent.model.zero_state(len(envs))
    last_obs_list = [env.reset()[0] for env in envs]
    last_done_list = [True for _ in envs]

    # random burnout
    for i, env in enumerate(envs):
        for k in range(i*5):
            env.step(np.random.randint(4))

    for i in range(num_training_steps):
        optimizer.zero_grad()

        if i % n_target_updates == 0:
            # update target networks
            agent.model.copy_target_networks()

        if np.random.randint(500 // num_steps) == 0:
            agent_states = agent.model.zero_state(len(envs))

        t0 = time()
        with torch.inference_mode():
            observations, rewards, values, logits, dones, actions, next_observations, last_obs_list, last_done_list, new_agent_states = \
                run_many(envs, agent, num_steps, last_obs_list, last_done_list, agent_states)

        observations = torch.tensor(observations).float()
        rewards = torch.tensor(rewards).float()
        values = torch.tensor(values).float()
        logits = torch.tensor(logits).float()
        dones = torch.tensor(dones, dtype=torch.bool)
        actions = torch.tensor(actions).long()
        next_observations = torch.tensor(next_observations).float()

        t_run = time()

        assert not agent.model.target_value_model[0].training
        losses = agent.model.losses(observations, rewards, values, logits, dones, actions, agent_states)
        loss = losses[0] + losses[1] + 1e-2 * losses[2]
        loss.backward()

        t_step = time()

        reward_per_step = float(rewards.mean().item())
        rewards_ema = ema_decay * rewards_ema + (1. - ema_decay) * reward_per_step if rewards_ema is not None else reward_per_step

        if i % 100 == 0:
            assert 100 % n_target_updates == 0, "expected multiple of target network updates, then rewards should be"
            if reward_per_step > best_rewards_per_step:
                    best_rewards_per_step = reward_per_step
                    agent.model.save()
                    print("saved. ", end="")
            loss_string = " ".join([f" {loss.item():0.03f} " for loss in losses])
            print(f"rewards @ {i} : {reward_per_step:0.03f} running mean {rewards_ema:0.03f} \t loss {loss_string} \t step {t_step - t_run:0.02f}s run {t_run - t0:0.02f}s")

        # perform optimizer step (after save to ensure that we save when target_network == network.eval()
        optimizer.step()

    return observations, rewards, dones, actions, next_observations

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

    def __init__(self, input_size, output_size, hidden_size=64):
        super(RNNModel, self).__init__()
        self.inputs = MyBatchNorm(input_size)
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers=3, batch_first=True, bidirectional=False)
        self.out = get_mlp(hidden_size, output_size)

    def zero_state(self, batch_size):
        h0 = torch.zeros([self.rnn.num_layers, batch_size, self.rnn.hidden_size])
        c0 = torch.zeros([self.rnn.num_layers, batch_size, self.rnn.hidden_size])
        return (h0,c0)



    def forward(self, x, state):

        if len(x.shape) == 2:
            x = x.unsqueeze(1)
            x = self.forward(x, state)
            return x.squeeze(1)

        x = self.inputs(x)
        x, final_state = self.rnn(x, state)
        x = self.out(x)
        return x, final_state

class RLModel(nn.Module):

    def __init__(self, env, gamma=0.99, lbda=0.95):
        super(RLModel, self).__init__()

        self.env = env

        self.policy_model = MLPModel(env.observation_space.shape[0]+1, env.action_space.n)
        self.value_model = MLPModel(env.observation_space.shape[0]+1, 1)

        self.copy_target_networks()
        self.register_buffer("_reward_var", torch.zeros((1,), dtype=torch.float32))

        self.gamma = gamma
        self.lbda = lbda

    def model_file_name(self):
        return "model.pt"

    def copy_target_networks(self):
        assert self.training # should not be necessary in eval mode
        self.target_policy_model = [deepcopy(self.policy_model)]
        self.target_value_model = [deepcopy(self.value_model)] # not saved in the pytorch A2CModel
        self.target_policy_model[0].eval() # put copies as eval
        self.target_value_model[0].eval()

    def normalize_rewards(self, rewards, epsilon=1e-8, momentum=0.9):
        if self._reward_var == 0:
            self._reward_var.data[0] = rewards.var()

        self._reward_var.data[0] = momentum * self._reward_var + (1-momentum) * rewards.var()

        return rewards / torch.clip(torch.sqrt(self._reward_var), min=epsilon)


    def zero_state(self, batch_size):
        return (self.policy_model.zero_state(batch_size), self.value_model.zero_state(batch_size))

    def action_selection(self, obs_list, done_list, states):
        o = torch.tensor(np.array(obs_list)).float()
        d = torch.tensor(np.array(done_list)).float().unsqueeze(1)
        x = torch.cat([o, d], 1)

        policy_model = self.target_policy_model[0] if self.training else self.policy_model
        value_model = self.target_value_model[0] if self.training else self.value_model

        policy_state, value_state = states
        logits, new_policy_state = policy_model(x, policy_state)
        value, new_value_state = value_model(x, value_state)

        D = torch.distributions.categorical.Categorical(logits=logits)
        action = D.sample()
        return action, logits, value, (new_policy_state, new_value_state)


    def losses(self, observations, rewards, value_old, logits_old, dones, actions, states=None):

        batch_size = observations.shape[0]
        policy_state, value_state = states if states is not None else self.zero_state(batch_size)

        xs = torch.cat([observations, dones.float().unsqueeze(2)], 2)
        logits, _ = self.policy_model(xs, policy_state)
        log_probs = torch.log_softmax(logits, -1)
        log_probs_old = torch.log_softmax(logits_old, -1)

        values_trainable, _ = self.value_model(xs, value_state)
        values_trainable = values_trainable.squeeze(2) #* self.value_scale

        # normalize rewards
        rewards = self.normalize_rewards(rewards)

        Rs = future_returns(rewards, dones, value_old, gamma=self.gamma).detach()

        action_oh = torch.nn.functional.one_hot(actions, self.env.action_space.n).float()

        log_pis = torch.einsum("bti, bti-> bt", log_probs, action_oh)
        log_pis_old = torch.einsum("bti, bti-> bt", log_probs_old, action_oh)

        #policy_loss = - ((Rs - value_old).detach() * log_pis).mean()
        A = generalized_advantage(rewards, dones, value_old, self.gamma, self.lbda)
        #A = (Rs - value_old).detach()
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

    def forward(self, x):
        raise NotImplemented()

if __name__ == "__main__":
    import gymnasium as gym
    from agent import Agent

    batch_size = 32
    num_steps = 100
    num_training_steps = 10_000

    envs = [gym.make("LunarLander-v2") for _ in range(batch_size)]

    agent =  Agent(envs[0])
    optimizer = torch.optim.AdamW(agent.model.parameters(), lr=1e-4)
    train(optimizer, envs, agent, num_steps, num_training_steps=num_training_steps)
    agent.model.save()
    print(f"saved agent for {num_training_steps}")