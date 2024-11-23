import torch
import torch.nn as nn
import numpy as np
import os
import math
from collections import deque
from time import time
from dataclasses import dataclass
from copy import deepcopy
from typing import List, Type, Tuple, Any, Dict

from lunar_lander.utils import deepconcatenate, deeppermute
from utils import to_torch, deepstack, deepshape, to_numpy, deepdetach


@dataclass
class ObservationBuffer:
    _deque : deque
    size : int
    shp : tuple

    @staticmethod
    def init(observation_shape, maxlen):
        # this is not a constructor so that we can reuse the deepconcate etc... implemented for generic dataclass
        _deque = deque(maxlen=maxlen)
        size = maxlen
        shp = observation_shape
        return ObservationBuffer(_deque, size, shp)

    def make_buffer_dimension(self, tensor):
        B, T, *_ = tensor.shape

        # last is most recent
        pad = tensor[:,0:1] # pad left with first element
        tensor_list = []
        for d in range(self.size):
            repeat_shp = [1 for _ in tensor.shape]
            repeat_shp[1] = d
            t = torch.cat([pad.repeat(repeat_shp), tensor[:,:T-d]], 1)
            tensor_list += [t]

        tensor_list.reverse()
        return torch.stack(tensor_list, 2) # B, T, d ...

    def last(self):
        assert len(self._deque) == self.size
        return self._deque[-1]

    def add(self, observation):
        assert observation.shape[1:] == self.shp, f"got shape {observation.shape} expected {self.shp}"
        if len(self._deque) == 0:
            for i in range(self.size-1): # file with first element at first.
                self._deque.append(observation)
        self._deque.append(observation)

    def numpy_stack(self, axis=1):
        assert len(self._deque) == self.size
        elements = list(self._deque)
        assert type(elements[0]) == np.ndarray, f"got queue of types: {[type(e) for e in elements]}"
        return np.stack(elements, axis=axis)


@dataclass
class AgentState:
    observation_buffer: ObservationBuffer
    #last_obs_array : any
    last_is_done: any
    model_state: any


class Simulation:

    def __init__(self, agent, envs):
        self.envs = envs
        self.agent = agent

        with torch.inference_mode():
            n_envs = envs.num_envs if not isinstance(envs, list) else len(envs)
            self.agent_state = agent.zero_state(n_envs)
            new_I, _ = self.env_reset()
            new_I = agent.preprocess_frame(new_I)
            self.agent_state.observation_buffer.add(new_I)
            # self.agent_state.last_obs_array = new_I
            # for i in range(self.agent_state.observation_buffer.size):
            #    self.agent_state.observation_buffer.add(new_I)

        self.last_data = None # some kind or replay buffer?

    def env_step(self, actions):
        if isinstance(self.envs, list):
            # list of envs:
            obs_list = []
            reward_list = []
            terminated_list = []
            truncated_list = []
            infos = []
            for i in range(len(self.envs)):
                obs, reward, terminated, truncated, info = self.envs[i].step(actions[i])

                done = terminated or truncated
                if done:
                    obs, *_ = self.envs[i].reset()

                obs_list += [obs]
                reward_list += [reward]
                terminated_list += [terminated]
                truncated_list += [truncated]
                infos += [info]

            return tuple([np.stack(t) for t in (obs_list, reward_list, terminated_list, truncated_list)] + [infos])
        else:
            return self.envs.step(actions)

    def env_reset(self, **kwargs):
        if isinstance(self.envs, list):
            # list of envs:
            new_Is = []
            infos = []
            for e in self.envs:
                new_I, info = e.reset(**kwargs)
                new_Is += [new_I]
                infos += [info]
            return np.stack(new_Is, 0), infos
        else:
            return self.envs.reset(**kwargs)

    @staticmethod
    def get_stacked_torch_tensors(simulations : List[Any]):

        tensor_stack = []
        agent_state_stack = []
        for simu in simulations:
            if simu.last_data is not None:
                tensors, init_agent_states = simu.last_data
                tensor_stack += [tensors]
                agent_state_stack += [init_agent_states]

        tensor_stack = deepconcatenate(to_torch(tensor_stack), dim=0)
        agent_state = deepconcatenate(to_torch(agent_state_stack), dim=0)
        return tensor_stack, agent_state

    @torch.inference_mode()
    def run_many_with_env_list(self, num_steps):
        envs = self.envs
        agent = self.agent
        agent_state = self.agent_state

        init_state = deepcopy(self.agent_state)

        batch_size = len(envs)
        env = envs[0]
        observations = np.zeros([batch_size, num_steps, *env.observation_space.shape], dtype=np.float32)
        rewards = np.zeros([batch_size, num_steps], dtype=np.float32)

        values = np.zeros([batch_size, num_steps], dtype=np.float32)
        logits = np.zeros([batch_size, num_steps, env.action_space.n], dtype=np.float32)
        dones = np.zeros([batch_size, num_steps], dtype=bool)
        actions = np.zeros([batch_size, num_steps], dtype=int)

        last_obs_list = agent_state.last_obs_array
        last_done_list = agent_state.last_is_done
        model_state = agent_state.model_state
        assert len(last_obs_list) == len(last_done_list) == len(envs)

        for t in range(num_steps):

            #chosen_actions, l, predicted_values, new_model_state = agent.model.action_selection(self.agent_state)
            chosen_actions, l, predicted_values, model_state = agent.model.action_selection(last_obs_list, last_done_list, model_state)

            next_obs_list, reward_list, terminated, truncated, _ = self.env_step(chosen_actions.detach().cpu().numpy())
            next_done_list = np.logical_or(terminated, truncated)

            observations[:,t] = last_obs_list
            rewards[:,t] = reward_list
            values[:,t] = predicted_values.squeeze(-1) if len(predicted_values.shape) == 2 else predicted_values
            logits[:,t] = l.detach().cpu().numpy()
            dones[:,t] = next_done_list
            actions[:,t] = chosen_actions

            # next iteration
            last_obs_list = next_obs_list
            last_done_list = next_done_list

            #self.agent_state.last_obs_array = next_obs_list
            #self.agent_state.last_is_done = next_done_list
            #self.agent_state.model_state = new_model_state

        # record new state
        self.agent_state = AgentState(
            last_obs_array=last_obs_list,
            last_is_done=last_done_list,
            model_state=model_state,
        )

        return (observations, rewards, values, logits, dones, actions), init_state

    @torch.inference_mode()
    def run_many(self, num_steps : int):
        profile = False
        agent = self.agent
        envs = self.envs
        agent_state : AgentState = self.agent_state

        init_agent_state = deepcopy(to_numpy(self.agent_state)) # store for returning this state, probably useless

        # could be different?
        n_envs = envs.num_envs if not isinstance(envs, list) else len(envs) # parallel env objects

        observations = np.zeros([n_envs, num_steps, *agent.model.observation_shape], dtype=np.float32)
        rewards = np.zeros([n_envs, num_steps], dtype=np.float32)
        values = np.zeros([n_envs, num_steps], dtype=np.float32)
        logits = np.zeros([n_envs, num_steps, agent.model.num_actions], dtype=np.float32)
        dones = np.zeros([n_envs, num_steps], dtype=bool)
        actions = np.zeros([n_envs, num_steps], dtype=np.int64)

        dt_actions = []
        dt_envs = []
        dt_stacks = []

        for t in range(num_steps):
            t0 = time()
            #chosen_actions, l, predicted_values, new_model_states = agent.model.action_selection(self.agent_state)

            chosen_actions, l, predicted_values, new_model_state = agent.model.action_selection(
                agent_state.observation_buffer.numpy_stack(),
                agent_state.last_is_done,
                to_torch(agent_state.model_state),
            )
            chosen_actions = chosen_actions.detach().cpu().numpy()

            t_action = time()

            new_I, reward, terminated, truncated, info = self.env_step(chosen_actions) #, agent.n_action_repeat)
            new_I = agent.preprocess_frame(new_I) # possible downscaling
            next_done_array = np.logical_or(terminated, truncated)

            t_envs = time()

            assert observations.dtype == new_I.dtype # be careful this would create a silent casting otherwise.
            observations[:,t] = agent_state.observation_buffer.last() #last_obs_list # store last element in buffer
            rewards[:,t] = reward
            values[:,t] = predicted_values.squeeze(-1) if len(predicted_values.shape) == 2 else predicted_values
            logits[:,t] = l.detach().cpu().numpy()
            dones[:,t] = next_done_array
            actions[:,t] = chosen_actions

            # for the next step
            t_stack = time()

            dt_actions += [t_action - t0]
            dt_envs += [t_envs - t_action]
            dt_stacks += [t_stack - t_envs]

            # update agent state
            agent_state.observation_buffer.add(new_I) #
            agent_state.last_is_done = to_numpy(next_done_array)
            agent_state.model_state = to_numpy(new_model_state)

        tensors = (observations, rewards, values, logits, dones, actions,)

        # record new state
        self.agent_state = to_numpy(agent_state)

        self.last_data = to_numpy((tensors, init_agent_state))
        if profile:
            print(f"dt_actions={np.sum(dt_actions):0.04f}s env={np.sum(dt_envs):0.04f} stack={np.sum(dt_stacks):0.04f}")
        return tensors, init_agent_state


def td_errors(rewards, dones, values, gamma):
    next_values = values[:,1:]

    next_values = torch.where(dones[:,:-1], torch.zeros_like(next_values), next_values)

    values = values[:,:-1]
    rewards = rewards[:,:-1]
    td_errors = rewards + gamma * next_values - values
    return td_errors

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


class MyBatchNorm(nn.Module):

    def __init__(self, n, **kwargs):
        super().__init__()
        self.bn = nn.BatchNorm1d(n, **kwargs)

    def forward(self, x):
        if len(x.shape) == 2: return self.bn(x)
        if len(x.shape) == 3:
            return self.bn(x.transpose(1, 2)).transpose(1, 2)


def get_mlp(n_in, n_out, n_hidden=128, n_hidden_layers=1, norm="batch"):

    layers = []
    for i in range(n_hidden_layers):
        layers += [nn.Linear(n_in if i == 0 else n_hidden, n_hidden)]
        if norm == "batch": layers += [MyBatchNorm(n_hidden)]
        elif norm == "layer": layers += [nn.LayerNorm(n_hidden)]
        else: assert norm is None, f"unknown born request: {norm}"
        layers += [nn.ReLU()]

    layers += [nn.Linear(n_hidden, n_out)]

    return nn.Sequential(*layers)

class RNNModel(nn.Module):

    def __init__(self, input_size, output_size, hidden_size=64):
        super(RNNModel, self).__init__()
        self.inputs = MyBatchNorm(input_size)
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers=3, batch_first=True, bidirectional=False)
        self.out = get_mlp(hidden_size, output_size)

    def zero_state(self, batch_size):
        # WARNING: state need to be batch first for concatenating multiple environment runs
        h0 = torch.zeros([batch_size, self.rnn.num_layers, self.rnn.hidden_size])
        c0 = torch.zeros([batch_size, self.rnn.num_layers, self.rnn.hidden_size])
        return (h0,c0)

    def forward(self, x, state):



        if len(x.shape) == 2:
            x = x.unsqueeze(1)
            x = self.forward(x, state)
            return x.squeeze(1)

        x = self.inputs(x)
        state = deeppermute(state, [1, 0 ,2])
        x, final_state = self.rnn(x, state)
        final_state = deeppermute(final_state, [1, 0 ,2])
        x = self.out(x)
        return x, final_state

class MLPModel(nn.Module):

    def __init__(self, input_size, output_size, hidden_size=128):
        super(MLPModel, self).__init__()
        self.inputs = MyBatchNorm(input_size)
        self.mlp = get_mlp(input_size, output_size, hidden_size, 2, "batch")

    def zero_state(self, batch_size):
        return (None,)

    def forward(self, x, state):

        x = self.inputs(x)
        x = self.mlp(x)

        return x, state

class RLModel(nn.Module):

    def __init__(self, get_target_network_fn, observation_shape, n_observation_buffer, num_actions, gamma=0.99, lbda=0.95):
        super(RLModel, self).__init__()

        self.get_target_network_fn = get_target_network_fn

        self.observation_shape = observation_shape
        self.n_observation_buffer = n_observation_buffer
        self.num_actions = num_actions  # for simplicy

        input_size = math.prod(observation_shape) * n_observation_buffer + 1
        self.policy_model = RNNModel(input_size, self.num_actions)
        self.value_model = RNNModel(input_size, 1)

        self.register_buffer("_reward_var", torch.zeros((1,), dtype=torch.float32))
        self.register_buffer("_step_count", torch.zeros((1,), dtype=torch.int64))

        self.gamma = gamma
        self.lbda = lbda

    def model_file_name(self):
        return "model.pt"

    def normalize_rewards(self, rewards, epsilon=1e-8, momentum=0.999):
        if self._reward_var == 0:
            self._reward_var.data[0] = rewards.var()

        self._reward_var.data[0] = momentum * self._reward_var + (1 - momentum) * rewards.var()
        return rewards / torch.clip(torch.sqrt(self._reward_var), min=epsilon)

    def zero_state(self, batch_size):
        return (self.policy_model.zero_state(batch_size), self.value_model.zero_state(batch_size))

    def action_selection(self, obs_array, done_array, model_state):

        #logits, values, final_states = self.forward(obs_array, done_array, agent_state.model_state, True)
        logits, values, final_states = self.forward(obs_array, done_array, model_state, True)

        D = torch.distributions.categorical.Categorical(logits=logits)
        action = D.sample()
        return action, logits, values, final_states

    def losses(self, agent_states, observations, rewards, value_old, logits_old, dones, actions):
        #model_state = agent_states
        model_state = agent_states.model_state
        observations = agent_states.observation_buffer.make_buffer_dimension(observations)
        logits, values_trainable, final_states = self.forward(observations, dones, model_state, False)

        # log small formatting
        log_probs = torch.log_softmax(logits, -1)
        log_probs_old = torch.log_softmax(logits_old, -1)
        values_trainable = values_trainable.squeeze(2)  # * self.value_scale

        # normalize rewards
        rewards = self.normalize_rewards(rewards) # not needed because max is 1

        # real RL losses
        V = value_old #values_trainable.detach()
        Rs = future_returns(rewards, dones, V, gamma=self.gamma).detach()
        A = generalized_advantage(rewards, dones, V, gamma=self.gamma, lbda=self.lbda)

        action_oh = torch.nn.functional.one_hot(actions, self.num_actions).float()
        log_pis = torch.einsum("bti, bti-> bt", log_probs, action_oh)
        log_pis_old = torch.einsum("bti, bti-> bt", log_probs_old, action_oh)

        # policy_loss = - ((Rs - value_old).detach() * log_pis).mean()

        epsi = 0.2
        r = torch.exp(log_pis - log_pis_old.detach())[:, :-1]  # the last value is dropped in the advantage
        r_clipped = r.clip(min=1 - epsi, max=epsi)
        # a2c_loss = - (A * log_pis).mean()
        ppo_loss = - (torch.minimum(r * A, r_clipped * A)).mean()
        value_loss = (Rs.detach() - values_trainable).square().mean()

        entropy_loss = - log_pis.mean()

        losses = {
            "policy": ppo_loss,
            "value": value_loss,
            "entropy": entropy_loss,
        }

        return losses

    def save(self, ):
        torch.save(self.state_dict(), self.model_file_name())

    def load(self):
        file_name = self.model_file_name()
        dirs = list(os.listdir("./"))
        if file_name in dirs:
            file_path = f"./{file_name}"
            self.load_state_dict(torch.load(file_path))
            print(f"loaded network: {file_path} with {self._step_count.item()} training steps")
        else:
            print(f"Warnings: file {file_name} not found dir has: {dirs}")

    def forward(self, obs_array, done_array, states, use_target):

        if isinstance(obs_array, list): obs_array = np.array(obs_array, dtype=np.float32)
        if isinstance(done_array, list): done_array = np.array(done_array, dtype=np.float32)

        if len(done_array.shape) == 1:  # only buffer size is added, need to add batch size
            obs_array = obs_array[:, None]
            done_array = done_array[:, None]
            logits, value, final_states = self.forward(obs_array, done_array, states, use_target)
            return logits[:, 0], value[:, 0], final_states

        #assert len(obs_array.shape) == len(self.observation_shape) + 3, f"obs_array.shape={obs_array.shape} but observation is {self.observation_shape}"

        if use_target:
            with torch.no_grad():
                return self.get_target_network_fn().forward(obs_array, done_array, states, False)

        B, T, *_ = obs_array.shape
        obs_array = to_torch(obs_array).reshape(B,T, -1).float() #
        done_array = to_torch(done_array).reshape(B, T, 1).float()
        x = torch.cat([obs_array, done_array], 2)

        # outputs
        policy_state, value_state = states
        logits, policy_model_state = self.policy_model(x, policy_state)
        value, value_model_state = self.value_model(x, value_state)
        final_states = (policy_model_state, value_model_state)

        return logits, value, final_states