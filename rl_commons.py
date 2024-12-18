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

from cloudpickle import instance
from mpmath.calculus.extrapolation import sumem
from numpy import dtype

from ema import EMA
from torch.distributions.categorical import Categorical


from torch.nested import nested_tensor

from utils import to_torch, deep_torch_stack, deepshape, to_numpy, deepdetach, deeptruncation, deep_torch_concatenate, \
    deeppermute, \
    EmaVal, to_scalar, deep_np_concatenate, RegressionAsClassificationLoss, save_args_to_json

import random


@dataclass
class ObservationBuffer:
    _deque : deque
    size : int
    shp : tuple

    @staticmethod
    def init(observation_shape, maxlen, batch_size, dtype):
        # this is not a constructor so that we can reuse the deepconcate etc... implemented for generic dataclass
        _deque = deque(maxlen=maxlen)
        size = maxlen
        shp = observation_shape
        if isinstance(shp, list): shp = tuple(shp)
        buffer = ObservationBuffer(_deque, size, shp)
        shp = [batch_size] + list(shp)
        zz = np.zeros(shp, dtype=dtype)
        buffer.add(zz) # fill with zz
        return buffer

    def reset_if_true(self, is_done):
        if not any(is_done): return

        assert len(self._deque) == self.size
        assert len(is_done.shape) == 1

        # unsqueeze until it's compatible
        while len(is_done.shape) < len(self._deque[0].shape):
            is_done = is_done[..., None]

        zz = np.zeros_like(self._deque[0])
        for i in range(self.size): # run over delays
            self._deque[i] = np.where(is_done, zz, self._deque[i]) #inplace assignment in numpy array. np.zeros_like(self._deque[i][b])

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
    extended_action_buffer: ObservationBuffer
    #last_obs_array : any
    last_is_done: any
    model_state: any
    accumulated_rewards : np.ndarray


class Simulation:

    def __init__(self, agent, envs, data_queue_length=20, gamma=0.99, lbda=0.93):
        self.envs = envs
        self.agent = agent
        self.gamma = gamma
        self.lbda = lbda

        with torch.inference_mode():
            n_envs = envs.num_envs if not isinstance(envs, list) else len(envs)
            self.agent_state = agent.zero_state(n_envs)
            new_I, _ = self.env_reset()
            new_I = agent.preprocess_frame(new_I)
            self.agent_state.observation_buffer.add(new_I)

        self.data_queue_train = deque(maxlen=data_queue_length)
        self.data_queue_test = deque(maxlen=data_queue_length)

    def empty_data_queues(self):
        # TODO: keep long replay buffer?

        # but clear on-policy buffers
        self.data_queue_test.clear()
        self.data_queue_train.clear()

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

    @torch.inference_mode()
    def run_many(self, num_steps : int, num_learner_steps : int):
        profile = False
        agent = self.agent
        envs = self.envs
        agent_state : AgentState = self.agent_state

        agent_state_list = [] #to_numpy(self.agent_state)] # store for returning this state, probably useless

        # could be different?
        n_envs = envs.num_envs if not isinstance(envs, list) else len(envs) # parallel env objects

        observations = np.zeros([n_envs, num_steps, agent.model.n_observation_buffer, *agent.model.observation_shape], dtype=np.float32)
        rewards = np.zeros([n_envs, num_steps], dtype=np.float32)
        values = np.zeros([n_envs, num_steps, agent.model.value_model_dim], dtype=np.float32)
        logits = np.zeros([n_envs, num_steps, agent.model.num_actions], dtype=np.float32)
        dones = np.zeros([n_envs, num_steps], dtype=bool)
        actions = np.zeros([n_envs, num_steps], dtype=np.int64)
        last_actions = np.zeros([n_envs, num_steps, agent.model.n_observation_buffer], dtype=np.int64)
        returns_at_done = np.zeros([n_envs, num_steps], dtype=np.float32)

        dt_actions = []
        dt_envs = []
        dt_stacks = []

        for t in range(num_steps):
            t0 = time()

            if t % num_learner_steps == 0:
                agent_state_list += [deepcopy(to_numpy(agent_state))]

            chosen_actions, l, predicted_values, new_model_state = agent.model.action_selection(
                agent_state.observation_buffer.numpy_stack(),
                agent_state.last_is_done,
                agent_state.extended_action_buffer.numpy_stack(),
                to_torch(agent_state.model_state),
            )

            chosen_actions = chosen_actions.detach().cpu().numpy()

            t_action = time()

            new_I, reward, terminated, truncated, info = self.env_step(chosen_actions) #, agent.n_action_repeat)
            new_I = agent.preprocess_frame(new_I) # possible downscaling
            is_done = np.logical_or(terminated, truncated)
            accumulated_rewards = agent_state.accumulated_rewards + reward # add current reward

            t_envs = time()


            assert observations.dtype == new_I.dtype # be careful this would create a silent casting otherwise.
            # TODO: Could add only last instead of entire buffer... not sure if it's worth it
            observations[:,t] = agent_state.observation_buffer.numpy_stack() #last_obs_list # store last element in buffer
            rewards[:,t] = reward
            values[:,t] = predicted_values #.squeeze(-1) if len(predicted_values.shape) == 2 else predicted_values
            logits[:,t] = l.detach().cpu().numpy()
            dones[:,t] = is_done
            actions[:,t] = chosen_actions
            last_actions[:, t] = agent_state.extended_action_buffer.numpy_stack()
            returns_at_done[:,t] = np.where(is_done, accumulated_rewards, np.nan * accumulated_rewards)

            # for the next step
            t_stack = time()

            dt_actions += [t_action - t0]
            dt_envs += [t_envs - t_action]
            dt_stacks += [t_stack - t_envs]

            # for next
            accumulated_rewards =  np.where(is_done, np.zeros_like(accumulated_rewards), accumulated_rewards)

            # update agent state
            agent_state.extended_action_buffer.add(chosen_actions + 1) # 0 means non-existing action (reset)
            agent_state.extended_action_buffer.reset_if_true(is_done)
            agent_state.observation_buffer.reset_if_true(is_done) # reset is before new state for obs
            agent_state.observation_buffer.add(new_I) #
            agent_state.last_is_done = to_numpy(is_done)
            agent_state.model_state = to_numpy(new_model_state)
            agent_state.accumulated_rewards = accumulated_rewards

        # td errors and generalized advantage
        raw_rewards = deepcopy(rewards)
        rewards = self.agent.model.normalize_rewards(rewards) # normalization

        RACL = RegressionAsClassificationLoss()
        value_val = RACL.value_from_logits(values) if agent.model.value_model_dim > 1 else values.squeeze(-1)
        returns = future_returns(rewards, dones, value_val, self.gamma)
        adv = generalized_advantage(rewards, dones, value_val, self.gamma, self.lbda)

        # these are all the data tensors:
        tensors = to_numpy( (observations, rewards, values, logits, dones, actions, returns, adv, last_actions, returns_at_done) )

        # record new state for later
        self.agent_state = to_numpy(agent_state)

        # those mini batches are returned for the learner
        tensors_list = []
        for i_start in range(0, num_steps, num_learner_steps):
            i_stop = i_start + num_learner_steps
            mini_tensors = deeptruncation(tensors, i_start, i_stop, dim=1)
            tensors_list += [mini_tensors]

        assert len(tensors_list) == len(agent_state_list) == num_steps // num_learner_steps
        new_data_list = [(t,a) for (t,a) in zip(tensors_list, agent_state_list)]

        if profile:
            print(f"dt_actions={np.sum(dt_actions):0.04f}s env={np.sum(dt_envs):0.04f} stack={np.sum(dt_stacks):0.04f}")
        return raw_rewards, returns_at_done, new_data_list

    def sample_frame_replay_buffer(self, train_or_test, n=1):
        if n==0:
            return []

        if train_or_test == "train":
            data_queue = self.data_queue_train
        else:
            data_queue = self.data_queue_test

        assert n <= data_queue.maxlen, f"requested {n} batch groups but queue has length {data_queue.maxlen}"
        old_data = list(data_queue)
        L = len(old_data)
        assert L > 0
        replay_buffer = []
        for d in old_data: replay_buffer += d # concatenate old data
        random.shuffle(replay_buffer)
        data_list = replay_buffer[:n] #+ new_data_list

        return data_list


def td_errors(rewards, dones, values, gamma):

    # TODO: This padding is a bit hacky, last value is used as if constant !
    last_value_pad = rewards[:,-1] + gamma * values[:,-1]
    next_values = np.concatenate([values[:,1:], last_value_pad[:,None]],  axis=1)
    next_values = np.where(dones, np.zeros_like(next_values), next_values)

    values = values
    rewards = rewards
    td_errors = rewards + gamma * next_values - values
    return td_errors

def future_returns(rewards, dones, values, gamma):
    B, T = rewards.shape

    if len(values.shape) == 3:
        values = values.squeeze(2)

    # Initialize returns array
    Rs = np.zeros_like(rewards)

    # Bootstrap from the last value unless the episode is done
    last_value = np.where(dones[:, -1], np.zeros_like(values[:, -1]), values[:, -1])
    Rs[:, -1] = rewards[:, -1] + gamma * last_value

    # Backwards value iteration
    for t in range(T - 2, -1, -1):  # Work backwards from T-2 to 0
        next_value = Rs[:, t + 1]
        # Zero out next value if current step leads to done
        next_value = np.where(dones[:, t], np.zeros_like(next_value), next_value)
        Rs[:, t] = rewards[:, t] + gamma * next_value

    return Rs

def generalized_advantage(rewards, dones, values, gamma, lbda):
    td_err = td_errors(rewards, dones, values, gamma)
    advantages = []
    gae = np.zeros_like(td_err[:, -1])  # Initialize with zeros for last timestep

    # Calculate GAE backwards
    for t in reversed(range(td_err.shape[1])):
        gae = np.where(dones[:,t], np.zeros_like(gae), gae)
        gae = td_err[:, t] + gamma * lbda * gae
        advantages.append(gae)

    # Reverse the accumulated advantages and stack them
    advantages = np.stack(advantages[::-1], axis=1)

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

class MyRNN(nn.Module):

    def __init__(self, dim, hidden_dim=64, num_layers=2, residual=True):
        super().__init__()
        self.residual = residual
        self.inputs = nn.LayerNorm(dim)
        self.rnn = nn.LSTM(dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=False)
        self.outputs = nn.Linear(hidden_dim, dim)

    def zero_state(self, batch_size):
        # WARNING: state need to be batch first for concatenating multiple environment runs
        h0 = torch.zeros([batch_size, self.rnn.num_layers, self.rnn.hidden_size])
        c0 = torch.zeros([batch_size, self.rnn.num_layers, self.rnn.hidden_size])
        return (h0,c0)

    def forward(self, x, state):

        if len(x.shape) == 2: # not time dimension
            x = x.unsqueeze(1)
            x, final_state = self.forward(x, state)
            return x.squeeze(1), final_state

        x = self.inputs(x)
        state = deeppermute(state, [1, 0 ,2]) # LSTM state should not be batch first
        x_lstm, final_state = self.rnn(x, state)
        final_state = deeppermute(final_state, [1, 0 ,2])
        x_out = self.outputs(x_lstm)

        if self.residual:
            x_out += x

        return x_out, final_state

def prepare_tensors(x, input_sizes):
    if not isinstance(input_sizes, int):
        assert isinstance(x, type(input_sizes)), f"got input_sizes={input_sizes}, but x shape={deepshape(x)}"
        B, T, *_ = x[0].shape
        x = [to_torch(t).view(B, T, -1) for t in x]
        x = torch.concat(x, -1)
    else:
        B, T, *_ = x[0].shape
        x = to_torch(x).view(B, T, -1)
    return x


class RNNModel(nn.Module):

    def __init__(self, input_sizes, output_size, hidden_size=128):
        super(RNNModel, self).__init__()
        self.input_sizes = input_sizes
        n_in = input_sizes if isinstance(input_sizes, int) else sum(input_sizes)

        self.input_mlp = get_mlp(n_in, hidden_size, norm="batch")
        self.rnn = MyRNN(hidden_size)
        self.output_mlp = get_mlp(hidden_size, output_size, norm="layer")

    def zero_state(self, batch_size):
        # WARNING: state need to be batch first for concatenating multiple environment runs
        return self.rnn.zero_state(batch_size)

    def forward(self, x, state):

        x = prepare_tensors(x, self.input_sizes)

        if hasattr(self, "input_mlp"):
            x = self.input_mlp(x)
        x, final_state = self.rnn(x, state)
        x = self.output_mlp(x)

        return x, final_state

class MLPModel(nn.Module):

    def __init__(self, input_sizes, output_size, n_hidden=256, n_hidden_layers=3, norm="batch"):
        super(MLPModel, self).__init__()
        self.input_sizes = input_sizes
        n_in = input_sizes if isinstance(input_sizes, int) else sum(input_sizes)
        self.mlp = get_mlp(n_in, output_size, n_hidden=n_hidden, n_hidden_layers=n_hidden_layers, norm=norm)

    def zero_state(self, batch_size):
        return (None,)

    def forward(self, x, state):
        x = prepare_tensors(x, self.input_sizes)
        x = self.mlp(x)
        return x, state

class RLModel(nn.Module):

    def __init__(self, get_target_network_fn, observation_shape, n_observation_buffer, num_actions, reward_scale):
        super(RLModel, self).__init__()

        self.get_target_network_fn = get_target_network_fn
        self.deterministic = False

        self.observation_shape = observation_shape
        self.n_observation_buffer = n_observation_buffer
        self.num_actions = num_actions  # for simplicy

        self.define_models()

        self.lock_reward_scale = reward_scale is not None
        s = 0 if reward_scale is None else reward_scale
        self.register_buffer("_reward_var", s * torch.ones((1,), dtype=torch.float32))
        self.register_buffer("_step_count", torch.zeros((1,), dtype=torch.int64))

    def define_models(self,):
        obs_size = math.prod(self.observation_shape) * self.n_observation_buffer

        self.embed_dim = 32
        self.value_model_dim = 1 #RegressionAsClassificationLoss().nbins if args.racl else 1
        self.is_done_embedding = nn.Embedding(1, self.embed_dim)
        self.action_embedding = nn.Embedding((self.num_actions+1) * self.n_observation_buffer, self.embed_dim)

        def make_model(n_ins, n_out):
            return MLPModel(n_ins, n_out) # RNNModel(n_ins, n_out)

        self.policy_model = make_model((obs_size, self.embed_dim), self.num_actions) #, 0.97)
        self.value_model = make_model((obs_size, self.embed_dim), self.value_model_dim)#, 0.97)
        self.model_list = [self.policy_model, self.value_model]

        return


    def model_file_name(self):
        return "model.pt"

    def normalize_rewards(self, rewards, epsilon=1e-8, momentum=0.999):
        if self.lock_reward_scale: return rewards / self._reward_var
        if self._reward_var == 0:
            self._reward_var.data[0] = float(rewards.var())

        self._reward_var.data[0] = momentum * self._reward_var + (1 - momentum) * rewards.var()
        return rewards / torch.clip(torch.sqrt(self._reward_var), min=epsilon)

    def zero_state(self, batch_size):
        def get_zero_state(m):
            if isinstance(m, EMA): return get_zero_state(m.model)
            return m.zero_state(batch_size)

        state = {
            "policy": get_zero_state(self.policy_model),
            "value": get_zero_state(self.value_model),
        }

        if hasattr(self, "Q_models"):
            state["Q"] = [get_zero_state(Q) for Q in self.Q_models]

        return state
        #return (self.policy_model.zero_state(batch_size), self.Q_model.model.zero_state(batch_size))

    @torch.inference_mode()
    def action_selection(self, obs_array, done_array, last_extended_actions, model_state):

        # un-squeeze dimension: expected no time dimension
        obs_array = obs_array[:,None]
        done_array = done_array[:,None]

        # logits, values, final_states = self.forward(obs_array, done_array, agent_state.model_state, True)

        net = self.get_target_network_fn()
        assert not net.training # target network is always in eval()
        logits, values, final_states = net.forward(obs_array, done_array, last_extended_actions, model_state)

        if not self.deterministic:
            action = Categorical(logits=logits).sample()
        else:
            action = logits.argmax(dim=-1)

        # squeeze time dimension
        action = action.squeeze(1)
        logits = logits.squeeze(1)
        values = values.squeeze(1)

        return action, logits, values, final_states

    def losses_ppo(self, mini_batch_tensors, agent_states, args):
        #model_state = agent_states
        model_state = agent_states.model_state
        tnet = self.get_target_network_fn()

        (observations, rewards, value_old, logits_old, dones, actions, Rs, A, last_actions, _) = mini_batch_tensors

        logits, values_trainable, final_state = self.forward(observations, dones, last_actions, model_state)
        #logits, final_policy_state = self.policy_model((observations, dones), model_state["policy"])
        #values_trainable, final_value_state = self.value_model((observations, dones), model_state["value"])

        # log small formatting
        log_probs = torch.log_softmax(logits, -1)
        log_probs_old = torch.log_softmax(logits_old, -1)
        #values_trainable = values_trainable.squeeze(2)  # * self.value_scale

        action_oh = torch.nn.functional.one_hot(actions, self.num_actions).float()
        log_pis = torch.einsum("bti, bti-> bt", log_probs, action_oh)
        log_pis_old = torch.einsum("bti, bti-> bt", log_probs_old, action_oh)

        # A2C
        # policy_loss = - ((Rs - value_old).detach() * log_pis).mean()
        # a2c_loss = - (A * log_pis).mean()

        # PPO
        epsi = 0.2
        r = torch.exp(log_pis - log_pis_old.detach())  # the last value is dropped in the advantage
        r_clipped = r.clip(min=1-epsi, max=1+epsi)
        ppo_loss = - (torch.minimum(r * A, r_clipped * A)).mean()
        neg_entropy = log_pis.mean()

        value_target = Rs.detach()
        if self.value_model_dim > 1:
            RACL = RegressionAsClassificationLoss()
            value_loss = RACL(values_trainable, value_target)
        else:
            value_loss = (value_target - values_trainable.squeeze(2)).square().mean()

        losses = {
            "policy": ppo_loss, #  + ppo_loss
            "value": value_loss, #value_loss + q_loss,
            "entropy": args.alpha * neg_entropy,
        }

        return losses

    def losses_sac(self, mini_batch_tensors, agent_states):
        raise NotImplementedError()

        # model_state = agent_states
        model_state = agent_states.model_state
        tnet = self.get_target_network_fn()

        (observations, rewards, value_old, logits_old, dones, actions, Rs, A) = mini_batch_tensors

        logits, final_policy_state = self.policy_model((observations, dones), model_state["policy"])
        #logits, values_trainable, final_states = self.forward(observations, dones, model_state, False)

        # log small formatting
        log_probs = torch.log_softmax(logits, -1)
        log_probs_old = torch.log_softmax(logits_old, -1)

        action_oh = torch.nn.functional.one_hot(actions, self.num_actions).float()

        #log_pis = torch.einsum("bti, bti-> bt", log_probs, action_oh)
        #log_pis_old = torch.einsum("bti, bti-> bt", log_probs_old, action_oh)

        # action with differentiable gradients
        probs = torch.exp(log_probs)
        soft_action_oh = probs + (action_oh - probs).detach()
        entropy = (log_probs * probs).sum(-1) #torch.einsum("bti,bti->bt", log_probs, soft_action_oh)

        # Value function
        values, _ = self.value_model.model((observations, dones), model_state["value"])

        # compute Q function targets
        with torch.inference_mode():
            values_shadow, _ = tnet.value_model.shadow((observations, dones), model_state["value"])
            Q_target = rewards[:,:-1,None] + self.gamma * values_shadow[:,1:].detach() #+ self.alpha * next_entropy)

        # q loss
        Q_value_list = []
        Q_value_shadow_list = []

        Q_losses = []
        for Q_model, tQ_model, Q_state in zip(self.Q_models, tnet.Q_models, model_state["Q"]):
            Q_value, _ = Q_model.model((observations, dones, action_oh), Q_state) # no action here
            Q_value_shadow, _ = tQ_model.shadow((observations, dones, soft_action_oh), Q_state) # action is diff

            Q_value_list += [Q_value]
            Q_value_shadow_list += [Q_value_shadow]

            assert Q_value.shape[1] > 1
            Q_target = rewards[:,:-1,None] + self.gamma * (Q_value_shadow[:,1:] + self.alpha * entropy[:,1:, None])
            Q_losses += [(Q_value[:,:-1] - Q_target.detach()).square().mean()] #
        Q_loss = sum(Q_losses)

        #Q_value_min = torch.min(*Q_value_shadow_list)
        #V_target = Q_value_min - self.alpha * log_pis[:,:,None]
        #value_loss = (values - V_target.detach()).square().mean()

        Q_value_shadow_mean = sum(Q_value_shadow_list) / len(Q_value_shadow_list)

        policy_loss = self.alpha * entropy[:,:,None] - Q_value_shadow_mean
        #policy_loss_ = self.alpha * log_pis_[:,:,None] - Q_value_shadow_mean_
        policy_loss = policy_loss.mean() #- policy_loss_.mean()

        losses = {
            "policy": policy_loss, # + self.alpha * neg_entropy, #  + ppo_loss
            "value": Q_loss, #value_loss + q_loss,
        }

        return losses

    def save(self,):
        # TODO: ensembles ?
        torch.save(self.state_dict(), self.model_file_name())

    def load(self):
        file_name = self.model_file_name()
        dirs = list(os.listdir("."))
        if file_name in dirs:
            file_path = f"./{file_name}"
            self.load_state_dict(torch.load(file_path))
            print(f"loaded network: {file_path} with {self._step_count.item()} training steps")
        else:
            print(f"Warnings: file {file_name} not found dir has: {dirs}")

    def context_embedding(self, done_array, last_extended_action):
        B, T = done_array.shape
        done_array = to_torch(done_array).reshape(B, T,).int()  # make time dim
        last_extended_action = to_torch(last_extended_action).reshape(B, T, self.n_observation_buffer).int()  # make time dim
        n_extended_actions = self.num_actions + 1 # start is not an action
        assert last_extended_action.min() >= 0
        assert last_extended_action.max() < n_extended_actions

        # TODO: vectorize this for loop
        context = self.is_done_embedding.weight * done_array[..., None]

        for i in range(self.n_observation_buffer):
            context += self.action_embedding(last_extended_action[:, :, i] + i * n_extended_actions)

        return context

    def forward(self, obs_array, done_array, last_extended_actions, model_state):
        context = self.context_embedding(done_array, last_extended_actions)
        x = (obs_array, context)
        logits, final_policy_state = self.policy_model(x, model_state["policy"])
        values, final_value_state = self.value_model(x, model_state["value"])

        final_states = {
            "policy": final_policy_state,
            "value": final_value_state,
        }

        # update Q values if necessary
        if not list(model_state.keys()) == list(final_states.keys()):
            raise NotImplementedError()

        return logits, values, final_states


def train(optimizer, envs, agent, writer, args):

    agent.model.train()
    agent.model.deterministic = False # not quite the same thing as eval

    n_prints = int(np.ceil(args.n_print // args.n_target_updates) * args.n_target_updates)
    n_target_updates = args.n_target_updates

    n_envs = envs.num_envs if not isinstance(envs, list) else len(envs)

    best_returns = args.returns_saving_thr
    returns_ema, rewards_ema, dt_run_ema, dt_step_ema = [EmaVal() for _ in range(4)]

    #buffer_length = n_target_updates #if args.algo == "ppo" else args.replay_buffer_steps // args.n_actor_steps
    simu = Simulation(envs=envs, agent=agent,
                      data_queue_length=n_target_updates,
                      gamma=args.gamma,
                      lbda=args.lbda) #for envs in env_groups]

    k_step=0
    for k_loop in  range(args.n_training_steps):
        # real stop condition is number of gradient steps:
        if k_step > args.n_training_steps: break

        if k_loop % n_target_updates == 0:
            # update target networks
            agent.copy_target_network()
            target_net_reward_list = []
            target_net_returns_list = []
            #simu.empty_on_policy_buffer() # make sure that buffer has only data from on-policy

        # ACTOR Part
        t0 = time()
        # TODO: reset RNN states?

        with torch.inference_mode():
            raw_rewards, returns_at_done, new_mini_batch_list = simu.run_many(args.n_actor_steps, args.n_learner_steps)
            simu.data_queue_train.append(new_mini_batch_list)

        target_net_reward_list.append(raw_rewards)
        target_net_returns_list.append(returns_at_done)
        if k_loop >= n_prints-1: dt_run_ema(time() - t0)
        rewards_ema(float(np.mean(raw_rewards)))
        returns_ema(float(np.nanmean(returns_at_done)))

        # LOGGING and SAVING
        if k_loop % n_prints == n_prints -1:
            assert len(target_net_reward_list) == n_target_updates, f"got reward-list of size: {len(target_net_reward_list)} reward data n_target_updates={n_target_updates} k_loop={k_loop}"
            reward_per_step = float(np.mean(target_net_reward_list))
            returns_at_done = float(np.nanmean(target_net_returns_list))
            writer.add_scalar("reward per step", reward_per_step, agent.model._step_count)
            if not np.isnan(returns_at_done):
                writer.add_scalar("returns", returns_at_done, agent.model._step_count)
            for k, v in losses.items(): writer.add_scalar(k, to_scalar(v), agent.model._step_count)

            assert n_prints % n_target_updates == 0, "expected multiple of target network updates, then rewards should be"
            if returns_at_done > best_returns: # best condition (based on full return ideally)
                    best_rewards_per_step = reward_per_step
                    best_returns = returns_at_done
                    if k_step > 10_000:
                        agent.get_target_network().save() # save target network with best perf
                        print("saved. ", end="")
            loss_string = " ".join([f" {key} {loss.item():0.03f} " for key, loss in losses.items()])
            print(f"rewards @ {k_step} : target {reward_per_step:0.03f} / {returns_at_done:0.02f} running mean {rewards_ema.read():0.03f} / {returns_ema.read():0.02f} \t loss {loss_string} \t step {dt_step_ema.read():0.02f}s run {dt_run_ema.read():0.02f}s")

        # LEARNER
        assert agent.model.training
        assert args.learner_batch_size % n_envs == 0
        K = args.learner_batch_size // n_envs

        def get_mini_batch(mini_batch_list, i_batch, K):
            # utils to concatenate actor data into learner data
            batches = mini_batch_list[i_batch:(i_batch + K)]
            assert len(batches) > 0
            (mini_batch_tensors, mini_batch_agent_state) = deep_np_concatenate(batches, 0)
            mini_batch_tensors = to_torch(mini_batch_tensors) #TODO add device?
            mini_batch_agent_state = to_torch(mini_batch_agent_state)
            return mini_batch_tensors, mini_batch_agent_state

        t0 = time()


        agent.model.train()

        buffer_data = simu.sample_frame_replay_buffer("train", n=args.resample_factor)
        buffer_data = buffer_data + new_mini_batch_list
        random.shuffle(buffer_data)

        # training: first take the current data, but then resample from samples with same on-policy target network
        for i_batch in range(0, len(buffer_data), K):

                # loading data:
                mini_batch_tensors, mini_batch_agent_state = get_mini_batch(buffer_data, i_batch, K)
                optimizer.zero_grad()

                losses = agent.model.losses_ppo(mini_batch_tensors, mini_batch_agent_state, args)

                # most updates are value only
                loss = sum(list(losses.values())) # add all.
                loss.backward()
                optimizer.step()
                # update EMA
                for m in agent.model.model_list:
                    if isinstance(m, EMA): m.update()
                agent.model._step_count.data += 1 # step count
                k_step += 1
        if k_loop >= n_prints-2: dt_step_ema(time() - t0)
