import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"]="python"

import torch
import torch.nn as nn
import numpy as np
from time import time
import torch.nn.functional as F

from pong_env import PongEnv, preprocess_frame

from utils import EmaVal, deepclone, ReplayBuffer, Simulation, to_scalar
from torch.utils.tensorboard import SummaryWriter

def train(optimizer, envs_groups, agent, writer : SummaryWriter, num_simu_steps: int, num_training_steps: int, run_stack_size=2, n_target_updates=100, n_prints=100):

    #run_stack_size = seq_len // num_simu_steps # 60 will be the number of RNN steps
    #if n_target_updates is None: n_target_updates = len(env_groups) * run_stack_size
    n_prints = min(n_prints, num_training_steps)
    n_target_updates = min(n_target_updates, n_prints)

    agent.model.train()
    n_envs = envs_groups[0].num_envs

    best_rewards_per_step = 0.0 # threshold, start saving
    rewards_ema = EmaVal()
    losses_ema = {} #[EmaVal() for _ in range(3)]

    # train off between training on the same batch and off-policy problems
    simulations = []
    for i_envs, envs in enumerate(envs_groups):

        # init the buffer
        I, _ = envs.reset(seed=[i_envs * n_envs + i for i in range(n_envs)])
        I = preprocess_frame(I)
        last_obs_array = np.stack([I for _ in range(agent.model.n_observation_buffer)], 1)
        last_done_array = np.ones(n_envs, dtype=bool)

        run_stack = ReplayBuffer(run_stack_size) #
        simu = Simulation(
            envs=envs,
            agent_states=agent.model.zero_state(n_envs),
            last_observations=last_obs_array,
            last_dones=last_done_array,
            replay_buffer=run_stack
        )

        simulations.append(simu)
    simulations = tuple(simulations)

    t_init = time()
    dt_step = 0.0
    losses = {}

    for i in range(num_training_steps):

        # ACTOR PART
        simu_index = i % len(simulations)
        simu : Simulation = simulations[simu_index]

        if i % n_target_updates == 0:
            # update target networks, for pong since we recycle recent runs, this is somewhat redundant.
            agent.copy_target_networks()

        t0 = time()
        with torch.inference_mode():
                agent_states = simu.agent_states
                if np.random.randint(10) == 0:
                    agent_states = agent.model.zero_state(n_envs)  # sometimes reset the agent RNN states randomly

                new_run, last_obs_array, last_done_array, new_agent_states = \
                    run_many(simu.envs, agent, num_simu_steps, simu.last_observations, simu.last_dones, agent_states)

                # update, TODO: could be done in run_many?
                simu.replay_buffer.add(new_run)
                simu.last_observations = last_obs_array
                simu.last_dones = last_done_array
                simu.agent_states = new_agent_states

        dt_run = time() - t0

        # REPORTING and SAVING
        # compute some EMA
        rewards = new_run[2]  # rewards is from target network but now they are the same.
        reward_per_step = float(rewards.mean().item()) / agent.n_action_repeat
        rewards_ema(reward_per_step)
        for k in losses.keys():
            if not k in losses_ema.keys(): losses_ema[k] = EmaVal()
            losses_ema[k](losses[k])
        assert n_prints % n_target_updates == 0, f"expected multiple of target network updates, then rewards should be got n_prints={n_prints} and n_target={n_target_updates}"

        if i % n_prints == 0:
            writer.add_scalar("reward per step", reward_per_step, i)
            for k, v in losses.items(): writer.add_scalar(k, to_scalar(v), i)

            loss_string = " ".join([f" {k} {ema.read():0.03f} " for k,ema in losses_ema.items()])
            print(f"rewards @ {i}: \t {reward_per_step:0.03f} running mean {rewards_ema.read():0.03f} \t|\t loss {loss_string} \t|\t step {dt_step:0.02f}s run {dt_run:0.02f}s ")
            if reward_per_step > best_rewards_per_step:
                best_rewards_per_step = reward_per_step
                agent.model.save()
                print("saved. ")

        t0 = time()

        # LEARNER PART
        #tensors, agent_states = RunStack.merge_tensor_stacks(simulations, simu_index, 2)
        tensors = agent.model.to_torch(new_run)
        observations, rewards, values, logits, dones, actions = tensors

        optimizer.zero_grad()
        losses = agent.model.losses(observations, rewards, values, logits, dones, actions, deepclone(agent_states))
        loss = losses['policy'] + losses['value'] + 1e-2 * losses['entropy']

        # perform optimizer step (after save to ensure that we save when target_network == network.eval()
        loss.backward()
        optimizer.step()

        dt_step = time() - t0

    print(f"Training finished {num_training_steps} steps in {(time() - t_init) / 3600} hours.")

def td_errors(rewards, dones, values, gamma):

    next_values = values[:,1:]
    values = values[:,:-1]
    rewards = rewards[:,:-1]
    dones = dones[:, :-1]

    next_values = torch.where(dones, torch.zeros_like(next_values), next_values)
    td_errors = rewards + gamma * next_values - values
    return td_errors

def generalized_advantage(rewards, dones, values, gamma, lbda):
    td_err = td_errors(rewards, dones, values, gamma)
    T = td_err.shape[1] # equivalent to T-1 if T is rewards.shape[1]
    dones = dones[:, :-1] # be careful, we changed dones size
    advantages = []

    gae = torch.zeros_like(td_err[:, -1])  # Initialize with zeros for last timestep

    # Calculate GAE backwards, T here is equivalent to T-1 in the rewards tensor
    for t in range(T-1, -1, -1):
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
    #last_value = torch.where(dones[:, -1], torch.zeros_like(values[:, -1]), values[:, -1])
    #Rs[:, -1] = rewards[:, -1] + gamma * last_value

    last_value = torch.where(dones[:, -1], rewards[:, -1], values[:, -1])
    Rs[:, -1] = last_value

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


class PongCNN(nn.Module):
    def __init__(self, n_actions=6, n_channels=3):  # Pong has 6 possible actions
        super(PongCNN, self).__init__()

        # Convolutional layers with BatchNorm
        self.conv1 = nn.Conv2d(n_channels, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)

        # Layer norm for the fully connected layer
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.ln1 = nn.LayerNorm(512)

        self.fc2 = nn.Linear(512, 128)
        self.ln2 = nn.LayerNorm(128)
        self.fc_out = nn.Linear(128, n_actions)

    def forward(self, x):
        # Convolutional layers with BatchNorm and ReLU
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers with LayerNorm
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        x = self.fc_out(x)

        return x

class RLModel(nn.Module):

    def __init__(self, get_target_network_fn, gamma=0.98, lbda=0.93):
        super(RLModel, self).__init__()

        self.get_target_network_fn = get_target_network_fn

        self.downscaled_shape = [1, 84, 84]
        #self.downscaled_shape = [52, 40 , 3]
        self.n_observation_buffer = 3
        self.num_actions = 6 # for simplicy
        #self.cnn_dim = 384 #self.cnn(dummy_input).numel()
        #self.hidden_dim = 128

        self.policy_model = PongCNN( self.num_actions, self.n_observation_buffer)
        self.value_model = PongCNN( 1, self.n_observation_buffer)

        #self.cnn = make_vgg_layers([8, "M", 16, "M", 32, "M", 64, "M"], in_channels=3 * self.n_observation_buffer, batch_norm=True)
        #self.fc = nn.Sequential(nn.Linear(self.cnn_dim+1, self.hidden_dim), MyBatchNorm(self.hidden_dim))
        #self.rnn = RNNModel(self.hidden_dim)
        
        #self.policy_head = get_mlp(self.hidden_dim, self.num_actions, n_hidden_layers=2, norm="layer")
        #self.value_head = get_mlp(self.hidden_dim, 1, n_hidden_layers=2, norm="layer")

        self.register_buffer("_reward_var", torch.zeros((1,), dtype=torch.float32))
        self.register_buffer("_step_count", torch.zeros((1,), dtype=torch.int64))

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
        return None # self.rnn.zero_state(batch_size)

    def action_selection(self, obs_array, done_array, states):
        logits, values, final_states = self.forward(obs_array, done_array, states, True)

        D = torch.distributions.categorical.Categorical(logits=logits)
        action = D.sample()
        return action, logits, values, final_states

    def roll_observation_buffer(self, I, I_buffer):
        assert len(I.shape) == len(I_buffer.shape) - 1

        if len(I_buffer.shape) == 4:  # d, W, H, C
            I = I[None, ...]
            I_buffer = I_buffer[None, ...]
            return self.roll_observation_buffer(I, I_buffer)[0]

        # [I(t), I(t-1), ... I(t-d)]
        return np.concatenate([I[:, None], I_buffer[:, :-1]], axis=1)

    def make_buffer_dimension(self, observations):
        B, T, *_ = observations.shape
        d = self.n_observation_buffer
        assert T > d
        pad = observations[:,0:1] # first image
        buffer = []
        for i in range(d):
            # [I(t), I(t-1), ... I(t-d)]
            o_delayed = torch.cat([pad.repeat([1, i, 1, 1, 1]), observations[:,:T-i]], 1)
            buffer += [o_delayed]

        # buffer.reverse() # in-place reverse list order, this is needed depending how the roll_observation_buffer is implemented
        return torch.stack(buffer, 2) # dimension 2 is for buffer

    def losses(self, observations, rewards, value_old, logits_old, dones, actions, states=None):

        observations_buffer = self.make_buffer_dimension(observations)
        logits, values_trainable, final_states = self.forward(observations_buffer, dones, states, False)

        # log small formatting
        log_probs = torch.log_softmax(logits, -1)
        log_probs_old = torch.log_softmax(logits_old, -1)
        values_trainable = values_trainable.squeeze(2) #* self.value_scale

        # normalize rewards
        # rewards = self.normalize_rewards(rewards) # not needed because max is 1

        # real RL losses
        V = values_trainable.detach()
        Rs = future_returns(rewards, dones, V, gamma=self.gamma).detach()
        A = generalized_advantage(rewards, dones, V, gamma=self.gamma, lbda=self.lbda)

        action_oh = torch.nn.functional.one_hot(actions, self.num_actions).float()
        log_pis = torch.einsum("bti, bti-> bt", log_probs, action_oh)
        log_pis_old = torch.einsum("bti, bti-> bt", log_probs_old, action_oh)

        #policy_loss = - ((Rs - value_old).detach() * log_pis).mean()

        epsi = 0.2
        r = torch.exp(log_pis - log_pis_old.detach())[:,:-1] # the last value is dropped in the advantage
        r_clipped = r.clip(min=1-epsi, max=epsi)
        #a2c_loss = - (A * log_pis).mean()
        ppo_loss = - (torch.minimum(r * A ,r_clipped * A)).mean()
        value_loss = (Rs.detach() - values_trainable).square().mean()

        entropy_loss = - log_pis.mean()

        losses = {
            "policy": ppo_loss,
            "value": value_loss,
            "entropy": entropy_loss,
        }

        return losses

    def save(self,):
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

    def to_torch(self, tensor):

        if isinstance(tensor, list):
            return [self.to_torch(t) for t in tensor]

        if isinstance(tensor, tuple):
            return tuple([self.to_torch(t) for t in tensor])

        if isinstance(tensor, dict):
            return dict([(k,self.to_torch(v)) for k,v in tensor.items()])

        if isinstance(tensor, np.ndarray):
            tensor = torch.from_numpy(tensor)

        if tensor.dtype == torch.uint8:
            tensor = tensor.float() / 255.0 # max is one.

        return tensor

    def forward(self, obs_array, done_array, states, use_target):
        obs_array = self.to_torch(obs_array)
        done_array = self.to_torch(done_array)

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

        B, T, d, C, W, H = obs_array.shape
        assert W == H == 84
        assert d == self.n_observation_buffer
        assert C == 1
        #o = torch.cat([obs_array[:,:,i] for i in range(d)], 3) # buffer buffer on channel dimension
        x = obs_array.view(B * T, C * d, W, H) # flatten time dimension
       # x = x.permute([0,3,1,2]) #  B * T, C * d, W, H

        # CNN
        #o = self.cnn(o.permute([0,3,1,2])).reshape([B, T, self.cnn_dim])

        # Merge visual input and done flags
        #d = done_array.float().unsqueeze(-1)
        #x = torch.cat([o, d], -1)
        #x = self.fc(x) #

        # RNN
        #x, final_states = self.rnn(x, states) #

        # outputs
        logits = self.policy_model(x).reshape([B, T, self.num_actions])
        value = self.value_model(x).reshape([B, T, 1])
        final_states = None # no rnn states

        return logits, value, final_states



if __name__ == "__main__":
    import gymnasium as gym
    from agent import Agent

    import ale_py # only needed for single player training
    gym.register_envs(ale_py)

    # parallelization parameters
    numb_parallel_agents = 8 # x2 this is model batch-size (keep it low if cpu only)
    env_groups = 1 # separate group of environments to alternative and avoid overfitting one history, too high will be off-policy
    num_steps = 10 # num steps so simulate in one group between each gradient descent step

    num_training_steps = 100_000

    agent =  Agent(None)

    # This option goes with env.step in run many, faster on my thinkpad laptop that the default parallelized atari
    env_groups = [gym.vector.AsyncVectorEnv([lambda : PongEnv(num_steps=agent.n_action_repeat, seed=i + numb_parallel_agents * i_g) for i in range(numb_parallel_agents)]) for i_g in range(env_groups)]

    # This option goes with async_multi_step
    #envs = gym.make_vec("ALE/Pong-v5", num_envs=batch_size, vectorization_mode="async")

    optimizer = torch.optim.AdamW(agent.model.parameters(), lr=1e-3)
    writer = SummaryWriter(f"runs/PongAgent_{agent.name}")
    train(optimizer, env_groups, agent, writer, num_steps, num_training_steps=num_training_steps, run_stack_size=1, n_prints=100)

    agent.model.save()
    print(f"saved agent for {num_training_steps}")