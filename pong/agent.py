from model import RLModel
from copy import deepcopy
import numpy as np
from datetime import datetime

from pong.pong_env import preprocess_frame


class Agent:
    def __init__(self, env):

        self.name = "agent_ppo_" + datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

        self.model = RLModel(lambda : self.get_target_network()) # also initialize to zero_state

        self.model.load()
        self.model.eval()

        self.states = self.model.zero_state(1)
        self.observation_buffer = None
        self.is_first_inference = True

        # action repeat variables
        self.n_action_repeat = 4
        self.repeat_cycle = -1 # next step is 1
        self.last_action = None

    def get_target_network(self):
        return self.target_network[0]

    def copy_target_networks(self):
        assert self.model.training # should not be necessary in eval mode
        self.target_network = [deepcopy(self.model)]
        for p in self.target_network[0].parameters():
            p.grad = None
            p.requires_grad = False
        self.target_network[0].eval()

    def choose_action(self, observation, action_mask=None):
        self.repeat_cycle = (self.repeat_cycle+1) % self.n_action_repeat
        assert 0 <= self.repeat_cycle < self.n_action_repeat

        if self.repeat_cycle > 0:
            return self.last_action

        I = preprocess_frame(observation) # downscale image !
        if self.observation_buffer is None:
            assert self.is_first_inference # init if there is not previous history
            self.observation_buffer = np.stack([I for _ in range(self.model.n_observation_buffer)], 1)
        self.observation_buffer = self.model.roll_observation_buffer(I, self.observation_buffer)

        last_done = self.is_first_inference
        action, _, _, self.states = self.model.action_selection(self.observation_buffer, [last_done], self.states)
        self.is_first_inference = False
        self.last_action = int(action.squeeze())
        return self.last_action