from model import PongModel
from copy import deepcopy
import numpy as np
from datetime import datetime

from pong.pong_env import preprocess_frame
from rl_commons import AgentState, ObservationBuffer
from utils import to_torch


class Agent:
    def __init__(self, env, player_name="first_0"):

        assert player_name in ["first_0", "second_0"], "unexpected player name " + player_name

        self.player_name = player_name
        self.name = "agent_ppo_" + datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

        self.model = PongModel(lambda : self.get_target_network(),
                               (1, 84, 84),
                               3,
                               6,
                               None
                               ) # also initialize to zero_state

        self.model.player_index = 0 if player_name == "first_0" else 1

        self.model.load()
        self.model.eval()
        #self.model.deterministic = True

        self.state = self.zero_state(1)

        # action repeat variables
        # self.n_action_repeat = 1
        # self.repeat_phase = -1 # next step is 0
        # self.last_action = None

    def preprocess_frame(self, frame):
        I = preprocess_frame(frame)
        if self.player_name == "second_0":
            I = I[..., ::-1] # reverse the width dimension
        return I

    def get_target_network(self):
        if not hasattr(self, 'target_network'):
            self.copy_target_network()
        return self.target_network[0]

    def copy_target_network(self):
        self.target_network  = [deepcopy(self.model)]
        self.target_network[0].eval()

    def zero_state(self, batch_size):
        return AgentState(
            step_count=np.zeros(batch_size, dtype=np.int64),
            #last_obs_array= np.stack([np.zeros(self.model.observation_shape) for _ in range(batch_size)]),
            observation_buffer=ObservationBuffer.init(self.model.observation_shape,maxlen=self.model.n_observation_buffer, batch_size=batch_size, dtype=np.float32),
            extended_action_buffer=ObservationBuffer.init([], maxlen=self.model.n_observation_buffer, batch_size=batch_size, dtype=np.int64),
            model_state=self.model.zero_state(batch_size),
            accumulated_rewards=np.zeros(batch_size),
        )

    def choose_action(self, observation, reward=0.0, terminated=False, truncated=False, info=None, action_mask=None):

        I = self.preprocess_frame(observation) # downscale image !
        self.state.observation_buffer.add(I[None, ...])

        action, _, _, final_model_state = self.model.action_selection(
            obs_array=self._state.observation_buffer.numpy_stack(),  # self._state.last_obs_array,
            step_count=self._state.step_count,
            last_extended_actions=self._state.extended_action_buffer.numpy_stack(),
            model_state=self._state.model_state,
        )

        self.state.extended_action_buffer.add(action.detach().cpu().numpy())
        self.state.model_state = final_model_state
        self.state.step_count += 1
        return int(action.squeeze().item())