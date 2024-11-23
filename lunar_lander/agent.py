from copy import deepcopy

from click.core import batch

from lunar_lander.rl_commons import AgentState, ObservationBuffer
from rl_commons import RLModel
import numpy as np
from utils import to_torch, to_numpy, deepshape


class Agent:
    def __init__(self, env):

        self.model = RLModel(lambda : self.get_target_network(),
                             observation_shape=(8,),
                             n_observation_buffer=3,
                             num_actions=4) # also initialize to zero_state

        self.model.load()
        self.model.eval()

    def get_target_network(self):
        if not hasattr(self, 'target_network'):
            self.copy_target_network()
        return self.target_network[0]

    def copy_target_network(self):
        self.target_network  = [deepcopy(self.model)]
        self.target_network[0].eval()

    def zero_state(self, batch_size):
        return AgentState(
            last_is_done=np.array([True for _ in range(batch_size)]),
            #last_obs_array= np.stack([np.zeros(self.model.observation_shape) for _ in range(batch_size)]),
            observation_buffer=ObservationBuffer.init(self.model.observation_shape,maxlen=self.model.n_observation_buffer),
            model_state=self.model.zero_state(batch_size),
        )

    def preprocess_frame(self, observation):
        return observation

    def choose_action(self, observation, action_mask=None):
        if not hasattr(self, '_state'):
            # init for the first step if it's run time.
            self._state = self.zero_state(1)

        self._state.observation_buffer.add(self.preprocess_frame(observation)[None, ...])
        action, _, _, model_state = self.model.action_selection(
            obs_array=self._state.observation_buffer.numpy_stack(), #self._state.last_obs_array,
            done_array=self._state.last_is_done,
            model_state=to_torch(self._state.model_state),
        )

        self._state.model_state = to_numpy(model_state)
        self._state.last_is_done = np.array([False], dtype=bool) # never true again at run time


        return int(action.squeeze())