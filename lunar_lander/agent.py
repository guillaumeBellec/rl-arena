from copy import deepcopy

from rl_commons import RLModel, AgentState, ObservationBuffer
import numpy as np
from utils import to_torch, to_numpy, deepshape
from ema import EMA


class Agent:
    def __init__(self, env=None, player_name=None):

        self.model = RLModel(lambda : self.get_target_network(),
                             observation_shape=(8,),
                             n_observation_buffer=3,
                             num_actions=4,
                             reward_scale=10., # None:
                             ) # also initialize to zero_state
        self.model.load()
        self.model.eval()

    def get_target_network(self):
        if not hasattr(self, 'target_network'):
            self.copy_target_network()

        net = self.target_network[0]
        return  net

    def copy_target_network(self):
         # copy in any case.
        self.target_network  = [deepcopy(self.model)]
        self.get_target_network().eval()


    def zero_state(self, batch_size):
        return AgentState(
            step_count=np.zeros(batch_size, dtype=np.int64), # TODO: Put time with embedding instead?
            observation_buffer=ObservationBuffer.init(self.model.observation_shape,maxlen=self.model.n_observation_buffer, batch_size=batch_size, dtype=np.float32),
            extended_action_buffer=ObservationBuffer.init([], maxlen=self.model.n_observation_buffer, batch_size=batch_size, dtype=np.int64),
            model_state=self.model.zero_state(batch_size),
            accumulated_rewards=np.zeros(batch_size),
        )

    def preprocess_frame(self, observation):
        return np.array(observation, np.float32)

    def choose_action(self, observation, reward=0.0, terminated=False, truncated=False, info=None, action_mask=None):
        if not hasattr(self, '_state'):
            # init for the first step if it's run time.
            self._state = self.zero_state(1)

        self._state.observation_buffer.add(self.preprocess_frame(observation)[None, ...])

        action, _, _, model_state = self.model.action_selection(
            obs_array=self._state.observation_buffer.numpy_stack(), #self._state.last_obs_array,
            step_count=self._state.step_count,
            last_extended_actions=self._state.extended_action_buffer.numpy_stack(),
            model_state=to_torch(self._state.model_state),
        )

        self._state.extended_action_buffer.add(action.detach().cpu().numpy() + 1) # 0 means reset
        self._state.model_state = to_numpy(model_state)
        self._state.step_count += 1 #self._state.step_count + 1# never true again at run time

        return int(action.squeeze())