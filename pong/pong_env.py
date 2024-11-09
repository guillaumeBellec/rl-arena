import gymnasium as gym
import numpy as np
from psutil import IOPRIO_CLASS_BE


def pre_process(image):
    if len(image.shape) == 3:
        return pre_process(image[None, ...])[0]

    Is = [
        image[:, :208:4, ::4, :],
        image[:, 2:208:4, ::4, :],
        image[:, :208:4, 2::4, :],
        image[:, 2:208:4, 2::4,:]
        ]
    # fast cpu downscaling
    I = np.maximum(np.maximum(Is[0], Is[1]), np.maximum(Is[2], Is[3]))
    return I.astype(np.uint8)

def roll_observation_buffer(I, I_buffer):
    assert len(I.shape) == len(I_buffer.shape)-1

    if isinstance(I_buffer, list):
        I_buffer = np.concatenate(I_buffer, axis=1)

    if len(I_buffer.shape) == 4: # d, W, H, C
        I = I[None, ...]
        I_buffer = I_buffer[None, ...]
        return roll_observation_buffer(I, I_buffer)[0]

    return np.concatenate([I_buffer[:,:-1], I[:, None]], axis=1)


class PongEnv(gym.Env):

    def __init__(self, num_steps):
        super().__init__()
        self._env = gym.make("ALE/Pong-v5", disable_env_checker=True)
        self.num_steps = num_steps
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        self.metadata = self._env.metadata
        self.spec = self._env.spec
        self.reward_range = self._env.reward_range
        self.render_mode = self._env.render_mode

    def reset(self, **kwargs):
        return self._env.reset( **kwargs)

    def close(self):
        return self._env.close()

    def step(self, action):

        total_rewards = 0
        for i in range(self.num_steps):
            obs, reward, terminated, truncated, info = self._env.step(action)
            total_rewards += reward
            done = terminated or truncated

            if done:
                obs, info = self._env.reset()

        return obs, total_rewards, terminated, truncated, info
