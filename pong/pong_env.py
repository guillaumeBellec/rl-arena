import gymnasium as gym
import numpy as np
from psutil import IOPRIO_CLASS_BE
import time
import cv2

import gymnasium as gym
from pettingzoo.atari import pong_v3


def preprocess_frame(frames):
    """
        Preprocess a batch of frames
        Input shape: (batch_size, height, width, channels)
        Output shape: (batch_size, 1, 84, 84)
        """

    if len(frames.shape) == 3:
        return preprocess_frame(frames[None, ...])[0]

    frames = frames[:, ::2, ::2, :] # rapid down sampling

    if frames.shape[1:] == [84, 84,1]:
        return frames # nothing to change

    batch_size = frames.shape[0]
    processed = np.zeros((batch_size, 1, 84, 84), dtype=np.float32)

    for i in range(batch_size):
        # Convert to grayscale
        gray = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)

        # Resize to 84x84
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)

        # Normalize and add channel dimension
        processed[i, 0] = resized / 255.0

    return processed

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


class PongVersusAgent(gym.Env):

    def __init__(self, seed, **kwargs):

        self._env = pong_v3.env(render_mode="rgb_array")

        self.seed = seed

        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        self.metadata = self._env.metadata
        self.spec = self._env.spec
        self.reward_range = self._env.reward_range
        self.render_mode = self._env.render_mode

        # Instantiate your agent (assuming first agent is controlled by your agent)
        my_agent = Agent(env)

        # Interact with the environment
        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            cum_reward[agent] += reward
            if termination or truncation:
                action = None
            else:
                action_mask = observation["action_mask"]


class PongEnv(gym.Env):

    def __init__(self, num_steps, seed):
        super().__init__()
        self._env = gym.make("ALE/Pong-v5")
        self.seed = seed
        self.verbose = False
        #self.action_space.seed(seed)

        self.num_steps = num_steps
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        self.metadata = self._env.metadata
        self.spec = self._env.spec
        self.reward_range = self._env.reward_range
        self.render_mode = self._env.render_mode

        #self.just_had_reset = False
        self.reset(seed=self.seed)


    def reset(self, **kwargs):

        obs, info = self._env.reset(**kwargs)
        self.step_count = 0

        random_burn_in = 30
        n = (np.random.randint(random_burn_in) + time.time_ns() + self.seed * 7) % random_burn_in # use time_ns() but the number generator is synced on the threads
        if self.verbose:
            print(f"(seed = {self.seed}) reset burn out n={n}")
        for i in range(n):
            obs, reward, terminated, truncated, info = self._env.step(np.random.randint(self.action_space.n))

        self.just_had_reset = True
        return obs, info

    def close(self):
        return self._env.close()

    def step(self, action):
        self.just_had_reset = False
        total_rewards = 0
        self.step_count += 1

        for i in range(self.num_steps):
            obs, reward, terminated, truncated, info = self._env.step(action)
            total_rewards += reward
            done = terminated or truncated


            if done:
                #obs, info = self.reset()
                if self.verbose:
                    print(f"(seed = {self.seed}) done @ step: {self.step_count} total rewards {total_rewards}")

                break

        #if self.verbose and total_rewards != 0:
        #    print(f"(seed = {self.seed}) step: {self.step_count} total rewards {total_rewards}")

        return obs, total_rewards, terminated, truncated, info
