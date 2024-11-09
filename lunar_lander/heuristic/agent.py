import numpy as np


def heuristic_agent(state):
    """
    A simple heuristic agent for Lunar Lander v2.

    Args:
    state (np.array): The current state of the environment

    Returns:
    int: The action to take (0: do nothing, 1: fire left engine, 2: fire main engine, 3: fire right engine)
    """
    angle = state[4]
    angular_velocity = state[5]
    x_position = state[0]
    y_velocity = state[3]

    # actually small detail corrected from claude
    if y_velocity < -0.5 and np.random.rand() > 0.5:
        return 2

    if abs(angle) > 0.2:  # If tilted too much
        if angle > 0:
            return 3  # Fire left engine
        else:
            return 1  # Fire right engine
    elif abs(angular_velocity) > 0.2:  # If rotating too fast
        if angular_velocity > 0:
            return 3  # Fire left engine
        else:
            return 1  # Fire right engine
    elif abs(x_position) > 0.1:  # If too far from center
        if x_position > 0:
            return 3  # Fire left engine
        else:
            return 1  # Fire right engine
    elif y_velocity < -0.5:  # If falling too fast
        return 2  # Fire main engine
    else:
        return 0  # Do nothing


def random_agent(state):
    return np.random.randint(4)


class Agent:
    def __init__(self, env):
        pass

    def choose_action(self, observation, action_mask=None):
        # action, _ = self.model.predict(observation, deterministic=True)
        action = heuristic_agent(observation)
        return action