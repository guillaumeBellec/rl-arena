from model import RLModel


class Agent:
    def __init__(self, env):

        self.model = RLModel(env) # also initialize to zero_state

        self.model.load()
        self.model.eval()

        self.states = self.model.zero_state(1)
        self.is_first_inference = True


    def choose_action(self, observation, action_mask=None):
        last_done = self.is_first_inference
        action, _, _, self.states = self.model.action_selection([observation], [last_done], self.states)
        self.is_first_inference = False
        return int(action.squeeze())