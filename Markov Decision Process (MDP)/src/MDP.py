import numpy as np

class MDP:
    def __init__(self, trans_prob, reward, gamma):
        self.trans_prob = trans_prob
        self.reward = reward
        self.gamma = gamma
        self.num_actions = trans_prob.shape[-1]
        self.num_states = trans_prob.shape[0]