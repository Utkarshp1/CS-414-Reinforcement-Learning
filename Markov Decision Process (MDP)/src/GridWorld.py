import yaml
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

class GridWorld:
    '''
    '''
    def __init__(self):
        self._get_config()

        self.num_actions = 4
        self.num_states = self.config['grid_size'][0]*self.config['grid_size'][1]

        self.gamma = self.config['discount']
        self._get_transition_dynamics() 

    def _get_config(self):
        with open('config1.yaml') as config_file:
            self.config = yaml.safe_load(config_file)

    def _get_transition_dynamics(self):
        def generate_MDP(x, y, goal_coordinates, rewards, prob, wall_coor):
    self.trans_prob = np.zeros((
        self.num_states, 
        self.num_states, 
        self.num_actions
    ))
    self._calculate_up_prob(trans_prob, x, y, prob, wall_coor)
    self._calculate_down_prob(trans_prob, x, y, prob, wall_coor)
    self._calculate_left_prob(trans_prob, x, y, prob, wall_coor)
    self._calculate_right_prob(trans_prob, x, y, prob, wall_coor)
    
    wall_index = wall_coor[1]*x + wall_coor[0] 
    for i in range(self.num_actions):
        for j in range(self.num_states):
            trans_prob[wall_index, j, i] = 0
        for j in range(self.num_actions):
            trans_prob[j, wall_index, i] = 0
            
    self.reward = np.full((n, n, m), rewards['general'])
    pos_goal_state_index = (goal_coordinates['positive'][1]*x + 
                            goal_coordinates['positive'][0])
    neg_goal_state_index = (goal_coordinates['negative'][1]*x + 
                            goal_coordinates['negative'][0])
    self.reward[pos_goal_state_index, :, :] = rewards['positive']
    self.reward[neg_goal_state_index, :, :] = rewards['negative']

    # for i in range(m):
    #     is_goal_trans = trans_prob[:, pos_goal_state_index, i] != 0
    #     # print(is_goal_trans.shape)
    #     # print(is_goal_trans)
    #     reward[is_goal_trans, pos_goal_state_index, i] = rewards['positive']

    #     is_goal_trans = trans_prob[:, neg_goal_state_index, i] != 0
    #     reward[is_goal_trans, neg_goal_state_index, i] = rewards['negative']
    
    return trans_prob, reward