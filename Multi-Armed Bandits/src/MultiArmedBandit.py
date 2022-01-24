from numpy as np

class MultiArmedBandit:
    '''
        This class implements the Multi Armed Bandits i.e.
        a set of reward distribution (one per arm).        
    '''
    def __init__(self, num_arms, reward_dist):
        '''
            Arguments:
            ---------
                - num_arms (int): Number of arms in the Multi Armed
                    Bandit.
                - reward_dist (RewardDistribution class): The class of
                    reward distribution for the arms.
        '''
        self.num_arms = num_arms
        
        _init_reward_dist(reward_dist)
        
    def _init_reward_dist(reward_dist):
        '''
            This method initializes the reward distribution for each
            of the arm in the Multi Armed Bandit.
        '''
        self.arms = []
        
        for i in range(self.num_arms):
            self.arms.append(reward_dist())