import numpy as np

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
        
        self._init_reward_dist(reward_dist)
        
    def _init_reward_dist(self, reward_dist):
        '''
            This method initializes the reward distribution for each
            of the arm in the Multi Armed Bandit.
        '''
        self.arms = []
        self.optimal_value = 0
        self.optimal_arm = None
        
        for i in range(self.num_arms):
            self.arms.append(reward_dist())
            
            if self.arms[i].mean > self.optimal_value:
                self.optimal_value = self.arms[i].mean
                self.optimal_arm = i
            
    def calculate_regret(self, reward, arm_index):
        '''
            This method calculates the regret given the reward and the
            arm.
            
            Arguments:
            ---------
                - reward (float): The reward that we got at timestamp t.
                - arm_index (int): Index for the arm which was pulled at
                    timestamp t (An int in the range [0, num_arms)). 
                    
        '''
        return self.optimal_value - self.arms[arm_index].mean