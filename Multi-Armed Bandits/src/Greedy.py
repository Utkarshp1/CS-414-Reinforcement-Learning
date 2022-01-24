import numpy as np

from Algorithm import Algorithm
from utils import running_avg_update

class Greedy(Algorithm):
    '''
        This class implements the Greedy, epsilon-greedy (fixed and 
        variable) algorithm for Multi-Armed Bandit.
    '''
    def __init__(self, multi_arm_bandit, eps=0, eps_schedule=None,
        total_time=1000):
        '''
            Arguments:
            ---------
                - multi_arm_bandit (MultiArmedBandit class): Object of
                    MultiArmedBandit class
                - eps (float): The value of epsilon for running 
                    epsilon-greedy algorithm.
                - eps_schedule (function): A function for decreasing the
                    value of the epsilon over time.
                - total_time (int): Total number of timestamps for which
                    the algorithm should be run.
            
            NOTE: If eps = 0 and eps_schedule is None, then pure Greedy
            algorithm will be run.
        '''
        
        super().__init__(multi_arm_bandit, total_time)

        self.eps = eps
        self.eps_schedule = eps_schedule
        
        self._init_params()
        
    def _init_params(self):
        '''
            This method should initialize all the parameters of the
            algorithm.
        '''
        
        self.act_val_esti = np.zeros((self.multi_arm_bandit.num_arms, ))
        
    def _initial_runs(self):
        '''
            This method plays all the arm once and returns the number
            of arms.
        '''
        
        for i in range(self.multi_arm_bandit.num_arms):
            reward = self.multi_arm_bandit.arms[i].sample()
            
            self._update_params(reward, i, i)
            
        return self.multi_arm_bandit.num_arms
        
    def _update_params(self, reward, arm_index, time):
        '''
            This method should update the parameters of the algorithm
            after each timestamp.
        '''
        
        self.regrets[time] = self.multi_arm_bandit.calculate_regret(
            reward, arm_index)
        self.counts[arm_index] += 1
        
        self.act_val_esti[arm_index] = running_avg_update(
            self.act_val_esti[arm_index],
            reward,
            alpha= 1/self.counts[arm_index]
        )
        
        
    def _pick_next_arm(self):
        '''
            This method should return the index of the next arm picked
            by the algorithm.
        '''
        # print(self.act_val_esti)
        return np.argmax(self.act_val_esti)