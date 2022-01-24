from numpy as np

from Algorithm import Algorithm

class Greedy(Algorithm):
    '''
        This class implements the Greedy, epsilon-greedy (fixed and 
        variable) algorithm for Multi-Armed Bandit.
    '''
    def __init__(self, multi_arm_bandit, eps=None, eps_schedule=None):
        '''
            Arguments:
            ---------
                - multi_arm_bandit (MultiArmedBandit class): Object of
                    MultiArmedBandit class
                - eps (float): The value of epsilon for running 
                    epsilon-greedy algorithm.
                - eps_schedule (function): A function for decreasing the
                    value of the epsilon over time.
            
            NOTE: If both eps and eps_schedule are None, then pure Greedy
            algorithm will be run.
        '''
        self.multi_arm_bandit = multi_arm_bandit
        self.eps = eps
        self.eps_schedule = eps_schedule
        _init_params()
        
    def _init_params(self):
        '''
            This method should initialize all the parameters of the
            algorithm.
        '''
        self.action_values = [0]*self.multi_arm_bandit.num_arms
        
    def _initial_runs(self):
        '''
            
        '''
        raise NotImplementedError
        
    def update_params(self):
        '''
            This method should update the parameters of the algorithm
            after each timestamp.
        '''
        raise NotImplementedError
        
    def pick_next_arm(self):
        '''
            This method should return the index of the next arm picked
            by the algorithm.
        '''
        raise NotImplementedError
        
    def run(self):
        '''
            This method should implement the running of the whole 
            algorithm.
        '''
        raise NotImplementedError