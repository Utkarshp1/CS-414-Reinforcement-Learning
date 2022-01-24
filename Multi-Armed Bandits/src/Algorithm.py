import numpy as np

class Algorithm:
    '''
        This class is an abstract class for implementing an algorithm
        for Multi Armed Bandit problem.
    '''
    def __init__(self, multi_arm_bandit, total_time=1000):
        '''
            Arguments:
            ---------
                - multi_arm_bandit (MultiArmedBandit class): Object of
                    MultiArmedBandit class
                - total_time (int): Total number of timestamps for which
                    the algorithm should be run.
        '''
        self.multi_arm_bandit = multi_arm_bandit
        self.total_time = total_time
        self.regrets = np.empty((self.total_time, ))
        self.counts = [0]*self.multi_arm_bandit.num_arms
        
    def _init_params(self):
        '''
            This method should initialize all the parameters of the
            algorithm.
        '''
        raise NotImplementedError
        
    def _initial_runs(self):
        '''
            This method should implement any initial default runs for
            the algorithm and should return the number of initial runs
            performed. For example, in the UCB algorithm, initially
            each of the arm is picked once.
        '''
        raise NotImplementedError
        
    def _update_params(self, reward, arm_index):
        '''
            This method should update the parameters of the algorithm
            after each timestamp.
        '''
        raise NotImplementedError
        
    def _pick_next_arm(self):
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
        num_runs = self._initial_runs()
        
        for i in range(num_runs, self.total_time):
            next_arm = self._pick_next_arm()
            # print(next_arm)
            reward = self.multi_arm_bandit.arms[next_arm].sample()
            
            self._update_params(reward, next_arm, i)