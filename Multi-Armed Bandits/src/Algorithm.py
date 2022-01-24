from numpy as np

class Algorithm:
    '''
        This class is an abstract class for implementing an algorithm
        for Multi Armed Bandit problem.
    '''
    def __init__(self):
        pass
        
    def _init_params(self):
        '''
            This method should initialize all the parameters of the
            algorithm.
        '''
        raise NotImplementedError
        
    def _initial_runs(self):
        '''
            This method should implement any initial default runs for
            the algorithm. For example, in the UCB algorithm, initially
            each of the arm is picked once.
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