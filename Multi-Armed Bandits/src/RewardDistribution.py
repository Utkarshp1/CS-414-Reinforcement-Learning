import numpy as np

from Distribution import Distribution

class RewardDistribution(Distribution):
    '''
        This is an abstract class for the Reward Distribution of an arm.
        For example, for a Bernoulli Bandits, the reward distribution is
        a Bernoulli distribution.
    '''
    
    def __init__(self):
        self._init_params()
    
    def _init_params(self):
        '''
            This method randomly initializes the parameters of the 
            reward distribution. For example, for a Bernoulli 
            distribution it will initialize the success rate (or mean).
        '''
        raise NotImplementedError
        
    def sample(self):
        '''
            This method returns a sample from the reward distribution.
        '''
        raise NotImplementedError
    
    def ucb_update(self):
        '''
            This method implements the UCB uncertainty calculation for
            the given reward distribution.
        '''
        raise NotImplementedError
        