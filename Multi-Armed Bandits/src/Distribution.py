import numpy as np

class Distribution:
    '''
        This is an abstract class for probability distributions.
    '''
    def __init__(self, mean, variance):
        self.mean = mean
        self.variance = variance
        
    def posterior_update(self):
        '''
            This method implements the Bayesian posterior update for
            algorithms which use Bayesian approach to model the reward
            distribution such as Thompson sampling.
        '''
        raise NotImplementedError