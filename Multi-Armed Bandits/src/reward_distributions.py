import numpy as np

from Distribution import Distribution

class RewardDistribution(Distribution):
    '''
        This is an abstract class for the Reward Distribution of an arm.
        For example, for a Bernoulli Bandits, the reward distribution is
        a Bernoulli distribution.
    '''
    
    def __init__(self, mean, variance=None):
        super().__init__(mean, variance)
    
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
        
class Bernoulli(RewardDistribution):
    '''
        This class implements the Bernoulli reward distribution.
    '''
    def __init__(self, p=None):
        '''
            Arguments:
            ---------
                - p (float): The success probability for a Bernoulli
                    distribution. If not provided, will be initialised
                    randomly using uniform distribution between [0, 1). 
        '''
        if p is None:
            p = self._init_params()
        
        super().__init__(p)
            
        
    def _init_params(self):
        '''
            This method initialises the success probability of a 
            Bernoulli distribution randomly using uniform distribution
            between [0, 1].
        '''
        return np.random.uniform(low=0.0, high=1.0)
        
    def sample(self):
        return np.random.binomial(n=1, p=self.mean)