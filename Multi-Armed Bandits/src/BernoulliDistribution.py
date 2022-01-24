from numpy as np

from RewardDistribution import RewardDistribution

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
        if p:
            self.p = p
        else:
            self._init_params()
        
    def _init_params(self):
        '''
            This method initialises the success probability of a 
            Bernoulli distribution randomly using uniform distribution
            between [0, 1].
        '''
        self.p = np.random.uniform(low=0.0, high=1.0)
        
    def sample(self):
        return np.random.binomial(n=1, p=self.p)