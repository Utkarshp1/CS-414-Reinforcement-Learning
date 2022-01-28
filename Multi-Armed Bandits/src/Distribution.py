import numpy as np

class Distribution:
    '''
        This is an abstract class for probability distributions.
    '''
    def __init__(self, mean, variance=None):
        self.mean = mean
        self.variance = variance
        
    def posterior_update(self):
        '''
            This method implements the Bayesian posterior update for
            algorithms which use Bayesian approach to model the reward
            distribution such as Thompson sampling.
        '''
        raise NotImplementedError
        
class Beta(Distribution):
    '''
        This class implements the Beta distribution.
        
        Refer:
        -----
            - Introduction to Probability, Blitzstein and Hwang, 2015.
    '''
    def __init__(self, alpha=1, beta=1):
        self.alpha = alpha
        self.beta = beta
        
        super().__init__(alpha/(alpha + beta))
        
    def posterior_update(self, reward):
        '''
            This method implements the Bayesian posterior update for
            the Beta Distribution.
                alpha = alpha + 1 if reward = 0
                beta = beta + 1 if reward = 1
        '''
        self.alpha += reward
        self.beta += (1-reward)
        
    def sample(self):
        '''
            This function returns a sample from the Beta distribution.
        '''
        return np.random.beta(self.alpha, self.beta)