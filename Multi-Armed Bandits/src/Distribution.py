import numpy as np

from utils import running_avg_update

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
        
class Normal(Distribution):
    '''
        This class implements the Normal distribution.
    '''
    def __init__(self, mean=0, variance=1, data_variance=0.01):
        self.init_mean = mean
        self.init_variance = variance
        self.data_variance = data_variance
        self.avg_reward = 0
        self.count = 1
        
        super().__init__(mean, variance)
        
    def posterior_update(self, reward):
        '''
            This method implements the Bayesian posterior update
            for the Normal Distribution where the variance of the
            Normal distribution from which data comes is known.
        '''

        self.variance = 1/(1/self.init_variance + self.count/self.data_variance)

        self.avg_reward = running_avg_update(
            self.avg_reward, 
            reward, 
            alpha=1/self.count
        )
        
        part1 = self.init_mean/self.init_variance
        part2 = self.avg_reward*self.count/self.data_variance
        self.mean = self.variance*(part1 + part2)
        
        self.count += 1
        
    def sample(self):
        '''
            This function returns a sample from the normal distribution.
        '''
        return np.random.normal(loc=self.mean, scale=np.sqrt(self.variance))