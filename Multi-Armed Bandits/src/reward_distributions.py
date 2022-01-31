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
    
    def ucb_bonus_term(self):
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
        
    def ucb_bonus_term(self, counts):
        '''
            This method implements the UCB uncertainty calculation for
            the given reward distribution.
            
            Arguments:
            ---------
                - counts (int): Number of times each arm has been 
                    played. 
        '''
        time = np.sum(counts)
        return np.sqrt(np.log(time)/counts)
        
class Normal(RewardDistribution):
    '''
        This class implements the Normal reward distribution.
    '''
    def __init__(self, mean=None, variance=None):
        '''
            Arguments:
            ---------
                - mean (float): The mean of the Normal Distribution.
                - variance (float): The variance of the Normal
                    Distribution.
        '''
        if mean is None or variance is None:
            mean, variance = self._init_params(mean, variance)
            
        super().__init__(mean, variance)
        
    def _init_params(self, mean, variance):
        '''
            It is initialises the mean or variance of the Normal 
            distribution if it is None randomly using a uniform
            distribution between [0, 1].
        '''
        if mean is None:
            mean = np.random.uniform(low=0.0, high=1.0)
            
        if variance is None:
            variance = np.random.uniform(low=0.0, high=1.0)
            
        return mean, variance
        
    def sample(self):
        return np.random.normal(loc=self.mean, scale=self.variance)
        
    def ucb_bonus_term(self, avg_reward, square_reward, counts):
        '''
            Arguments:
            ---------
                - avg_reward (1D NumPy array): A 1D NumPy array with
                    size equal to the number of arms in the Bandit
                    where jth element is the average reward obtained
                    from the jth arm.
                - square_reward (1D NumPy array): A 1D NumPy array with
                    size equal to the number of arms in the Bandit where
                    jth element is the sum of sqaure rewards obtained
                    from jth arm.
                - counts (1D NumPy array): A 1D NumPy array with
                    size equal to the number of arms in the Bandit where
                    jth element is the number of times the jth arm has
                    been pulled.
                    
            Refer: Finite-time Analysis of the Multiarmed Bandit 
            Problem, Auer, Cesa-Bianchi and Fischer, 2002.
        '''
        print(square_reward)
        print(counts)
        print(avg_reward)
        time = np.sum(counts)
        part1 = (square_reward - counts*(avg_reward**2))/(counts - 1)
        print(part1)
        part2 = np.log(time - 1)/counts
        print(part2)
        print(np.sqrt(16*part1*part2))
        return np.sqrt(16*part1*part2)