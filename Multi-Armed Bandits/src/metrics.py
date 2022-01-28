import numpy as np

class Metric:
    '''
        This is an abstract class for different metrics for evaluating
        the performance of different Multi-Arm Bandit Problem.
    '''
    
    def __init__(self):
        pass
        
    def _get_appropriate_quantity(self):
        pass
        
    def __call__(self):
        pass
        
class MeanCummulativeRegret(Metric):
    '''
    '''
    def __init__(self):
        self.name = "Mean Cummulative Regret"
    
    def __call__(self, regrets):
        cum_sum = np.cumsum(regrets, axis=1)
        return np.mean(cum_sum, axis=0)