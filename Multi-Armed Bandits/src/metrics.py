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
    def __init__(self, results):
        self._get_appropriate_quantity(results)
        
    def _get_appropriate_quantity(self, results):
        self.regrets = results["regrets"]
    
    def __call__(self, regrets):
        cum_sum = np.cumsum(self.regrets, axis=1)
        return np.mean(cum_sum, axis=0)