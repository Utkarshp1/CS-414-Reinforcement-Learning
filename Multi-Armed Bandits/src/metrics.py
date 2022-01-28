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
        
class PercentOptimalArmPull(Metric):
    '''
    '''
    def __init__(self):
        self.name = "Percentage Optimal Arm Pulled"
        
    def __call__(self, regrets):
        tmp = np.ones_like(regrets)
        cum_arm_played = np.cumsum(tmp, axis=1)
        
        cum_optimal_arm_played = np.cumsum(
            (regrets == 0).astype(np.int32), axis=1)
            
        return np.mean(cum_optimal_arm_played/cum_arm_played, axis=0)