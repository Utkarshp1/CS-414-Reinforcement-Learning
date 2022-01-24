import numpy as np

from Greedy import Greedy
from MultiArmedBandit import MultiArmedBandit
from BernoulliDistribution import Bernoulli

class Experiment:
    '''
    '''
    def __init__(self):
        self.regrets = np.empty((100, 100))
        
    def simulate(self):
        for i in range(100):
            # print(i)
            self.multi_armed_bandit = MultiArmedBandit(2, Bernoulli)
            self.algo = Greedy(self.multi_armed_bandit, total_time=100)
            for j in self.multi_armed_bandit.arms:
                print(j.mean)
            print('-'*10)
            self.algo.run()
            print(self.algo.act_val_esti)
            self.regrets[i, :] = self.algo.regrets

        