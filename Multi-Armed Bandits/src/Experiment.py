import numpy as np
import matplotlib.pyolot as plt

from algorithms import Greedy
from MultiArmedBandit import MultiArmedBandit
from reward_distributions import Bernoulli
from utils import create_inverse_schedule
from metrics import CummulativeRegret

class Experiment:
    '''
    '''
    def __init__(self, algo_list, exper_name, num_runs=1000, total_time=1000,
        num_arms=2, reward_dist=Bernoulli, metrics=None):
        '''
            Arguments:
            ---------
                - algo_dict (dict): Dictionary of algorithms whose 
                    performance needs to be compared where key is the
                    name of the algorithm (string) and value should be 
                    a tuple with first element as an instance of 
                    Algorithm class and second element being dictionary
                    of hyperparameters of the algorithm.
                - exper_name (string): Name of the experiment.
                - num_runs (int): Number of times the experiment should
                    be repeated.
                - totoal_time (int): Total time till when the algorithm
                    should play the arms.
                - num_arms (int): Number of arms in the Multi Armed 
                    Bandit.
                - reward_dist (RewardDistribution class): The class of
                    reward distribution for the arms.
                - metrics (list): Metrics to be calculated to compare 
                    the performance of the algorithms. Each element of
                    the list should be of type Metric class.
        '''
        self.algo_dict = algo_dict
        self.exper_name = exper_name
        self.num_runs = num_runs
        self.total_time = total_time
        self.num_arms = num_arms
        self.reward_dist = reward_dist
        self.metrics = metrics
        
    def _init_params(self):
        '''
            This method initialises the results dictionary which will 
            contain the results for each algorithm.
        '''
        self.results = {}
        
        for algo_name in self.algo_dict:
            self.results[algo_name] = {}
            self.results[algo_name]["regrets"] = np.empty(
                (self.num_runs, self.total_time))
            self.results[algo_name]["counts"] = np.empty(
                (self.num_runs, self.num_arms))
            self.results[algo_name]["optimal_arm"] = np.empty(
                (self.num_runs, 1))
            self.results[algo_name]["metrics"] = {}
        
    def simulate(self):
        for i in range(self.num_runs):
            multi_armed_bandit = MultiArmedBandit(
                self.num_arms, 
                self.reward_dist
            )
            
            for algo_name in self.algo_dict:
                algo, hyperparams = self.algo_dict[algo_name]
                
                hyperparams["multi_arm_bandit"] = multi_arm_bandit
                hyperparams["total_time"] = self.total_time
                
                algo = algo(**hyperparams)
                algo.run()
                
                self.results[algo_name]["regrets"][i, :] = algo.regrets
                self.results[algo_name]["counts"][i, :] = algo.counts
                self.results[algo_name]["optimal_arm"][i, 0] = (
                    multi_arm_bandit.optimal_arm) 
                
        self._calculate_metrics()
                
    def _calculate_metrics(self):
        for algo_name in self.results:
            for metric in self.metrics:
                metric = metric(self.results[algo_name])
                self.results[algo_name]["metrics"][metric.__name__] = metric()
                
    def generate_plots(self):
        pass 
            
            # regrets = 
        # for i in range(1):
        # self.regrets = np.empty((1, 100))
            # print(i)
            
            # self.algo = Greedy(self.multi_armed_bandit, total_time=100, eps_schedule=create_inverse_schedule(0.1))
            # for j in self.multi_armed_bandit.arms:
                # print(j.mean)
            # print('-'*10)
            # self.algo.run()
            # print(self.algo.act_val_esti)
            # self.regrets[i, :] = self.algo.regrets

        