import os

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from algorithms import Greedy
from MultiArmedBandit import MultiArmedBandit
from reward_distributions import Bernoulli
from utils import create_inverse_schedule
from metrics import MeanCummulativeRegret, AverageReward

class Experiment:
    '''
    '''
    def __init__(self, algo_dict, exper_name, num_runs=1000, total_time=1000,
        num_arms=2, reward_dist=Bernoulli, metrics=None, random_seed=42):
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
                - random_seed (int): NumPy random seed.
        '''
        self.algo_dict = algo_dict
        self.exper_name = exper_name
        self.num_runs = num_runs
        self.total_time = total_time
        self.num_arms = num_arms
        self.reward_dist = reward_dist
        self.metrics = metrics
        self.random_seed = random_seed
        
        self._init_params()
        self._init_directory()
        
        np.random.seed(random_seed)
        
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
            self.results[algo_name]["rewards"] = np.empty(
                (self.num_runs, self.total_time))
            self.results[algo_name]["metrics"] = {}
            
    def _init_directory(self):
        '''
            This method initialises the directory for saving the results
            of the experiment.
        '''
        try:
            self.exp_dir = os.path.join("../Experiments/", self.exper_name)
            os.mkdir(self.exp_dir)
        except Exception as e:
            import traceback
            # traceback.print_exc()
        
    def simulate(self):
        multi_armed_bandit = MultiArmedBandit(
            self.num_arms, 
            self.reward_dist
        )
        
        mean_reward = [arm.mean for arm in multi_armed_bandit.arms]
        print("Expected Reward given an arm:", mean_reward)
        
        for i in tqdm(range(self.num_runs)):
            for algo_name in self.algo_dict:
                algo, hyperparams = self.algo_dict[algo_name]
                
                hyperparams["multi_arm_bandit"] = multi_armed_bandit
                hyperparams["total_time"] = self.total_time
                
                algo = algo(**hyperparams)
                algo.run()
                
                self.results[algo_name]["regrets"][i, :] = algo.regrets
                self.results[algo_name]["counts"][i, :] = algo.counts
                self.results[algo_name]["optimal_arm"][i, 0] = (
                    multi_armed_bandit.optimal_arm)
                self.results[algo_name]["rewards"][i, :] = algo.rewards
                
        self._calculate_metrics()
                
    def _calculate_metrics(self):
        avg_reward = AverageReward()
        for algo_name in self.results:
            for metric in self.metrics:
                self.results[algo_name]["metrics"][metric.name] = metric(
                    self.results[algo_name]["regrets"]).tolist()
                    
            self.results[algo_name]["metrics"][avg_reward.name] = (
                avg_reward(self.results[algo_name]["rewards"]).tolist())
                
    def generate_plots(self):
        avg_reward = AverageReward()
        for metric in self.metrics + [avg_reward]:
            plt.figure()
            for algo_name in self.results:
                plt.plot(self.results[algo_name]["metrics"][metric.name], 
                    label=algo_name)
            plt.legend()
            plt.title(metric.name)
            plt.xlabel("Time")
            plt.ylabel(metric.name)
            plt.savefig(os.path.join(self.exp_dir, 
                metric.name.replace(' ', '_') + ".png"))
            plt.show()
        
        
            
    def generate_report(self):
        self.generate_plots()
        
        exp_config = {
            "exper_name": self.exper_name,
            "num_runs": self.num_runs,
            "total_time": self.total_time,
            "num_arms": self.num_arms,
            "reward_distribution": self.reward_dist.__class__.__name__,
            "random_seed": self.random_seed,
            "algos": []
        }
        
        for algo_name in algo_dict:
            algo, hyperparams = self.algo_dict[algo_name]
                
            hyperparams["multi_arm_bandit"] = multi_armed_bandit
            hyperparams["total_time"] = self.total_time
            
            algo = algo(**hyperparams)
            
            exp_config["algos"][algo_name] = algo.get_config()
            
        with open(os.path.join(self.exp_dir, "summary.json", "w")) as outfile: 
            json.dump(exp_config, outfile)        