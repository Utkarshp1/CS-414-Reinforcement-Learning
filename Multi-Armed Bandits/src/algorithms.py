import numpy as np

from utils import running_avg_update

class Algorithm:
    '''
        This class is an abstract class for implementing an algorithm
        for Multi Armed Bandit problem.
    '''
    def __init__(self, multi_arm_bandit=None, total_time=1000):
        '''
            Arguments:
            ---------
                - multi_arm_bandit (MultiArmedBandit class): Object of
                    MultiArmedBandit class
                - total_time (int): Total number of timestamps for which
                    the algorithm should be run.
        '''
        self.multi_arm_bandit = multi_arm_bandit
        self.total_time = total_time
        self.regrets = np.empty((self.total_time, ))
        self.counts = [0]*self.multi_arm_bandit.num_arms
        
    def _init_params(self):
        '''
            This method should initialize all the parameters of the
            algorithm.
        '''
        raise NotImplementedError
        
    def _initial_runs(self):
        '''
            This method should implement any initial default runs for
            the algorithm and should return the number of initial runs
            performed. For example, in the UCB algorithm, initially
            each of the arm is picked once.
        '''
        raise NotImplementedError
        
    def _update_params(self, reward, arm_index):
        '''
            This method should update the parameters of the algorithm
            after each timestamp.
        '''
        raise NotImplementedError
        
    def _pick_next_arm(self):
        '''
            This method should return the index of the next arm picked
            by the algorithm.
        '''
        raise NotImplementedError
        
    def run(self):
        '''
            This method should implement the running of the whole 
            algorithm.
        '''
        num_runs = self._initial_runs()
        
        for i in range(num_runs, self.total_time):
            next_arm = self._pick_next_arm()
            # print(next_arm)
            reward = self.multi_arm_bandit.arms[next_arm].sample()
            
            self._update_params(reward, next_arm, i)
            
    def get_config(self):
        '''
            This method returns the hyperparameters for the algorithm.
        '''
        raise NotImplementedError
            
            
class Greedy(Algorithm):
    '''
        This class implements the Greedy, epsilon-greedy (fixed and 
        variable) algorithm for Multi-Armed Bandit.
    '''
    def __init__(self, multi_arm_bandit=None, eps=0, eps_schedule=None,
        total_time=1000):
        '''
            Arguments:
            ---------
                - multi_arm_bandit (MultiArmedBandit class): Object of
                    MultiArmedBandit class
                - eps (float): The value of epsilon for running 
                    epsilon-greedy algorithm.
                - eps_schedule (function): A function for decreasing the
                    value of the epsilon over time.
                - total_time (int): Total number of timestamps for which
                    the algorithm should be run.
            
            NOTE: If eps = 0 and eps_schedule is None, then pure Greedy
            algorithm will be run.
        '''
        
        super().__init__(multi_arm_bandit, total_time)

        self.eps = eps
        self.eps_schedule = eps_schedule
        
        self._init_params()
        
    def _init_params(self):
        '''
            This method should initialize all the parameters of the
            algorithm.
        '''
        
        self.act_val_esti = np.zeros((self.multi_arm_bandit.num_arms, ))
        
    def _initial_runs(self):
        '''
            This method plays all the arm once and returns the number
            of arms.
        '''
        
        for i in range(self.multi_arm_bandit.num_arms):
            reward = self.multi_arm_bandit.arms[i].sample()
            
            self._update_params(reward, i, i)
            
        return self.multi_arm_bandit.num_arms
        
    def _update_params(self, reward, arm_index, time):
        '''
            This method should update the parameters of the algorithm
            after each timestamp.
        '''
        
        self.regrets[time] = self.multi_arm_bandit.calculate_regret(
            reward, arm_index)
        self.counts[arm_index] += 1
        
        self.act_val_esti[arm_index] = running_avg_update(
            self.act_val_esti[arm_index],
            reward,
            alpha= 1/self.counts[arm_index]
        )
        
        if self.eps_schedule:
            self.eps = self.eps_schedule(time+1)
            # print(self.eps)
        
        
    def _pick_next_arm(self):
        '''
            This method should return the index of the next arm picked
            by the algorithm.
        '''
        num_arms = self.multi_arm_bandit.num_arms
        greedy_arm = np.argmax(self.act_val_esti)
        
        arm_prob = [self.eps/num_arms]*num_arms
        arm_prob[greedy_arm] += (1 - self.eps)
        return np.random.choice(num_arms, p=arm_prob)
        
    def get_config(self):
        '''
            This function returns the epsilon value if the epsilon used
            is fixed otherwise it returns the configurations of the 
            schedule that decreases the epsilon over time.
        '''
        params = {}
        
        if self.eps_schedule:
            params["algo_name"] = "Variable epsilon-greedy"
            params["schedule"] = {}
            params["schedule"]["name"] = self.eps_schedule.__name__
            params["schedule"]["config"] = self.eps_schedule.config
        elif self.eps != 0:
            params["algo_name"] = "Epsilon-greedy"
            params["eps"] = self.eps
        else:
            params["algo_name"] = "Greedy"
            
        print(params)
        
        return params