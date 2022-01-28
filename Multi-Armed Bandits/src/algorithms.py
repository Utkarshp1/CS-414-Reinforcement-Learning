import numpy as np

from Distribution import Beta
from utils import running_avg_update, softmax

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
            By default, we return 0 that means the algorithm has no 
            initial runs.
        '''
        return 0
        
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
        return {}
            
            
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
        
class SoftmaxPolicy(Algorithm):
    '''
        This class implements the softmax policy for Multi-Armed Bandit
        which picks each arm with a probability proportional to its
        average reward.        
        
        Refer: Algorithms for the multi-armed bandit problem Volodymyr
        Kuleshov and Doina Precup, Journal of Machine Learning Research.
    '''
    def __init__(self, multi_arm_bandit=None, temp=0.1, temp_schedule=None, 
        total_time=1000):
        '''
            Arguments:
            ---------
                - multi_arm_bandit (MultiArmedBandit class): Object of
                    MultiArmedBandit class
                - total_time (int): Total number of timestamps for which
                    the algorithm should be run.
                - temp (float): Temperature value to be used for 
                    algorithm. If temperature = 0, then softmax policy
                    acts like pure greedy. When temperature tends 
                    infinity, the algorithm picks arms uniformly at
                    random.
                - temp_schedule (function): A function for decreasing 
                    the value of the temperature over time.
        '''
        
        super().__init__(multi_arm_bandit, total_time)
        
        self.temp = temp
        self.temp_schedule = temp_schedule
        
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
        
        if self.temp_schedule:
            self.temp = self.temp_schedule(time+1)
    
    def _pick_next_arm(self):
        '''
            This method should return the index of the next arm picked
            by the algorithm.
        '''
        num_arms = self.multi_arm_bandit.num_arms
        
        arm_prob = softmax(self.act_val_esti/self.temp)
        return np.random.choice(num_arms, p=arm_prob)
        
    def get_config(self):
        '''
            This function returns the value of temperature if the 
            temperature is constant otherwise it returns the 
            configurations of the schedule that decreases the 
            temperature over time.
        '''
        params = {}
        
        if self.temp_schedule:
            params["algo_name"] = "Variable temp-softmax"
            params["schedule"] = {}
            params["schedule"]["name"] = self.temp_schedule.__name__
            params["schedule"]["config"] = self.temp_schedule.config
        else:
            params["algo_name"] = "Softmax"
            params["temp"] = self.temp
        
        return params
        
class UCB(Algorithm):
    '''
        This class implements the Upper Confidence Bound (UCB) algorithm
        for Multi-Armed Bandit problem.
        
        Refer:
        -----
            - Finite-Time Analysis of the Multiarmed Bandit Problem,
              Auer, Cesa-Bianchi, Fischer. 2002
            - Reinforcement Learning: An Introduction, Sutton and Barto
              2nd Edition, 2018.
    '''
    
    def __init__(self, multi_arm_bandit=None, C=1, C_schedule=None, 
        total_time=1000):
        '''
            Arguments:
            ---------
                - multi_arm_bandit (MultiArmedBandit class): Object of
                    MultiArmedBandit class
                - total_time (int): Total number of timestamps for which
                    the algorithm should be run.
                - C (float): The coefficient of the uncertainty (bonus)
                    term in the UCB algorithm which controls the 
                    exploration-exploitation tradeoff. For larger values
                    of C, we explore more whereas for smaller values of
                    C, we explore less.
                - C_schedule (function): A function for decreasing the
                    value of the C over time.
        '''
        
        super().__init__(multi_arm_bandit, total_time)
        
        self.C = C
        self.C_schedule = C_schedule
        
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
        
        if self.C_schedule:
            self.temp = self.C_schedule(time+1)
            
    def _pick_next_arm(self):
        '''
            This method should return the index of the next arm picked
            by the algorithm.
        '''
        num_arms = self.multi_arm_bandit.num_arms
        
        bonus_term = self.multi_arm_bandit.arms[0].ucb_bonus_term(self.counts)
        
        ucb = self.act_val_esti + self.C*bonus_term
        return np.argmax(ucb)
        
    def get_config(self):
        '''
            This function returns the C value if the C used
            is fixed otherwise it returns the configurations of the 
            schedule that decreases the C over time.
        '''
        params = {}
        
        if self.C_schedule:
            params["algo_name"] = "Variable C-UCB"
            params["schedule"] = {}
            params["schedule"]["name"] = self.C_schedule.__name__
            params["schedule"]["config"] = self.C_schedule.config
        else:
            params["algo_name"] = "UCB"
            params["eps"] = self.C
        
        return params
        
class ThompsonSampling(Algorithm):
    '''
        This class implements the Thompson Sampling algorithm for the
        Multi-Armed Bandit problem.
        
        Refer:
        -----
            - Algorithms for the multi-armed bandit problem Kuleshov and 
              Precup, Journal of Machine Learning Research.
    '''
    
    def __init__(self, multi_arm_bandit=None, prior_dist=Beta, 
        total_time=1000):
        '''
            Arguments:
            ---------
                - multi_arm_bandit (MultiArmedBandit class): Object of
                    MultiArmedBandit class
                - total_time (int): Total number of timestamps for which
                    the algorithm should be run.
                - prior_dist (Distribution): Should be an instance of 
                    Distribution class which will be used as a prior 
                    distribution over the rewards of the arms in the 
                    Multi-armed Bandit problem. Note: It should be class
                    rather than instance of the class so that multiple
                    objects can be created from it.
        '''
        
        super().__init__(multi_arm_bandit, total_time)
        
        self.prior_dist = prior_dist
        
        self._init_params()
        
    def _init_params(self):
        '''
            This method should initialize all the parameters of the
            algorithm.
        '''
        
        self.posterior = [self.prior_dist() for _ in 
            range(self.multi_arm_bandit.num_arms)]
            
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
        
        self.posterior[arm_index].posterior_update(reward)
        
    def _pick_next_arm(self):
        '''
            This method should return the index of the next arm picked
            by the algorithm.
        '''
        num_arms = self.multi_arm_bandit.num_arms
        
        sampled_act_val = np.array([post.sample() for post in self.posterior])

        return np.argmax(sampled_act_val)
        
class Reinforce(Algorithm):
    '''
        This class implements the Reinforce algorithm (Gradient Bandit 
        Algorithm).
        
        Refer:
        -----
            - Reinforcement Learning: An Introduction, Sutton and Barto
              2nd Edition, 2018.
    '''
    def __init__(self, multi_arm_bandit=None, lr=0.001,
        total_time=1000, use_baseline=False, init_theta=None):
        '''
            Arguments:
            ---------
                - multi_arm_bandit (MultiArmedBandit class): Object of
                    MultiArmedBandit class
                - total_time (int): Total number of timestamps for which
                    the algorithm should be run.
                - lr (float): The learning rate that should be used for 
                    performing Stochastic Gradient Ascent.
                - use_baseline (boolean): Whether the algorithm should 
                    use baselines or not. If True, then the algorithm
                    uses average reward uptil that time as the baseline.
                - init_theta (1D NumPy Array): The initial parameters of
                    the policy (Should be a 1D NumPy array with number 
                    of parameters equal to the number of arms). If not
                    provided then parameters are randomly initialised
                    using Nomral Distribution with mean = 0 and 
                    variance = 1.
                    
            NOTE: We parametrize our policy using softmax function.
        '''
        
        super().__init__(multi_arm_bandit, total_time)
        
        self.lr = lr
        self.use_baseline = use_baseline
        self.theta = init_theta
        
        self._init_params()
        
    def _init_params(self):
        '''
            This method should initialize all the parameters of the
            algorithm.
        '''
        
        self.rewards = np.zeros((self.total_time, ))
        if not self.theta:
            self.theta = np.random.randn(self.multi_arm_bandit.num_arms)
        
    def _update_params(self, reward, arm_index, time):
        '''
            This method should update the parameters of the algorithm
            after each timestamp.
            
            theta = theta + lr*reward*(1-pi_t(arm_index)) for arm_index
            theta = theta - lr*reward*pi_t(arm_index)
        '''
        
        self.regrets[time] = self.multi_arm_bandit.calculate_regret(
            reward, arm_index)
        self.counts[arm_index] += 1
        self.rewards[time] = reward
        
        pi = softmax(self.theta)
        
        baseline = np.sum(self.rewards)/(time+1) if self.use_baseline else 0
        
        tmp = self.theta[arm_index]
        self.theta -= self.lr *(reward - baseline)* pi
        self.theta[arm_index] = (tmp + 
            self.lr * (reward - baseline)*(1 - pi[arm_index]))
            
    def _pick_next_arm(self):
        '''
            This method should return the index of the next arm picked
            by the algorithm.
        '''
        
        num_arms = self.multi_arm_bandit.num_arms
        pi = softmax(self.theta)
        
        return np.random.choice(num_arms, p=pi)
        
    def get_config(self):
        '''
            This function returns the different hyperparameters 
            configurations for the algorithm.
        '''
        
        params = {}
        
        if self.use_baseline:
            params["algo_name"] = "Reinforce with Baseline"
        else:
            params["algo_name"] = "Reinforce"
            
        params["lr"] = self.lr
        
        return params