import numpy as np

from MDP import MDP

class ValueIteration:
    '''
        This class implements the Value Iteration algorithm for finding
        the optimal policy in an MDP.
    '''

    def __init__(self, mdp : MDP, eps : float) -> None:
        '''
            Arguments:
            ---------
                - mdp (An instance of MDP class): MDP for which the
                    optimal policy is to be found.
                - eps (float): The policy evaluation algorithm stops
                    when the difference between value function of two
                    consecutive iteration is less than eps.
        '''
        self.mdp = mdp
        self.eps = eps
        self.opt_policy = np.full((self.mdp.num_states, 1), -1)

    def run(self) -> None:
        '''
            This method implements the Value Iteration Algorithm.
        '''
        val_func = np.zeros((self.mdp.num_states, 1))
        val_func_prime = np.zeros((self.mdp.num_states, 1))
        delta = (1-self.mdp.gamma)/self.mdp.gamma

        while (delta > self.eps*(1-self.mdp.gamma)/self.mdp.gamma):
            delta = 0
            val_func = val_func_prime.copy()
            
            for i in range(self.mdp.num_states):
                product = np.dot(self.mdp.trans_prob[i, :, :].T, val_func)
                exp_imm_reward = np.sum(
                    self.mdp.trans_prob[i, :, :]*self.mdp.reward[i, :, :], 
                    axis=0, 
                    keepdims=True).T

                self.opt_policy[i, 0] = np.argmax(exp_imm_reward + 
                    self.mdp.gamma*product, axis=0)
                val_func_prime[i, 0] = np.amax(exp_imm_reward + 
                    self.mdp.gamma*product, axis=0)
                
                if np.abs(val_func_prime[i, 0] - val_func[i, 0]) > delta:
                    delta = np.abs(val_func_prime[i, 0] - val_func[i, 0])

        self.opt_val_func = val_func_prime