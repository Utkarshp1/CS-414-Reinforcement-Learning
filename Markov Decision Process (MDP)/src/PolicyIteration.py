import numpy as np

from MDP import MDP

class PolicyIteration:
    '''
        This class implements the Policy Iteration algorithm for finding
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
        self.history = {'policy': [], 'val_func': []}

    def run(self) -> None:
        '''
            This method implements the Policy Iteration Algorithm.
        '''

        policy = np.ones((self.mdp.num_states, 1), dtype=np.int32)
        policy_prime = np.zeros((self.mdp.num_states, 1), dtype=np.int32)
        self.num_iters = 0
        count = 0

        while np.any(policy != policy_prime):
            print(count)
            count+=1
            self.num_iters += 1
            policy = policy_prime.copy()

            val_func_policy = self._policy_evaluation(policy)
            if count == 1:
                print(val_func_policy)
            state_act_val_func = self._calc_state_act_val_func(val_func_policy)
            policy_prime = self._policy_improvement(state_act_val_func)

            self.history['policy'].append(policy)
            self.history['val_func'].append(val_func_policy)

        self.opt_val_func = val_func_policy
        self.opt_policy = policy

    def _policy_improvement(self, state_act_val_func : np.ndarray) -> np.ndarray:
        '''
            This method greedily picks the action which has the highest 
            value of state action value function for each state.

            Arguments:
            ---------
                - state_act_val_func (np.ndarray): A NumPy array of 
                    shape (num_states, num_actions) which stores the
                    value of the state action values.

            Returns:
            -------
                An NumPy array of shape (num_states, 1)
        '''
        return np.expand_dims(np.argmax(state_act_val_func, axis=1), axis=1)

    def _policy_evaluation(self, policy : np.ndarray) -> np.ndarray:
        '''
            This method finds the total expected discounted reward for
            a state given the agent follows the policy, i.e. the value
            function for each state.

            Arguments:
            ---------
                - policy (np.ndarray): A NumPy array of shape 
                    (num_states, 1) indicating the action is to be 
                    picked in a given state.

            Returns:
            -------
                An NumPy array of shape (num_states, 1)
        '''

        delta = (1-self.mdp.gamma)/self.mdp.gamma
        val_func = np.zeros((self.mdp.num_states, 1))
        val_func_prime = np.zeros((self.mdp.num_states, 1))

        while (delta > self.eps*(1-self.mdp.gamma)/self.mdp.gamma):
            delta = 0
            trans_prob_policy = self._get_trans_prob_policy(policy)
            reward_func_policy = self._get_reward_func_policy(policy)
            # print(reward_func_policy)
            val_func = val_func_prime.copy()
            # print(val_func)

            reward = reward_func_policy + self.mdp.gamma*(val_func.T)
            # print(reward)
            val_func_prime = np.sum(trans_prob_policy*reward, axis=1, 
                keepdims=True)

            delta = np.amax(np.abs(val_func_prime - val_func))

        return val_func_prime

    def _get_trans_prob_policy(self, policy : np.ndarray) -> np.ndarray:
        '''
            This method calculates the matrix P_mu (the transition
            probability matrix for a given policy) i.e. the transition
            probabilities are according to action suggested by the 
            policy.

            Arguments:
            --------
                - policy (np.ndarray): A NumPy array of shape 
                    (num_states, 1) indicating the action is to be 
                    picked in a given state.

            Returns:
            -------
                An NumPy array of shape (num_states, num_states)
        '''

        trans_prob_policy = np.zeros((self.mdp.num_states, 
            self.mdp.num_states))

        for i in range(self.mdp.num_states):
            trans_prob_policy[i, :] = self.mdp.trans_prob[i, :, policy[i, 0]]

        return trans_prob_policy

    def _get_reward_func_policy(self, policy : np.ndarray) -> np.ndarray:
        '''
            This method calculates the reward function for the actions
            suggested by the policy.

            Arguments:
            ---------
                - policy (np.ndarray): A NumPy array of shape 
                    (num_states, 1) indicating the action is to be 
                    picked in a given state.

            Returns:
            -------
                An NumPy array of shape (num_states, num_states)

        '''
        reward_func_policy = np.zeros((self.mdp.num_states, 
            self.mdp.num_states))

        for i in range(self.mdp.num_states):
            reward_func_policy[i, :] = self.mdp.reward[i, :, policy[i]]

        return reward_func_policy

    def _calc_state_act_val_func(self, 
        val_func_policy : np.ndarray) -> np.ndarray:
        '''
            This method calculates the state action value function i.e.
            the total expected discounted reward that we collect given
            a state and an action and then following the policy.

            Arguments:
            ---------
                - val_func_policy (np.ndarray) : An NumPy array of shape
                    (num_states, 1) which contains the value function
                    for each state.

            Returns:
            -------
                A NumPy array of shape (num_states, num_actions)
        '''
        state_act_val_func = np.zeros((self.mdp.num_states, 
            self.mdp.num_actions))

        for i in range(self.mdp.num_actions):
            reward = (self.mdp.reward[:, :, i] + 
                self.mdp.gamma*(val_func_policy.T))
            state_act_val_func[:, i] = np.sum(
                self.mdp.trans_prob[:, :, i]*reward, axis=1)

        return state_act_val_func