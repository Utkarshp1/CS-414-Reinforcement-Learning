import numpy as np

class MDP:
    def __init__(self, trans_prob, reward, gamma):
        self.trans_prob = trans_prob
        self.reward = reward
        self.gamma = gamma
        self.m = trans_prob.shape[-1]
        self.n = trans_prob.shape[0]
        self.best_policy = np.zeros((self.n, 1))
        self.best_value_function = np.zeros((self.n, 1))
        
    def value_iteration(self, epsilon):
        U = np.zeros((self.n, 1))
        # U[3, 0] = 1
        # U[7, 0] = -100
        U_prime = np.zeros((self.n, 1))
        delta = (1-self.gamma)/self.gamma
        
        print(epsilon*(1-self.gamma)/self.gamma)
        while (delta > epsilon*(1-self.gamma)/self.gamma):
            # print("Hello")
        # for j in range(101):
            delta = 0
            U = U_prime.copy()
            # U[3, 0] = 1
            # U[7, 0] = -100
            for i in range(self.n):
                # if i==3 or i==7:
                #     continue
                # print(U)
                product = np.dot(self.trans_prob[i, :, :].T, U)
                # print(product.shape)
                exp_imm_reward = np.sum(self.trans_prob[i, :, :]*self.reward[i, :, :], axis=0, keepdims=True).T
                # if i == 7:
                #     print(product)
                # print(exp_imm_reward.shape)
                # print(product)
                self.best_policy[i, 0] = np.argmax(exp_imm_reward + self.gamma*product, axis=0)
                U_prime[i, 0] = np.amax(exp_imm_reward + self.gamma*product, axis=0)
                # if i==2:
                #     print(self.best_policy[i])
                #     print(U_prime[i])
                
                
                if np.abs(U_prime[i, 0] - U[i, 0]) > delta:
                    delta = np.abs(U_prime[i, 0] - U[i, 0])
                    # print(delta)
            # print(self.best_policy)
        # print(U_prime)

        self.best_value_function = U_prime

    def policy_iteration(self, epsilon):
        policy = np.ones((self.n, 1), dtype=np.int32)
        policy_prime = np.zeros((self.n, 1), dtype=np.int32)

        while np.any(policy != policy_prime):
            policy = policy_prime.copy()

            val_func_policy = self._policy_evaluation(policy, epsilon)
            state_act_val_func = self._calc_state_act_val_func(val_func_policy)
            policy_prime = self._policy_improvement(state_act_val_func)

        self.opt_val_func = val_func_policy
        self.opt_policy = policy


    def _policy_improvement(self, state_act_val_func):
        return np.expand_dims(np.argmax(state_act_val_func, axis=1), axis=1)

    def _policy_evaluation(self, policy, epsilon):
        delta = (1-self.gamma)/self.gamma
        val_func = np.zeros((self.n, 1))
        val_func_prime = np.zeros((self.n, 1))

        while (delta > epsilon*(1-self.gamma)/self.gamma):
            delta = 0
            trans_prob_policy = self._get_trans_prob_policy(policy)
            reward_func_policy = self._get_reward_func_policy(policy)
            val_func = val_func_prime.copy()

            reward = reward_func_policy + self.gamma*(val_func.T)
            val_func_prime = np.sum(trans_prob_policy*reward, axis=1)

            delta = np.amax(np.abs(val_func_prime - val_func))

        return val_func_prime

    def _get_trans_prob_policy(self, policy):
        trans_prob_policy = np.zeros((self.n, self.n))

        for i in range(self.n):
            # print(policy[i, 0])
            trans_prob_policy[i, :] = self.trans_prob[i, :, policy[i, 0]]

        return trans_prob_policy

    def _get_reward_func_policy(self, policy):
        reward_func_policy = np.zeros((self.n, self.n))

        for i in range(self.n):
            reward_func_policy[i, :] = self.reward[i, :, policy[i]]

        return reward_func_policy

    def _calc_state_act_val_func(self, val_func_policy):
        state_act_val_func = np.zeros((self.n, self.m))

        for i in range(self.m):
            reward = self.reward[:, :, i] + self.gamma*(val_func_policy.T)
            state_act_val_func[:, i] = np.sum(self.trans_prob[:, :, i]*reward, axis=1)

        return state_act_val_func