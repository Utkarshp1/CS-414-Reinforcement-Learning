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
        print(U_prime)

        self.best_value_function = U_prime   