'''
    Jack's Car Rental Problem Ex. 4.2 from Reinforcement Learning: An
    Introduction by Richard S. Sutton and Andrew G. Barto. 2nd Edition.
    2018. 
'''

import yaml
import numpy as np
import seaborn as sns
from scipy.stats import poisson
import matplotlib.pyplot as plt

class JackCarRentalProblem:
    def __init__(self):
        self._get_config()

        self.num_actions = 2*self.config['MAX_CARS_MOVEMENT'] + 1
        self.num_states = (self.config['MAX_CARS'] + 1)**2

        self._poisson_cache = {}
        self._get_transition_dynamics

    def _get_config(self):
        with open('config.yaml') as config_file:
            self.config = yaml.safe_load(config_file)

    def _get_transition_dynamics(self):
        self.trans_prob = np.zeros((
            self.num_states,
            self.num_states,
            self.num_actions
        ))

        self.reward = np.zeros_like(self.trans_prob)

        # Number of cars in any location can take values in 
        # [0, ..., MAX_CARS] 
        tmp = self.config['MAX_CARS'] + 1

        for i in range(self.num_states):
            for j in range(self.num_actions):
                action = j - 5

                # Move Car from Second Location to First Location
                if action < 0:
                    # State = (MAX_CARS + 1)*num_car_loc1 + num_car_loc2
                    num_car_loc1 = i//tmp
                    num_car_loc2 = i%tmp

                    # Movement of car only possible when enough cars
                    # are available
                    if num_car_loc2 >= -action:

                        num_car_loc1 = min(num_car_loc1 - action, 
                            self.config['MAX_CARS'])

                        dst_state = tmp*num_car_loc1 + num_car_loc2 + action

                        for k in range(self.num_states):
                            self.reward[i, k, j], self.trans_prob[i, k, j] = self._get_ret_req_dynamics(
                                dst_state, k) 

                        # self.trans_prob[i, dst_state, j] = 1
                        self.reward[i, :, j] += (
                            self.config['COST_OF_MOVEMENT']*action)
                    
                    else:
                        for k in range(self.num_states):
                            self.reward[i, k, j], self.trans_prob[i, k, j] = self._get_ret_req_dynamics(
                                i, k)
                    

                # Do not move car
                if action == 0:
                    for k in range(self.num_states):
                        # init_num_car_loc1 = i//tmp
                        # init_num_car_loc2 = i%tmp

                        # final_num_car_loc1 = k//tmp
                        # final_num_car_loc2 = k%tmp

                        # if init_num_car_loc1 < final_num_car_loc1:
                        #     prob1 = self._get_poisson_prob(
                        #         final_num_car_loc1 - init_num_car_loc1,
                        #         self.config['LOC1_RETURN_POISSON_PARAM']
                        #     )
                        #     self.reward1 = 0
                        # elif init_num_car_loc1 > final_num_car_loc1:
                        #     prob1 = self._get_poisson_prob(
                        #         init_num_car_loc1 - final_num_car_loc1,
                        #         self.config['LOC1_REQUEST_POISSON_PARAM'] 
                        #     )
                        #     self.reward1 = 
                        # else:
                        #     prob1 = self._get_net_zero_return_request_prob(
                        #         1, init_num_car_loc1)

                        self.reward[i, k, j], self.trans_prob[i, k, j] = self._get_ret_req_dynamics(
                            i, k) 

                        

                # Move Car from First Location to Second Location
                if action > 0:
                    # Movement of car only possible when enough cars
                    # are available
                    if num_car_loc1 >= action:

                        num_car_loc2 = min(num_car_loc2 + action, 
                            self.config['MAX_CARS'])

                        dst_state = tmp*(num_car_loc1 - action) + num_car_loc2

                        for k in range(self.num_states):
                            self.reward[i, k, j], self.trans_prob[i, k, j] = self._get_ret_req_dynamics(
                                dst_state, k) 

                        # self.trans_prob[i, dst_state, j] = 1
                        self.reward[i, :, j] -= (
                            self.config['COST_OF_MOVEMENT']*action)
                    

                        # self.trans_prob[i, dst_state, j] = 1
                        # self.reward[i, dst_state, j] = (
                        #     -self.config['COST_OF_MOVEMENT']*action)


                    
                    else:
                        # self.trans_prob[i, i, j] = 1

                        for k in range(self.num_states):
                            self.reward[i, k, j], self.trans_prob[i, k, j] = self._get_ret_req_dynamics(
                                i, k) 

                        # # self.trans_prob[i, dst_state, j] = 1
                        # self.reward[i, :, j] += (
                        #     self.config['COST_OF_MOVEMENT']*action)
                    

    def _get_poisson_prob(self, n, lmb):
        key = str(n) + '_' + str(lmb)
        if key not in self._poisson_cache:
            self._poisson_cache[key] = poisson.pmf(n, lmb)
        return self._poisson_cache[key]

    # def _get_net_zero_return_request_prob(self, loc, num_cars):
    #     if loc == 1:
    #         lmb_request = self.config['LOC1_REQUEST_POISSON_PARAM']
    #         lmb_return = self.config['LOC1_RETURN_POISSON_PARAM']
        
    #     if loc == 2:
    #         lmb_request = self.config['LOC2_REQUEST_POISSON_PARAM']
    #         lmb_return = self.config['LOC2_RETURN_POISSON_PARAM']

    #     prob = 0
    #     for i in range(self.config['MAX_CARS'] - num_cars + 1):
    #         req_prob = self._get_poisson_prob(i, lmb_request)
    #         ret_prob = self._get_poisson_prob(i, lmb_return)
    #         prob += req_prob*ret_prob
        
    #     return prob

    def _get_ret_req_dynamics(self, init_state, final_state):
        # Number of cars in any location can take values in
        # [0, ..., MAX_CARS] 
        tmp = self.config['MAX_CARS'] + 1

        init_num_car_loc1 = init_state//tmp
        init_num_car_loc2 = init_state%tmp

        final_num_car_loc1 = final_state//tmp
        final_num_car_loc2 = final_state%tmp

        exp_reward = 0
        prob = 0

        for num_req_loc1 in range(self.config['MAX_CARS']+1):
            prob1 = self._get_poisson_prob(
                num_req_loc1, 
                self.config['LOC1_REQUEST_POISSON_PARAM']    
            )

            # valid rental requests should be less than actual # of cars
            num_req_loc1 = min(num_req_loc1, init_num_car_loc1)

            for num_ret_loc1 in range(self.config['MAX_CARS']+1):
                prob2 = self._get_poisson_prob(
                    num_ret_loc1,
                    self.config['LOC1_RETURN_POISSON_PARAM']
                )

                # Valid return should be such that total cars is less 
                # than MAX_CARS
                num_ret_loc1 = min(
                    num_ret_loc1,
                    self.config['MAX_CARS'] - init_num_car_loc1 + num_req_loc1 - num_ret_loc1
                )

                # If the condition is true then cannot reach the final 
                # state
                if ((final_num_car_loc1 - init_num_car_loc1) !=
                    (num_ret_loc1 - num_req_loc1)):
                    continue

                for num_req_loc2 in range(self.config['MAX_CARS']+1):
                    prob3 = self._get_poisson_prob(
                        num_req_loc2, 
                        self.config['LOC2_REQUEST_POISSON_PARAM']    
                    )

                    # valid rental requests should be less than actual # of cars
                    num_req_loc2 = min(num_req_loc2, init_num_car_loc2)

                    for num_ret_loc2 in range(self.config['MAX_CARS']+1):
                        prob4 = self._get_poisson_prob(
                            num_ret_loc2,
                            self.config['LOC2_RETURN_POISSON_PARAM']
                        )

                        # Valid return should be such that total cars is less 
                        # than MAX_CARS
                        num_ret_loc2 = min(
                            num_ret_loc2,
                            self.config['MAX_CARS'] - init_num_car_loc2 + num_req_loc2 - num_ret_loc2
                        )

                        # If the condition is true then cannot reach the
                        # final state
                        if ((final_num_car_loc2 - init_num_car_loc2) !=
                            (num_ret_loc2 - num_req_loc2)):
                            continue

                        prob += prob1*prob2*prob3*prob4
                        exp_reward += (num_req_loc1 + num_req_loc2)*self.config['RENT']

            return exp_reward, prob
                        



        # for k in range(self.num_states):
            

        #     # 
        #     if init_num_car_loc1 < final_num_car_loc1:
        #         prob1 = self._get_poisson_prob(
        #             final_num_car_loc1 - init_num_car_loc1,
        #             self.config['LOC1_RETURN_POISSON_PARAM']
        #         )
        #         self.reward1 = 0
        #     elif init_num_car_loc1 > final_num_car_loc1:
        #         prob1 = self._get_poisson_prob(
        #             init_num_car_loc1 - final_num_car_loc1,
        #             self.config['LOC1_REQUEST_POISSON_PARAM'] 
        #         )
        #         reward1 = 
        #     else:
        #         prob1 = self._get_net_zero_return_request_prob(
        #             1, init_num_car_loc1)

    def plot(history):
        for i, pol in enumerate(history['policy']):
            pol = pol.reshape(
                (self.config['MAX_CARS'] + 1, self.config['MAX_CARS'] + 1)
            )

            plt.figure()
            fig = sns.heatmap(np.flipud(policy), cmap="YlGnBu")
            plt.savefig(str(i) + '.png')