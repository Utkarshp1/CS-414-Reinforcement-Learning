'''
    Jack's Car Rental Problem Ex. 4.2 from Reinforcement Learning: An
    Introduction by Richard S. Sutton and Andrew G. Barto. 2nd Edition.
    2018. 
'''

import yaml
import numpy as np
from scipy.stats import poisson

class JackCarRentalProblem:
    def __init__(self):
        self._get_config()

        self.num_actions = 2*self.config['MAX_CARS_MOVEMENT'] + 1
        self.num_states = (self.config['MAX_CARS'] + 1)**2

        self._poisson_cache = {}

    def _get_config(self):
        with open('config.yaml') as config_file:
            self.config = yaml.safe_load(config_file)

    def _get_transition_dynamics(self):
        trans_prob = np.zeros((
            self.num_states,
            self.num_states,
            self.num_actions
        ))

        reward = np.zeros_like(trans_prob)

        # Number of cars in any location can take values in 
        # [1, ..., MAX_CARS+1] 
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

                        trans_prob[i, dst_state, j] = 1
                        reward[i, dst_state, j] = (
                            self.config['COST_OF_MOVEMENT']*action)
                    
                    else:
                        trans_prob[i, i, j] = 1

                # Do not move car
                if action == 0:
                    for k in range(self.num_states):
                        init_num_car_loc1 = i//tmp
                        init_num_car_loc2 = i%tmp

                        final_num_car_loc1 = k//tmp
                        final_num_car_loc2 = k%tmp

                        if init_num_car_loc1 < final_num_car_loc1:
                            prob1 = self._get_poisson_prob(
                                final_num_car_loc1 - init_num_car_loc1,
                                self.config['LOC1_RETURN_POISSON_PARAM']
                            )
                            reward1 = 0
                        elif init_num_car_loc1 > final_num_car_loc1:
                            prob1 = self._get_poisson_prob(
                                init_num_car_loc1 - final_num_car_loc1,
                                self.config['LOC1_REQUEST_POISSON_PARAM'] 
                            )
                            reward1 = 
                        else:
                            prob1 = self._get_net_zero_return_request_prob(
                                1, init_num_car_loc1)

                        

                # Move Car from First Location to Second Location
                if action > 0:
                    # Movement of car only possible when enough cars
                    # are available
                    if num_car_loc1 >= action:

                        num_car_loc2 = min(num_car_loc2 + action, 
                            self.config['MAX_CARS'])

                        dst_state = tmp*(num_car_loc1 - action) + num_car_loc2

                        trans_prob[i, dst_state, j] = 1
                        reward[i, dst_state, j] = (
                            -self.config['COST_OF_MOVEMENT']*action)
                    
                    else:
                        trans_prob[i, i, j] = 1

    def _get_poisson_prob(self, n, lmb):
        key = str(n) + '_' + str(lmb)
        if key not in self._poisson_cache:
            self._poisson_cache[key] = poisson.pmf(n, lmb)
        return self._poisson_cache[key]

    def _get_net_zero_return_request_prob(self, loc, num_cars):
        if loc == 1:
            lmb_request = self.config['LOC1_REQUEST_POISSON_PARAM']
            lmb_return = self.config['LOC1_RETURN_POISSON_PARAM']
        
        if loc == 2:
            lmb_request = self.config['LOC2_REQUEST_POISSON_PARAM']
            lmb_return = self.config['LOC2_RETURN_POISSON_PARAM']

        prob = 0
        for i in range(self.config['MAX_CARS'] - num_cars + 1):
            req_prob = self._get_poisson_prob(i, lmb_request)
            ret_prob = self._get_poisson_prob(i, lmb_return)
            prob += req_prob*ret_prob
        
        return prob