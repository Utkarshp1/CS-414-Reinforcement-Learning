import numpy as np

from utils import generate_MDP
from MDP import MDP
from PolicyIteration import PolicyIteration
from ValueIteration import ValueIteration
from JackCarRentalProblem import JackCarRentalProblem

goal_coordinates = {'positive': (3, 0), 'negative':(3, 1)}
rewards = {'general': 0.0,
            'positive': 1,
            'negative': -100}
            
prob = {'success_prob': 0.8,
        'left_prob': 0.1,
        'right_prob': 0.1}
        
wall_coor = (1, 1)

# trans_prob, reward = generate_MDP(4, 3, goal_coordinates, rewards, prob, wall_coor)

# for i in range(4):
#       print(trans_prob[:, :, i])
#       print()
#       print(reward[:, :, i])
#       print()

# print(trans_prob[:, :, 0])
# print()
# print(reward[:, :, 0])
# print()
# print(trans_prob[:, :, 1])
# print()
# print(trans_prob[:, :, 2])
# print()
# print(trans_prob[:, :, 3])
# print()
# print(reward)

# grid_world = MDP(trans_prob, reward, 0.9)
# grid_world.value_iteration(1e-10)
# print(grid_world.best_policy)
# print(grid_world.best_value_function)
# print('-'*10)
# grid_world.policy_iteration(1e-10)
# print(grid_world.opt_policy)
# print(grid_world.opt_val_func)

# pol_iter = PolicyIteration(grid_world, 1e-10)
# pol_iter.run()
# print(pol_iter.opt_policy)
# print(pol_iter.opt_val_func)

# val_iter = ValueIteration(grid_world, 1e-10)
# val_iter.run()
# print(val_iter.opt_policy)
# print(val_iter.opt_val_func)

jack = JackCarRentalProblem()
pol_iter = PolicyIteration(jack, 1e-10)
pol_iter.run()
jack.plot(pol_iter.history)
# print(pol_iter.opt_policy)
# print(pol_iter.opt_val_func)
