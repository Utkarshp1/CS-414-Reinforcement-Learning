import numpy as np
import matplotlib.pyplot as plt

from Experiment import Experiment
from algorithms import Greedy
from utils import create_inverse_schedule
from metrics import MeanCummulativeRegret

algo = {"Greedy": (Greedy, {}),
       "$\epsilon$-Greedy": (Greedy, {"eps": 0.1}),
       "Variable$\epsilon$-Greedy": (Greedy, {"eps_schedule": create_inverse_schedule(0.1)})}

experiment = Experiment(algo, "Greedy", metrics=[MeanCummulativeRegret()])

experiment.simulate()