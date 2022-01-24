import numpy as np

def running_avg_update(old_avg, new_val, alpha):
    '''
        This function implements the running average update:
            new_avg = old_avg + alpha*(new_val - old_avg)
    '''
    return old_avg + alpha*(new_val - old_avg)
    