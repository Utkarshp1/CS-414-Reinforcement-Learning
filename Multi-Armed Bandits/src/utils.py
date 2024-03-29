import numpy as np

def running_avg_update(old_avg, new_val, alpha):
    '''
        This function implements the running average update:
            new_avg = old_avg + alpha*(new_val - old_avg)
    '''
    return old_avg + alpha*(new_val - old_avg)
    
def create_inverse_schedule(const):
    def inverse_schedule(time):
        '''
            This function implements the inverse schedule function:
                new_val = const/time
        '''
        return const/time
        
    inverse_schedule.config = {"C": const}
    
    return inverse_schedule
    
def softmax(x):
    '''
        This function accepts a 1-D NumPy array and calculates 
        element-wise softmax of the function.
    '''
    exp_arr = np.exp(x)
    return exp_arr/np.sum(exp_arr)
    