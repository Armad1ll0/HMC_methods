import numpy as np 

def proposed(x, step_size):
    '''
    Proposed gives the new proposed sample that we need to decide whether to accept or reject 
    later 
    
    inputs:
    x is the current position of the most recent sample 
    
    step_size is how far from the current position our sample can be taken from 
    
    outputs:
    x_new is the new sample proposed 
    '''
    
    x_new = np.random.uniform(low=x - 0.5*step_size, high = x + 0.5*step_size, size = x.shape)

    return x_new 

def p_acc_MH(x_new, x_old, log_prob):
    '''
    Metropolis hastings energy calculation 
    
    inputs:
    x_new is the proposed sample 
    
    x_old is the previous sample
    
    log_prob is the log probability of the pdf we are sampling from 
    
    outputs:
    returns the minimum value of 1 or the exponential of the energy difference, 
    whichever one is smaller 
    '''
    
    return min(1, np.exp(log_prob(x_new, A) - log_prob(x_old, A)))

def sample_MH(x_old, log_prob, step_size):
    '''
    Metropolis Hastings acceptance criteria 
    
    inputs: 
    see above functions for explanation 
    
    outputs:
    return x_new if the value from p_acc is smaller than the random number chosen, 
    else return the previous sample 
    '''
    
    x_new = proposed(x_old, step_size)
    num = np.random.random()
    accept = num < p_acc_MH(x_new, x_old, log_prob)
    if accept:
        return accept, x_new 
      
    else: 
        return accept, x_old 
