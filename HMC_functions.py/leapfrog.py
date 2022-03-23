#Basic leapfrog algorithm 
import numpy as np 

def leapfrog(step_size, trajectory_length, p, x, NLP_grad, inv_cov, M):
    '''
    inputs: 
    trajector_length is how long we want our hamiltonian dynamics to run for 
    
    step_size is the size of the step we want to do in between each trajectory length, 
    e.g. if we have a trajectory length of 1 and a step size of 0.1, we will have 10 steps 
    to do out full trajectory length 
    
    p is the gaussian picked momentum 
    
    x is our current position
    
    NLP_grad is the gradient of the negative log probability 
    
    inv_cov is the inverse of our covariance 
    
    M is the mass matrix, if not doing an adaptive mass matrix part then this is just an 
    identity matrix 
    
    outputs:
    x, p are the new momentums and positions calculated via the leapfrog algorithm 
    
    in_between is the positions in between each leapfrog step, we take these as they can 
    be used for extra samples when trying to estimate the covariance or mass matrix
    '''
    
    p = p - 0.5*step_size*(M @ NLP_grad(x, inv_cov))
    in_between = []
    for i in range(int(trajectory_length) - 1):
        x = x + step_size*(p @ np.linalg.inv(M))
        in_between.append(x)
        p = p - 0.5*step_size*(M @ NLP_grad(x, inv_cov))
    x = x + step_size*(p @ np.linalg.inv(M))
    p = p - 0.5*step_size*(M @ NLP_grad(x, inv_cov))
    
    return x, p, in_between 

def leapfrog_alt(step_size, p, x, LP_grad, inv_cov, M):
    '''
    same function as the one before but without the trajectory length. We use log prob 
    and not neg log prob to make things simpler. 
    
    '''
    #need to put in the LP_grad evals to speed things up 
    p = p + 0.5*step_size*(M @ LP_grad(x, inv_cov))
    x = x + step_size*(p @ np.linalg.inv(M))
    p = p + 0.5*step_size*(M @ LP_grad(x, inv_cov))
    
    return x, p
