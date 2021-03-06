#defining an initial step size function for HMC 
from leapfrog import leapfrog_alt
import numpy as np 

def initial_step_size(x, NLP, LP_grad, inv_cov, M):
    '''
    Getting the initial epsilon by using Metropolis-adjusted Langevin 
    algorithm/Langevin Proposal mathematics. 
    
    Parameters
    ----------
    x : the initial position of sampler
    
    NLP : negative log prob 
    
    NLP_grad : gradient of the negative log prob 
    
    inv_cov : our inverse covariance, may have to be calculated by autograd 
    tools later on 
    
    M : Mass Matrix 

    Returns
    -------
    step_size : our initial step size to start the burn in samples from 

    '''
    
    step_size = 1 
    parems = len(x)
    p = []
    for i in range(parems):
        p_element = np.random.normal(loc=0, scale=1)
        p.append(p_element)
    p = np.asarray(p)
    
    x_new, p_new = leapfrog_alt(step_size, p, x, LP_grad, inv_cov, M)
    
    def K(p, M):
        pT = p.T
        M_1 = np.linalg.inv(M)
        K = 0.5*(pT @ M_1 @ p)
        return K

    def U(x, inv_cov):
        U = NLP(x, inv_cov)
        return U

    def E(x, p, inv_cov, M):
        return K(p, M) + U(x, inv_cov)

    def p_acc_func(x_new, p_new, x_old, p_old):
        return np.exp(-(E(x_new, p_new, inv_cov, M) - E(x_old, p_old, inv_cov, M)))
    
    p_acc = p_acc_func(x_new, p_new, x, p)
    
    a = 2*int(p_acc > 0.5) - 1
    
    while p_acc**a > 2**(-a):
        p = []
        for i in range(parems):
            p_element = np.random.normal(loc=0, scale=1)
            p.append(p_element)
        p = np.asarray(p)
        step_size = step_size*(2**(a))
        x_new, p_new = leapfrog_alt(step_size, p, x, LP_grad, inv_cov, M)
        p_acc = p_acc_func(x_new, p_new, x, p)
        alpha = min(1, p_acc)
        x = x_new
        p = p_new

    return step_size, alpha
