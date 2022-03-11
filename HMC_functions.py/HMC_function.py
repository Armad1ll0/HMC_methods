import numpy as np 
from update_mass import update_mass 
from leapfrog import leapfrog 
#HMC sampler 

def HMC(x_old, NLP, NLP_grad, step_size, trajectory_length, inv_cov, M, old_inbetween):
    '''
    inputs:
    x_old is the previous sample, if it is the first time round, we set it as the initial point 
    
    NLP is the negative log probability 
    
    BLP_grad is the gradient of the negative log probability 
    
    step_size, trajectory length, NLP, NLP_grad, inv_cov and M are explained in the leapfrog.py 
    file 
    
    old_inbetween are the in between samples from the previous HMC step, it is an empty list 
    when we first do it 
    
    output: 
    if the sample is accepted then we output the new position (x_new) and inbetwen samples 
    (new_inbetween). If sample is rejected we just return the old sample position and inbetween
    steps. 
    '''
    
    def K(p, M):
        '''
        standard kinetic energy equations in linear algebra form 
        
        '''
        
        pT = p.T
        M_1 = np.linalg.inv(M)
        K = 0.5*(pT @ M_1 @ p)
        return K

    def U(x, A):
        '''
        potential energy is just based on the NLP of the particles current position
        '''
        
        U = NLP(x, inv_cov)
        return U

    def E(x, p, inv_cov, M):
        '''
        total energy is the sum of the kinetic and potential energy 
        '''
        
        return K(p, M) + U(x, inv_cov)

    def log_r(x_new, p_new, x_old, p_old):
        '''
        to make things easier, we calculate the difference in energy from the logs of 
        exponents
        '''
        
        return -(E(x_new, p_new, inv_cov, M) - E(x_old, p_old, inv_cov, M))

    p_old = []
    for i in range(len(x_old)):
        p_element = np.random.normal(loc=0, scale=M[i][i])
        p_old.append(p_element)
        
    p_old = np.asarray(p_old)

    x_new, p_new, new_inbetween = leapfrog(step_size, trajectory_length, p_old.copy(), x_old.copy(), NLP_grad, inv_cov, M)

    accept = np.log(np.random.random()) < log_r(x_new, p_new, x_old, p_old)

    if accept:
        return accept, x_new, new_inbetween
    
    else: 
        return accept, x_old, old_inbetween
    
