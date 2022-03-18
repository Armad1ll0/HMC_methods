from initial_step_size import initial_step_size 
import numpy as np 
from leapfrog import leapfrog
from tqdm import tqdm
from update_mass import update_mass

def HMC_Dual_Averaging(init, delta, burn_in, n_total, trajectory_length, NLP, NLP_grad, inv_cov, M, init_inbetween, theory_cov, theory_samples, seed=48):
    '''
    Dual Averaging HMC sampler, I am having problems witrh the step size 
    getting to small which is weird. Sometimes works so need to go through 
    it a bit more. 

    Parameters
    ----------
    init : initial starting position of the sampler 
    
    delta : our ideal acceptance ratio
    
    burn_in : number of samples we use to estimate epsilon 
    
    n_total : total number of samples 
    
    trajectory_length : trajectory length 
    
    NLP : negative log prob function 
    
    NLP_grad : negative log probability gradient function 
    
    inv_cov : inverse of our covariance 
    
    M : Mass matrix 
    
    init_inbetween : initial inbetween steps, normally just a placeholder 
    so we can store all the values from the leapfrog integrator later 
    
    theory_cov : theoretical covariance matrix 
    
    theory_samples : sample weighting for theoretical cov 
    
    seed : optional number for random number generator, useful if we want 
    to parallelize the process. 
        DESCRIPTION. The default is 48.

    Returns
    -------
    chain: chain of samples
    
    all_states: all samples including inbetween leapfrog steps 
    
    acceptance_rate: ratio of accepted moves to total samples 
    
    alphas: list of p_acc values 
    
    M: final mass matrix 
    
    step_size: final step size calculated 
    
    trajectory_length: final trajectory length 

    '''
    step_size_0, alpha = initial_step_size(init, trajectory_length, NLP, NLP_grad, inv_cov, M)
    mu = np.log(10*step_size_0)
    step_size = step_size_0

    step_size_bar = 1
    H = 0
    gamma = 0.05
    t_0 = 10
    k = 0.75
    lambda_ = 1
    trajectory_length = round(lambda_/step_size)
    np.random.seed(seed)
    chain = [init]
    steps_in_between = [init_inbetween]
    n_accepted = 0
    alphas = [alpha]
    all_states = []
    steps_in_between = [init_inbetween]

    for j in tqdm(range(1, n_total)):

        parems = len(init)
        p = []
        for i in range(parems):
            p_element = np.random.normal(loc=0, scale=1)
            p.append(p_element)
        p = np.asarray(p)
        
        trajectory_length = max(1, round(lambda_/step_size)) 
        x_new, p_new, in_between = leapfrog(step_size, trajectory_length, p, chain[-1], NLP_grad, inv_cov, M)
        
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
        
        p_acc = min(1, p_acc_func(x_new, p_new, chain[-1], p))
        
        accept = np.random.random() <= p_acc
        

        if accept == True:
            alphas.append(p_acc)
            chain.append(x_new)
            all_states.append(x_new)
            steps_in_between.append(in_between)
            for a in in_between:
              all_states.append(a)
            
        if j < burn_in:

            H = (1 - 1/(j+t_0))*H + (1/(j+t_0)) * (delta - p_acc)
            step_size = np.exp(mu - np.sqrt(j)/gamma * H)
            step_size_bar = np.exp(j**(-k) * np.log(step_size) + (1-j**(-k))*np.log(step_size_bar))

        else:
            step_size = step_size_bar
        
        if j > 0 and j % 2 == 0:
            chain_array = np.asarray(chain)
            all_vals = np.asarray(all_states)
            M, actual_cov = update_mass(theory_cov, theory_samples, all_vals)
            
    acceptance_rate = len(chain)/n_total
    return chain, all_states, acceptance_rate, alphas, M, step_size, trajectory_length
