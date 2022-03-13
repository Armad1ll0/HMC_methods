from MCMC_sub_functions import sample_MH
from tqdm import tqdm

def MCMC_chain(init, step_size, n_total, log_prob, inv_cov):
    '''
    inputs:
    init is the initial starting point for the sampler 
    
    step_size is how far we are allowed to sample away from our current sample 
    
    n_total is the number of total samples we want to return 
    
    log_prob is the log probability of the pdf we are sampling from 
    
    outputs:
    chain is the chain of sample after the algorithm has run for n_total iterations 
    
    acceptance rate is the amount of new samples that are accepted
    
    '''
    n_accepted = 0
    chain = [init]
    for i in tqdm(range(n_total)):
        accept, state = sample_MH(chain[-1], log_prob, step_size, inv_cov)
        chain.append(state)
        n_accepted += accept
    acceptance_rate = n_accepted/float(n_total)
    
    return chain, acceptance_rate 
