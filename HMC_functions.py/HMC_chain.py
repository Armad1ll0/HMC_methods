from leapfrog import leapfrog 
from update_mass import update_mass
from HMC_function import HMC
import numpy as np 
from tqdm import tqdm

def HMC_chain(init, step_size, trajectory_length, n_total, NLP, NLP_grad, inv_cov, M, init_inbetween, theory_cov, theory_samples):
    '''
    inputs: 
    step_size, trajectory_length, NLP, NLP_grad, inv_cov and M are explained in leapfrog.py
    
    n_total is the total number of samples you want the chain to return 
    
    init_inbetween is the initial number of in between samples which will always just be an array of zeros 
    
    init is the first point you want to start the sampler off with 
    
    output: 
    chain is the chain of samples that have bveen accepted 
    
    acceptance_rate is the amount of new samples accepted 
    
    all_vals returns the chain samples and the inbetween samples as well 
    
    M is the mass matrix 
    
    actual_cov is what the covariance we have calculated from our samples is 
    
    similarity is the euclidean distance measurement between our actual_cov calculated from the samples 
    and the covariance/target posterior we generate initially. 
    
    '''
    chain = [init]
    steps_in_between = [init_inbetween]
    n_accepted = 0
    all_states = []
    similarity = []

    for j in tqdm(range(n_total)):
        accept, state, in_between = HMC(chain[-1], NLP, NLP_grad, step_size, trajectory_length, inv_cov, M, steps_in_between[-1])

        if accept == True:
            chain.append(state)
            all_states.append(state)
            steps_in_between.append(in_between)
            for a in in_between:
              all_states.append(a)
            n_accepted += 1

        if j > 0 and j % 2 == 0:
            all_vals = np.asarray(all_states)
            M, actual_cov = update_mass(theory_cov, theory_samples, all_vals)
            euclid_dist = np.linalg.norm(np.linalg.inv(inv_cov) - actual_cov)
            similarity.append(euclid_dist)
            
        if j > 0 and j%1000 == 0:
            print('We are currently on iteration: ', j)
            print('Our current estimated mass matrix is: ')
            print(M)
            
    acceptance_rate = n_accepted/float(n_total)
    
    return chain, acceptance_rate, all_vals, M, actual_cov, similarity

