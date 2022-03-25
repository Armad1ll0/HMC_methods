# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 12:32:51 2022

@author: amill212
"""

from initial_step_size import initial_step_size
import numpy as np 
from leapfrog import leapfrog_alt 
from tqdm import tqdm
from update_mass import update_mass

def HMC_DA_MM_Chain(init, LP, LP_grad, inv_cov, M, burn_in, num_samples, theory_cov, theory_samples):
    '''
    HMC with Dual averaging function based on the algorithm 5 given in 
    http://www.stat.columbia.edu/~gelman/research/published/nuts.pdf

    Parameters
    ----------
    init : initial position 

    LP : log probability function 
    
    LP_grad : gradient of log probability 
    
    inv_cov : inverse of the covariance 
    
    M : Mass matrix, next version will contain an adaptive MM function but
    this one is just the identity 
    
    burn_in : warm up/burn in samples we have to estimate step size 
    
    num_samples : number of total samples including burn in 

    Returns
    -------
    chain : chain of accepted states 
    
    acceptance_rate : acceptance rate of the samples 
    
    step_size : the final step_size calculated by the dual averaging 
    functions after burn in

    '''
    #initial step_size
    step_size_0, alpha = initial_step_size(init, LP, LP_grad, inv_cov, M)
    step_size = step_size_0
    
    #set constants
    parems = len(init)
    mu = np.log(10*step_size_0)
    step_size_bar = 1
    H_bar = 0
    gamma = 0.05
    t_0 = 10
    k = 0.75
    delta = 0.7
    x_old = init 
    lambda_ = 1
    
    #setting up the lists 
    chain = [init]
    n_accepted = 0
    all_vals = []
    similarities = []
    
    #then for each sample 
    for i in tqdm(range(1, num_samples+1)):
        
        #resampling the momentum from a Gaussian 
        p_old = []
        for j in range(parems):
            p_element = np.random.normal(loc=0, scale=1/np.sqrt(M[j][j]))
            p_old.append(p_element)
            
        p_old = np.asarray(p_old)
        
        x = x_old
        p = p_old
        
        #doing the leapfrog steps 
        trajectory_length = max(1, round(lambda_/step_size))
        leapfrog_steps = []
        for j in range(trajectory_length):
            leapfrog_steps.append(x)
            x, p = leapfrog_alt(step_size, p, x, LP_grad, inv_cov, M)
            leapfrog_steps.append(x)

        #calculating whether I should accept the move or not using MH 
        M_inv = np.linalg.inv(M)
        alpha = min(1, np.exp(LP(x, inv_cov) - (0.5 * p.T @ M_inv @ p) - (LP(x_old, inv_cov) - (0.5 * p_old.T @ M_inv @ p_old))))
        accept = np.random.random() < alpha
        
        
        if accept == True:
            chain.append(x)
            all_vals.append(leapfrog_steps)
            x_old = x
            n_accepted += 1
        else: 
            chain.append(x_old)
            
        #Dual averaging equations
        if i < burn_in:
            H_bar = (1 - 1/(i + t_0))*H_bar + (1/(i + t_0))*(delta - alpha)
            step_size = np.exp(mu - np.sqrt(i)*H_bar/gamma)
            step_size_bar = np.exp(i**(-k)*np.log(step_size) + (1- i**(-k))*np.log(step_size_bar))
        
        else: 
            step_size = step_size_bar
            
        #adding in the adaptive mass matrix function 
        if i >= burn_in and i%100 == 0:
            results = np.concatenate(all_vals, axis=0)
            M, results_cov = update_mass(theory_cov, theory_samples, results)
            
        #creating the similarity metric
        if i > 2: 
            chain_cov = np.cov(np.asarray(chain).T)
            similarity = chain_cov - np.linalg.inv(inv_cov)
            similarity = similarity**2
            similarity = similarity.sum()
            similarities.append(similarity)
            
    acceptance_rate = n_accepted/len(chain)
    return chain, acceptance_rate, step_size, results, similarities
        