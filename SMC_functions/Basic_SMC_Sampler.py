# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 15:05:08 2022

@author: amill212
"""

from resampling import systematic_resampling
import numpy as np 
from distributions import distribution

#smc sampler algorithm built from alogrithm 2 of this paper 
#https://livrepository.liverpool.ac.uk/3003664/1/ISMA_2016.pdf

def Basic_SMC_Sampler(k, N, num_runs, p_x, q_x, mu_approx, sigma_approx=1):
    '''
    Basic SMC sampler 

    Parameters
    ----------
    k : Is the iteration of the SMC sampler which we are on 
    
    N : Number of samples we want to take at each iteration 
    
    num_runs : The number of cycles we want the sampler to run for 
    
    p_x : the pdf of the target distribution 
    
    q_x : pdf of the initial guess distribution 
    
    mu_approx : our best guess of the mean of the distribution 
    
    sigma_approx : approximate std_dev of the gaussian, optional
        The default is 1.

    Returns
    -------
    samples : Final samples from the last SMC run 
    
    weights : The final weights of the samples 
    
    means : the means from each run 

    '''
    #getting the initial samples and weights from the guess distribution 
    samples = []
    weights = []
    for i in range(N):
        x_i = np.random.normal(mu_approx, sigma_approx)
        samples.append(x_i)
        weight = p_x.pdf(x_i)/q_x.pdf(x_i)
        weights.append(weight)
    
    #storing the means so we can graph it later if need be 
    means = [mu_approx]
    for i in range(num_runs):
        print('We are currently on run ', i+1, '/', num_runs)
        
        #normalizing the weights
        weights = np.asarray(weights)
        weights = weights/(np.sum(weights))
        
        #estimating the means from the samples
        mu_approx = (np.asarray(weights).T) @ np.asarray(samples)
        means.append(mu_approx)
        
        #calculating N effective 
        N_eff = round(1/(np.sum(np.square(weights))))
        
        if N_eff < N/2:
            #print('Resampling in progress')
            #resampling if it drops below a certain threshold 
            sampling_index = systematic_resampling(weights)
            
            #selecting the samples from the indexed weights 
            new_samples = []
            for i in sampling_index:
                new_samples.append(samples[i])
                
            #resetting the weights 
            #weights = np.ones(len(weights))
            
            #resetting the lsamples as well 
            samples = new_samples 
            
        k += 1
        
        #now sample from the new proposal distribution given by mu_approx and 
        #sigma_approx
        q_x = distribution(mu_approx, sigma_approx)
        
        #now sampling from the new proposal distribution
        new_samples = []
        for i in range(N):
            x_i = np.random.normal(mu_approx, sigma_approx)
            new_samples.append(x_i)
            
        #now calculating the new weights 
        new_weights = []
        for i in range(len(new_samples)):
            weight = p_x.pdf(new_samples[i])/p_x.pdf(samples[i])
            new_weights.append(weight)
            
        #resetting everying 
        weights = new_weights
        samples = new_samples
        
    return weights, samples, means 
        

