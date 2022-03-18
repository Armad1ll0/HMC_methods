# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 15:28:00 2022

@author: amill212
"""

#HMC Basic Example 
import numpy as np 
from HMC_chain_DA import HMC_Dual_Averaging
from sklearn.datasets import make_spd_matrix

def NLP(x, inv_cov):
    xT = x.T
    return 0.5*(xT @ inv_cov @ x)

def NLP_grad(x, inv_cov):
    return inv_cov @ x

dimensions = 2
theory_cov = np.eye(dimensions) 
theory_samples = 100
M = np.eye(dimensions)
init = np.zeros(dimensions) 
n_total = 2000
burn_in = 200
trajectory_length = 10
init_inbetween = np.zeros(dimensions)
delta = 0.6
cov = make_spd_matrix(dimensions)
inv_cov = np.linalg.inv(cov)

chain, all_states, acceptance_rate, alphas, M, step_size, trajectory_length = HMC_Dual_Averaging(init, delta, burn_in, n_total, trajectory_length, NLP, NLP_grad, inv_cov, M, init_inbetween, theory_cov, theory_samples)

all_vals_array = np.asarray(all_states)
chain_array = np.asarray(chain)
all_vals_mean = np.mean(all_vals_array, axis=0)
HMC_DA_mean = np.mean(chain_array, axis=0)
final_cov = np.cov(all_vals_array.T)
print('The final covariance from all the samples and leapfrog steps is: ')
print(final_cov)
print('The mean value of the the samples including the leapfrog steps: ')
print(all_vals_mean)
print('The mean value of the the samples without leapfrog steps is: ')
print(HMC_DA_mean)