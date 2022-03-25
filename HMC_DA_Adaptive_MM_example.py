# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 15:28:00 2022

@author: amill212
"""

#HMC Basic Example 
import numpy as np 
from HMC_DA_Chain_Adaptive_MM import HMC_DA_MM_Chain
from sklearn.datasets import make_spd_matrix
import matplotlib.pyplot as plt 

def LP(x, inv_cov):
    xT = x.T
    return -0.5*(xT @ inv_cov @ x)

def LP_grad(x, inv_cov):
    return -inv_cov @ x

dimensions = 5
theory_cov = np.eye(dimensions) 
theory_samples = 100
M = np.eye(dimensions)
init = np.zeros(dimensions) 
n_total = 2000
burn_in = 1000
trajectory_length = 10
init_inbetween = np.zeros(dimensions)
delta = 0.7
five_d = np.array([[1.15417, -1.01342, 0.494256, 0.738341, 1.54744],
                   [-1.01342, 1.18772, -0.572723, -0.863335, -1.34392],
                    [0.494256, -0.572723, 1.03581, 0.363349, 0.656954],
                    [0.738341, -0.863335, 0.363349, 1.66525, 1.23275],
                    [1.54744, -1.34392, 0.656954, 1.23275, 2.91554]])
cov = five_d
#cov = np.eye(dimensions)
#cov = make_spd_matrix(dimensions)
inv_cov = np.linalg.inv(cov)

#%%
chain, acceptance_rate, step_size, results, similarities = HMC_DA_MM_Chain(init, LP, LP_grad, inv_cov, M, burn_in, n_total, theory_cov, theory_samples)

#%%

chain_array = np.asarray(chain)

HMC_DA_mean = np.mean(chain_array, axis=0)
all_vals_mean = np.mean(results, axis = 0)
final_cov = np.cov(chain_array.T)
all_vals_cov = np.cov(results.T)
print('The final covariance from all the samples without leapfrog steps is: ')
print(final_cov)
print('The final covariance from all the samples with leapfrog steps is: ')
print(all_vals_cov)
print('The mean value of the samples without leapfrog steps is: ')
print(HMC_DA_mean)
print('The mean value of the samples with leapfrog steps is: ')
print(all_vals_mean)


