# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 15:28:00 2022

@author: amill212
"""

#HMC Basic Example 
import numpy as np 
from HMC_DA_Chain import HMC_DA_Chain
from sklearn.datasets import make_spd_matrix

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
n_total = 25000
burn_in = 5000
trajectory_length = 10
init_inbetween = np.zeros(dimensions)
delta = 0.3
cov = make_spd_matrix(dimensions)
inv_cov = np.linalg.inv(cov)

chain, acceptance_rate, step_size = HMC_DA_Chain(init, LP, LP_grad, inv_cov, M, burn_in, n_total)

chain_array = np.asarray(chain)

HMC_DA_mean = np.mean(chain_array, axis=0)
final_cov = np.cov(chain_array.T)
print('The final covariance from all the samples without leapfrog steps is: ')
print(final_cov)
print('The mean value of the the samples without leapfrog steps is: ')
print(HMC_DA_mean)