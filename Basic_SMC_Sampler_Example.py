# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 13:18:49 2022

@author: amill212
"""

#Basic SMC sampler example script 
from distributions import distribution
from Basic_SMC_Sampler import Basic_SMC_Sampler
import matplotlib.pyplot as plt 

mu_true = 3.5
mu_approx = 2
sigma_true = 1.0
sigma_approx = 1.0

p_x = distribution(mu_true, sigma_true)

q_x = distribution(mu_approx, sigma_approx)

k = 1
N = 1000
num_runs = 100

weights, samples, means = Basic_SMC_Sampler(k, N, num_runs, p_x, q_x, mu_approx) 

plt.plot(means, label='Calculated Means')  
plt.axhline(y=mu_true, color='r', linestyle='-', label='True Mean')
plt.legend()
plt.show()