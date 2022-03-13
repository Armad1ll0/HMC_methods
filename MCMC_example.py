import MCMC_chain as MCMC_chain 
from MCMC_sub_functions import proposed, p_acc_MH, sample_MH
from sklearn.datasets import make_spd_matrix
import numpy as np 
import matplotlib.pyplot as plt 

dimensions = 2
init = np.zeros(dimensions)
inv_cov = make_spd_matrix(dimensions)
step_size = 1
n_total = 10000

def gaussian(x, inv_cov):
    xT = x.T
    first = np.dot(xT, inv_cov)
    return np.exp(-0.5*np.dot(first, x))

def log_prob(x, inv_cov):
    xT = x.T
    first = np.dot(xT, inv_cov)
    return -0.5*np.dot(first, x)
  
MCMC_chain, MCMC_acceptance_rate = MCMC_chain.MCMC_chain(init, step_size, n_total, log_prob, inv_cov)
MCMC_results = np.asarray(MCMC_chain)

#plot the scatter results
plt.scatter(MCMC_results.T[0, :], MCMC_results.T[1, :], color='b', s=0.1)
plt.title('MCMC Samples')
plt.show()
