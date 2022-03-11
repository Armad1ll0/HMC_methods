#importing dependencies 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.datasets import make_spd_matrix
from HMC_chain import HMC_chain
from matplotlib.patches import Ellipse
import time as time 


start = time.time()
dimensions = 2 #number of dimensions of our model 
theory_cov = np.eye(dimensions) #theory cov and theory samples are needed for the mass matrix 
theory_samples = 100 #how much the theoretical covariance is weighted by 
M = np.eye(dimensions)
init = np.zeros(dimensions) #for ease we will set the initial point at the origin 
n_total = 10000 #total number of liunks in our chain 
step_size = 0.05 #step_size for the leapfrog integrator 
trajectory_length = 10 #trajectory length for leapfrog integrator 
init_inbetween = np.zeros(dimensions) #initial inbetween steps is empty 
actual_cov = np.eye(dimensions) #setting the initial covariance matrix guess 
#REMEMBER WE WILL BE PULLING SAMPELS FROM THE INVERSE OF THIS MATRIX 
inv_cov = make_spd_matrix(dimensions) #creating the symmetric matrix which we want to sample from 

print('This is our original covariance matrix:')
print(np.linalg.inv(inv_cov))

#functions needed for our NLP and the grad of it 
def NLP(x, inv_cov):
    xT = x.T
    return 0.5*(xT @ inv_cov @ x)

def NLP_grad(x, inv_cov):
    return inv_cov @ x
  
HMC_results, acceptance, all_vals, M, actual_cov, similarity = HMC_chain(init, step_size, trajectory_length, n_total, NLP, NLP_grad, inv_cov, M, init_inbetween, theory_cov, theory_samples)
HMC_results_array = np.asarray(HMC_results)
all_vals_array = np.asarray(all_vals)

#This cell is only suitable for 2 dimensions. 
nstd = 3
ax = plt.subplot(111)

#Covariance ellipse of what we are drawing from 
#REMEMBER, IN THE PROBABILITY DENSITY FUNCTION, A = Sigma^-1, SO WE ARE ACTUALLY DRAWING 
#FROM THE INVERSE OF a AND NOT a ITSELF WHEN WE TAKE SAMPLES 
#This took me far to long to figure out. 
sigma = np.linalg.inv(inv_cov)
vals, vecs = np.linalg.eigh(sigma)
theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
w, h = 2 * nstd * np.sqrt(vals)
ell1 = Ellipse(xy=(0, 0),
              width=w, height=h,
              angle=theta, color='black')
ell1.set_facecolor('none')
ax.add_artist(ell1)

#Covariance ellipse from what we have calculated 
cov = actual_cov
vals, vecs = np.linalg.eigh(cov)
theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
w, h = 2 * nstd * np.sqrt(vals)
ell2 = Ellipse(xy=(0, 0),
              width=w, height=h,
              angle=theta, color='blue')
ell2.set_facecolor('none')
ax.add_artist(ell2)

plt.scatter(HMC_results_array[:, 0], HMC_results_array[:, 1], color='r', s=0.1)
plt.scatter(all_vals_array[:, 0], all_vals_array[:, 1], color='g', s=0.1)
plt.show()

#The following graph plots the similarity between the real covariance of A^-1 and our calculated
#covariance from the samples. 
count = []
for i in range(len(similarity)):
    count.append(i*2)
plt.plot(count, similarity)
plt.xlabel('Iteration')
plt.ylabel('Euclidean Similarity')
plt.title('Euclidean Similarity Between Covariance from Samples and Real Covariance')
plt.show()

print('This is our final covariance from the samples that have been accepted:')
print(np.linalg.inv(actual_cov))

#Printing the mean of the samples obtained as well 
all_vals_mean = np.mean(all_vals_array, axis=0)
HMC_vals_mean = np.mean(HMC_results_array, axis=0)
print(all_vals_mean)
print(HMC_vals_mean)

#NOTE: I need to chnage the mass matrix so that it only appends the covariances of the new samples 
#as otherwise it will slow this process down 
