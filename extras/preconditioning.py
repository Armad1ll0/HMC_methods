import matplotlib.pyplot as plt 
import numpy as np 

def preconditioning(samples):
    '''
    Transforming the results so we get a diagonalized covariance matrix. 
    
    inputs: 
    samples is the samples generated from a previous sampling method 
    
    outputs:
    transformed_cov is the diagonalized covariance matrix of the transformed samples 
    
    transformed_results is the samples after they have been rotated and diagonalized 
    
    transformed_scatter is a scatter plot of the transformed samples
    '''
    cov = np.cov(samples.T)

    vals, vecs = np.linalg.eigh(cov)
    
    #standardizing the samples 
    mean = np.mean(samples, axis=0)
    std = np.std(samples, axis=0)
    standard_samples = (samples - mean)/std
    
    standard_cov = np.cov(standard_samples.T)
    vals, vecs = np.linalg.eigh(standard_cov)
    transformed_results = np.dot(vecs.T, standard_samples.T)
    
    #plot the scatter results again 
    plt.scatter(transformed_results[0, :], transformed_results[1, :], color='r', s=0.1)
    plt.title('Rotated and Centred Samples')
    transformed_scatter = plt.show()

    #find the new covariance matrix 
    transformed_cov = np.cov(transformed_results)
    print('This is the covariance matrix of our transformed dataset')
    print(transformed_cov)
    
    return transformed_cov, transformed_results, transformed_scatter
