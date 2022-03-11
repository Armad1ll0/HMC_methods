#Updating the mass matrix based on the samples 
import numpy as np 

def update_mass(theory_cov, theory_samples, results):
    '''
    inputs: 
    theory_cov is the theoretical mass matrix we may have estimated from previous knowledge, 
    normally though it is just set as the identity matrix 
    
    theory_samples is the weighting we give to the theoretical prior covariance 
    
    results is the current samples we have from the HMC sampler 
    
    outputs:
    actual_cov is the actual covariance we have calculated from our HMC samples 
    
    new_M is our new mass matrix based on the theoretical covariance and the actual 
    covariance 
    
    '''
    
    actual_samples = len(results)
    if len(results) == 1 or len(results) == 0:
        new_M = theory_cov
        
    else:
        actual_cov = np.cov(results.T)
        #and new we weight the 2 seperate covariances to create a new one 
        new_M = (theory_samples*theory_cov + actual_samples*actual_cov)/(actual_samples + theory_samples)
        
    new_M = np.linalg.inv(new_M)
    
    return new_M, actual_cov
