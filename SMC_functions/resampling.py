#defining a systematic resmapling method for smc samplers 
import random as random 
import numpy as np 


def systematic_resampling(weights):
    '''
    Function which performs systematic resampling in order to 
    give back the index of weights we want to resample. 
    
    inputs:
        weights - list of particles with there corresponding weights 
        
    outputs:
        resample_index - list of indexes for the pareticles we want to 
        resample
    '''
    #checking wether the sample weights have been normalized
    #have to give some leyway due to floating point errors 
    if sum(weights) > 0.9999 and sum(weights) < 1.0001:
    
        #setting up the points in the list with a random offset 
        N = len(weights)
        step = 1/N
        point = random.uniform(0, 1)
        steps = [point]
        
        for i in range(1, N):
            steps.append(point + i*step)
        
        #sorting it to make sure all values are in order and between 1 and 0
        steps_sorted = []
        for i in steps:
            if i > 1:
                i = i-1
                steps_sorted.append(i)
            else: 
                steps_sorted.append(i)
    
        steps_sorted.sort()
        
        #getting the cumulative weights 
        cumulative_weights = np.cumsum(weights)
        
        #searching through the steps list and adding the particles which are less 
        #than a certain weight 
        resample_index = []
        i = 0
        j = 0
        while i < N:
            if steps_sorted[i] < cumulative_weights[j]:
                resample_index.append(j)
                i += 1
            else: 
                j += 1
        
        return resample_index 
    
    else: 
        raise Exception('These weights do not add up to 1, maybe you need to normalize.')
