import multiprocessing as mp 
from HMC_chain import HMC_chain

num_cores = mp.cpu_count()

def set_up(init, step_size, trajectory_length, n_total, NLP, NLP_grad, inv_cov, M, init_inbetween, theory_cov, theory_samples, num_cores):
    '''
    function that sets up the inputs needed fr the hmc_chain function
    
    inputs:
    the same as we would need for HMC_chain, see that code for more detail 
    
    outputs:
    chain_samples divides the number of total samples we want by the number 
    cpu cores our machine has 
    
    seed is a random number seed so we dont pick the same random points when 
    choosing the momentums 
    
    every other input is the same as we would have for HMC_chain 
    '''
    
    chain_samples = n_total/num_cores
    seed = np.random.randint(0, 100)
    
    return init, step_size, trajectory_length, n_total, NLP, NLP_grad, inv_cov, M, init_inbetween, theory_cov, theory_samples, seed

def input_list_creator(num_cores):
    '''
    generates a list of inputs for the number of cpu cores we have available 
    
    inputs: 
    num_cores is the number of cpu cores we have available 
    
    outputs
    input list gives the list needed for the star map function in the 
    parallelize function below. Each cpu core requires an input and the
    seed is different for each one 
    '''
    
    input_list = [set_up(init, step_size, trajectory_length, n_total, NLP, NLP_grad, inv_cov, M, init_inbetween, theory_cov, theory_samples, num_cores) for i in range(num_cores)]
    
    return input_list

def parallelize_HMC(num_cores):
    '''
    parallelizes the HMC code into multiple chains to speed up the process. 
    
    inputs: 
    num_cores is the amount of cpu cores we have available 
    
    outputs:
    all the different HMC_chain outputss we would normally have but divided up 
    into the number of cpu cores we have available
    '''
    
    input_list = input_list_creator(num_cores)
    print(len(input_list))
    if __name__ == '__main__':
        with mp.Pool(num_cores = num_cores) as pool:
            #we map the inputs to the function 
            results = pool.starmap(HMC_chain, input_list)
            #and then pull the individual results out as it stores them as a list
            HMC_chains = [x[0] for x in results]
            HMC_acceptance_rate = [x[1] for x in results]
            all_vals = [x[2] for x in results]
            mass_matrices = [x[3] for x in results]
            similarity = [x[4] for x in results]

    return HMC_chains, HMC_acceptance_rate, all_vals, mass_matrices, similarity
