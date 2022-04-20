#class version of the HMC code I have previously written 
import numpy as np 
from autograd import elementwise_grad 
import matplotlib.pyplot as plt 

class HMC():
    
    #defining the initialisation process 
    def __init__(self, init):
        #defining the dimensionality of the model 
        self.dim = init.shape[0]
        #initialise empty lsit of states 
        self.chain = []
        #tracking the acceptance ratios 
        self.accepted, self.rejected, self.total_proposed = 0, 0, 0
        #initial values of position and momentum 
        self.x_old, self.x_new, self.p_old, self.p_new = np.zeros(self.dim), np.zeros(self.dim), np.zeros(self.dim), np.zeros(self.dim)
        #initializing the mass matrix 
        self.M = np.eye(self.dim)

        
    #defining the leapfrog function, taken from the leapfrog_alt code 
    def leapfrog(self, epsilon, L, M, gradient):
        #set x_old to x_new so we can propogate x_new 
        self.x_new = self.x_old.copy()
        #draw from a normal distribution for the momentum values
        self.p_old = []
        for i in range(len(self.x_new)):
            self.p_old.append(np.random.normal(loc = 0, scale = M[i][i]))
        self.p_old = np.asarray(self.p_old)
        self.p_new = np.add(self.p_old, 0.5 * epsilon * gradient(self.x_new))
        
        for _ in range(L):
            self.x_new = np.add(self.x_new, epsilon * self.p_new)
            self.p_new = np.add(self.p_old, 0.5 * epsilon * gradient(self.x_new))
            
        self.x_new = np.add(self.x_new, epsilon * self.p_new)
        self.p_new = np.add(self.p_old, 0.5 * epsilon * gradient(self.x_new))
        return self.x_new, self.p_new
    
    #acceptance criteria 
    def acceptance(self, log_prob, M):
        #update the new total 
        self.total_proposed += 1
        #hamiltonian value prior to the leapfrog steps 
        H_old = self.H(self.x_old, self.p_old, log_prob, M)
        #hamiltonian value post leapfrog steps 
        H_new = self.H(self.x_new, self.p_new, log_prob, M)
        #MH ratio 
        MH = -(H_new - H_old)
        #deciding whether to accept 
        if np.log(np.random.random()) < min(0, MH):
            self.accepted += 1
            #setting new position as old if it is accepted 
            self.x_old = self.x_new
            #updating the chain
            self.chain.append(self.x_old)
        else: 
            self.rejected += 1
        
        return MH
    
    #hamiltonian energy function 
    def H(self, x, p, log_prob, M):
        E = -log_prob(x)
        K = 0.5 * (np.transpose(p) @ np.linalg.inv(M) @ p)
        return E + K
    
    #updating the mass matrix 
    def prior_M_adapt(self, num_prior_samples, prior_var):
        num_samples = len(self.chain)
        array = np.asarray(self.chain).T
        current_var = np.cov(array)
        M = np.linalg.inv((num_samples*current_var + num_prior_samples*prior_var)/(num_samples + num_prior_samples))
        return M 
    
    #defining a plotting function for the chain 
    def plot_samples(self, dim1, dim2, x_start, x_end):
        samples = np.asarray(self.chain)  
        x1 = np.linspace(x_start, x_end, 100)
        X, Y = np.meshgrid(x1, x1)
        flattened_X = X.flatten()
        flattened_Y = Y.flatten()
        #getting the coordinates we want to graph across 
        coords = []
        for i in range(len(flattened_X)):
            xs = flattened_X[i]
            ys = flattened_Y[i]
            coords.append(np.array([xs, ys]))
        Zs = []
        for i in coords:
            Zs.append(log_prob(i))
        Zs = np.asarray(Zs)
        Zs = Zs.reshape((len(X), len(Y)))
        plt.contourf(X, Y, Zs, 100, cmap='jet')
        plt.scatter(samples[:, dim1], samples[:, dim2], s=0.5, c='w')
        plt.title(f'XY plot of the samples for dimensions {dim1} and {dim2}')
        plt.show()
    
    #defining the dual averaging function if we decide we want to use it 
    def dual_averaging(self, burn_in, sample_num, M, mu, alpha, k = 0.75, H_bar = 0, gamma = 0.05, t_0 = 10, delta = 0.7, lambda_ = 1, step_size_bar = 1):
        H_bar = (1 - 1/(sample_num + t_0))*H_bar + (1/(sample_num + t_0))*(delta - alpha)
        step_size = np.exp(mu - np.sqrt(sample_num)*H_bar/gamma)
        step_size_bar = np.exp(sample_num**(-k)*np.log(step_size) + (1- sample_num**(-k))*np.log(step_size_bar))
        self.epsilon = step_size
        self.epsilon_bar = step_size_bar
        return self.epsilon, self.epsilon_bar
    
    #The dual averaging and initial step size functions are tkane from the NUTS paper 
    def initial_step_size(self, log_prob, gradient, M):
        epsilon = 1
        L = 1
        parems = len(self.x_old)
        p = []
        for i in range(parems):
            p_element = np.random.normal(loc=0, scale=1)
            p.append(p_element)
        p = np.asarray(p)
        x_new, p_new = HMC.leapfrog(self, epsilon, L, M, gradient)
        p_acc = HMC.acceptance(self, log_prob, M)
        a = 2*int(p_acc > 0.5) - 1
        while p_acc**a > 2**(-a):
            p = []
            for i in range(parems):
                p_element = np.random.normal(loc=0, scale=1)
                p.append(p_element)
            p = np.asarray(p)
            epsilon = epsilon*(2**(a))
            x_new, p_new = HMC.leapfrog(self, epsilon, L, M, gradient)
            p_acc = HMC.acceptance(self, log_prob, M)
            alpha = min(1, p_acc)
            x = x_new
            p = p_new  
        self.x_old = np.zeros(parems)
        return epsilon, alpha
