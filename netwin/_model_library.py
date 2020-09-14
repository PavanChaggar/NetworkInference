""" Scipt containing different network models 
"""
import numpy as np
from netwin import Model

def network_diffusion(u0, t, params):
    """Function to implement the network diffusion model 
    args: 
        u0 : array  
             array containing intial conditions 
         t : array
             numpy array containing time steps at which to evaluate model
             e.g. t = np.linespace(0, 1, 100)
    params : array 
             array containing parameter values
             e.g. params = [L, k]
                  L = Graph Laplacian 
                  k = diffusion coefficient 
    returns: 
          u : array 
              solution to differential equation at times t 
    """
    p = u0
    L, k = params
    du = k * (np.matmul(-L, p))
    return du

def network_fkpp(u0, t, params):
    """Function to implement the network Fisher-Kolmogorov–Petrovsky–Piskunov
    args: 
        u0 : array  
             array containing intial conditions 
         t : array
             numpy array containing time steps at which to evaluate model
             e.g. t = np.linespace(0, 1, 100)
    params : array 
             array containing parameter values
             e.g. params = [L, k]
                  L = Graph Laplacian 
                  k = diffusion coefficient
                  a = rate coefficient 
    returns: 
          u : array 
              solution to differential equation at times t 
    """  

    p = u0
    L, k, a = params
    du = k * np.matmul(-L, p) + (a * p) * (1 - p)
    return du

def exponential_decay(u0, t): 
    """ Sipmle exponential decay model

    args: 
    theta : array, float
            parameters for function
        t : array, float 
            timesteps to evaluate at
    """
    a, l = u0
    f1 = a * np.exp(-l * t)

    return f1

class NetworkDiffusion(Model):

    def f(self, p, t, theta):
        k = theta
        du = k * (np.matmul(-self.L, p))
        return du

    def solve(self, p, t, theta):
        return odeint(self.f, p, t, args=(theta,))

    def forward(self, u0, t): 
        p = np.exp(u0[:-1])
        theta = u0[-1]
        #
        # print(u0)
        u = self.solve(p, t, theta) 
        return u.T 

def set_model(model: str):
    model_map = {
        "network_diffusion": network_diffusion, 
        "network_fkpp": network_fkpp,
        "exponential_decay" : exponential_decay
    }
    return model_map[model]
    
    