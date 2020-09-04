""" Scipt containing different network models 
"""
def network_diffusion(u0, t, params):
    """Function to implement the network diffusion model 
    args: 
        u0 : array  
             array containing intial conditions 
         t : arrau 
             arrau containing time steps at which to evaluate model   
    params : array 
             array containing parameter values
    returns: 
          u : array 
              solution to differential equation at times t 
    """
    p = u0
    L, k = params
    du = k * (-L @ p)
    return du