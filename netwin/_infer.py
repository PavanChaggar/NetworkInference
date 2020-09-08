"""script containing classe/functions for inference methods
"""
from netwin._inference import * 
import numpy as np

class vb(object): 
    """class to set up inference schemes for vb
    """
    def __init__(self):
        pass

    def __setpriors(self, init_means):

        m0 = np.zeros_like(init_means)
        p0 = np.linalg.inv(np.diag(np.ones_like(m0) * 1e5))

        beta_mean0 = 1e-2
        beta_var0  = 1e-2

        c0 = beta_var0 / beta_mean0
        s0 = beta_mean0**2 / beta_var0

        priors = m0, p0, c0, s0

        return priors

    def __initialise(self, init_means, priors):
        if priors == None:
            priors = self.__setpriors(init_means)
        
        m = init_means
        p = np.linalg.inv(np.diag(np.ones_like(m)))
        c = priors[2]
        s = priors[3]

        params = m, p, c, s
        
        return params, priors

    def fit(self, data, init_means, t, priors = None, n=50):
        
        self.init_params, self.priors = self.__initialise(init_means, priors)
        
        params, theta_n = fit(self.f, data, init_params, priors, t, n)
        
        return params, theta_n 

class mcmc(object):
    pass 

class sbi(object):
    pass


