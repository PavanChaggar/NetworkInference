"""script containing classe/functions for inference methods
"""
from netwin._inference import * 
import numpy as np

class VB(object): 
    """class to set up inference schemes for vb
    """
    def __init__(self):
        print('vb')
        self.priors = None
        self.init_params = None

    def __setpriors(self, init_means):

        m0 = np.zeros_like(init_means)
        p0 = np.linalg.inv(np.diag(np.ones_like(m0) * 1e5))

        beta_mean0 = 1.0
        beta_var0  = 1000.0

        c0 = beta_var0 / beta_mean0
        s0 = beta_mean0**2 / beta_var0

        priors = m0, p0, c0, s0

        return priors

    def __initialise(self, init_means, priors):
        if priors == None:
            priors = self.__setpriors(init_means)
        
        m = init_means
        p = np.linalg.inv(np.diag(np.ones_like(m)))
        #c = np.array([priors[2]])
        #s = np.array([priors[3]])
        c = np.array([1e-8])
        s = np.array([50.0])
        params = m, p, c, s
        
        return params, priors

    def fit(self, f, data, init_means, t, priors = None, n=50):
        
        self.init_params, self.priors = self.__initialise(init_means, priors)

        params, theta_n = fit(f, data, self.init_params, self.priors, t, n)

        return params, theta_n

class MCMC(object):
    pass 

class SBI(object):
    pass


def set_inference(inference_scheme: str):
    inference_map = {
        "vb": VB(), 
        "mcmc": MCMC(),
        "sbi" : SBI()
    }
    return inference_map[inference_scheme]

class InferenceProblem(object):
    def __init__(self, inference:str, model=None, data=None, time=None, init_means=None, priors=None):
        if inference == 'VB': 
            self.which_inference = 'VB'
            self.model = model
            self.data = data 
            self.t = t
            self.init_means = init_means
            self.params, self.priors = self.__vbinferenceproblem(self)

    def __vbinferenceproblem(self): 
        if priors == None:
        priors = setpriors(init_means)
    
        m = init_means
        p = np.linalg.inv(np.diag(np.ones_like(m)))
        #c = np.array([priors[2]])
        #s = np.array([priors[3]])
        c = np.array([1e-8])
        s = np.array([50.0])
        params = m, p, c, s
        
        return params, priors

    def __vbsetpriors(init_means):

        m0 = np.zeros_like(init_means)
        p0 = np.linalg.inv(np.diag(np.ones_like(m0) * 1e5))

        beta_mean0 = 1.0
        beta_var0  = 1000.0

        c0 = beta_var0 / beta_mean0
        s0 = beta_mean0**2 / beta_var0

        priors = m0, p0, c0, s0

        return priors

    def infer(self):
        return fit(problem)