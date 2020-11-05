"""script containing classe/functions for inference methods
"""
from netwin._inference import * 
import numpy as np
from ._model import Model

class VBProblem(object):
    """Class for setting Inference Problems 
    Presently implemented for VB only. 
    Will return a structured object that can be passed into 'fit' to perform variational inference
    For VB, the required arguments are: 
        model : class implementing forward model 
        data  : data one wishes to perform inference against
        t     : time steps at which to evaluate forward model 
    init means: initial guess for the means of each free parameter
    
    In addition to the input variables, the __init__ will return initial distribution
    parameter values and prior distribution parameter values
    """
    def __init__(self, model=None, data=None, init_means=None, priors=None):
        self.__which_inference = 'VB'

        if not isinstance(model, Model):
            raise TypeError('Please change this class to inherit from nw.Model.')

        self.__model = model #check model is instance of model class
        self.__data = data 
        self.__t = model.t
        self.__init_means = init_means
        self.__params, self.__priors = self.__vbinferenceproblem(init_means)
        self.__n_params = len(init_means) - len(model.L())

    def __vbinferenceproblem(self, init_means, priors=None): 
        if priors == None:
            priors = self.__vbsetpriors(init_means)
    
        m = init_means
        p = np.linalg.inv(np.diag(np.ones_like(m) * 1e5))
        #c = np.array([priors[2]])
        #s = np.array([priors[3]])
        c = np.array([1e-8])
        s = np.array([50.0])
        params = m, p, c, s
        
        return params, priors

    def __vbsetpriors(self, init_means):

        m0 = np.zeros_like(init_means)
        p0 = np.linalg.inv(np.diag(np.ones_like(m0) * 1e5))

        beta_mean0 = 1.0
        beta_var0  = 1000.0

        s0 = beta_var0 / beta_mean0
        c0 = beta_mean0**2 / beta_var0

        priors = m0, p0, c0, s0

        return priors
        
    def infer(self, n = 10, priors=None):
        if priors != None:
            priors = priors
        else:
            priors = self.__priors
        if self.__which_inference == 'VB': 
            return vb(M=self.__model, data=self.__data, t=self.__t, params=self.__params, priors=priors, n_params=self.__n_params, n=n)

    def get(self,attribute:str):
        attributes = {
             'model': self.__model,
              'data': self.__data,
              'time': self.__t,
        'init_means': self.__init_means,
            'params': self.__params,
            'priors': self.__priors,
          'n_params': self.__n_params,
             'which': self.__which_inference
        }
        return attributes[attribute]