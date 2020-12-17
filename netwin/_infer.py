"""script containing classe/functions for inference methods
"""
from netwin._inference import * 
import numpy as np
from ._model import Model

class VBModel(object):
    """Class for setting VB Model 
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
        """Initialise VBProblem 
        This sets of the VBProblem class and ensures the set up is correctly set up to interface 
        with the vb protocol. 
        
        args: 
             model : Class object
                     implementation of abstract Model class 
              data : array, float
                     array containing data. This should be the same shape as the output of 
                     the generative model 
        init_means : array, float, 
                     vector of initial guesses for generative model parameters
            priors : None
                     custom priors not yet implemented

        returns: 
         VBProblem : Class object
                     Initialised class object for which inference can be conducted
        """
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
        """ Function to set up vb inference protocol.  
        Sets up initial parameters and priors according to MVN distribution on model parameters 
        and Gamma distribution on noise. 
        args: 
        init_means : array, float
                     vector of initial guesses for generative model parameters
            priors : default None
                     Otherwise not implemented
        
        returns: 
            params : multidimensional array 
                     stores values for MVN and Gamma distribution
            params : multidimensional array 
                     stores values for MVN and Gamma distribution priors
        """
        if priors == None:
            priors = self.__vbsetpriors(init_means)
        else:
            priors = priors
        
        m = init_means
        p = np.linalg.inv(np.eye(len(m)) * 1e5)
        #c = np.array([priors[2]])
        #s = np.array([priors[3]])
        c = np.array([priors[2]])
        s = np.array([priors[3]])
        params = m, p, c, s
        
        return params, priors

    def __vbsetpriors(self, init_means):
        """ Function to set priors based on initial means

        args: 
        init_means : array, float
                     vector of initial guesses for generative model parameters
        
        returns: 
            priors : multidimensional array 
                     stores values for MVN and Gamma distribution                     
        """
        m0 = np.zeros_like(init_means)
        p0 = np.linalg.inv(np.eye(len(m0)) * 1e5)

        #beta_mean0 = 1.0
        #beta_var0  = 1000.0

        #s0 = beta_var0 / beta_mean0
        #c0 = beta_mean0**2 / beta_var0

        c0 = 1e-8
        s0 = 50.0

        priors = m0, p0, c0, s0

        return priors
        
    def optimise(self, n = 10):
        """ Runs inference by calling from _vb.py 
        
        args: 
            n : int
                number of iterations to run vb 
        
        returns :
          sol : multidimensional array
                fitted parameters for MVN and Gamma moments
            F : array 
                vector array containing free energy tracking 
        """
        if self.__which_inference == 'VB': 
            return vb(M=self.__model, data=self.__data, t=self.__t, params=self.__params, priors=self.__priors, n_params=self.__n_params, n=n)

    def get(self, attribute:str):
        """function to return values from hidden variables 
        
        args: 
        attribute : string
                    name of attribute one wishes to retrieve. 
                    Options are: 
                    'model'
                    'data'
                    'time'
                    'init_means'
                    'params'
                    'priors'
                    'n_params'
                    'which'

        returns: 
        attributes[arg] : argument value
        """
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