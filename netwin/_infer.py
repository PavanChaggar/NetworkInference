"""script containing classe/functions for inference methods
"""
from netwin._inference import * 
import numpy as np
from ._model import Model

class VBModel(object):
    """Class for setting Inference  
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
        """Initialise VBModel 
        This sets of the VBModel class and ensures the set up is correctly set up to interface 
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
        #self.__params, self.__priors = self.__vbinferenceproblem(init_means)

        # moments
        self.__m, self.__p, self.__c, self.__s = self.__vbsetparams(init_means)
        self.__m0, self.__p0, self.__c0, self.__s0 = self.__vbsetpriors(init_means)
        
        self.__n_params = len(init_means) - len(model.L())

        self.__F = None

    def __vbsetparams(self, init_means): 
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
        m = init_means
        p = np.linalg.inv(np.eye(len(m)) * 1e5)
        #c = np.array([priors[2]])
        #s = np.array([priors[3]])
        c = np.array([1e-8])
        s = np.array([50.0])

        return m, p, c, s
        

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
        c0 = np.array([1e-8])
        s0 = np.array([50.0])


        return m0, p0, c0, s0

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
        return infer(pm=self, M=self.__model, data=self.__data)

    def data(self): 
        return self.__data

    def init_means(self):
        return self.__init_means

    def params(self):
        return self.__params
    
    def priors(self):
        return self.__priors

    def n_params(self):
        return self.__n_params

    def m(self):
        return self.__m
    
    def p(self):
        return self.__p

    def Cov(self):
        return np.linalg.inv(self.__p)

    def c(self):
        return self.__c 
    
    def s(self):
        return self.__s

    def m0(self):
        return self.__m0
    
    def p0(self):
        return self.__p0

    def Cov0(self):
        return np.linalg.inv(self.__p0)

    def c0(self):
        return self.__c0
    
    def s0(self):
        return self.__s0

    def F(self):
        return self.__F

    def set_params(self, params) 
        self.__m, self.__p, self.__c, self.__s = params
        
    def set_priors(self, priors)
        self.__m0, self.__p0, self.__c0, self.__s0 = priors