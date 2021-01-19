"""script containing classe/functions for inference methods
"""
#from netwin._inference import * 
from netwin import Model
from netwin._inference import vb
import numpy as np
from functools import singledispatch
#from ._model import Model

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
        if isinstance(init_means, np.ndarray):
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
        return fit(pm=self)
    
    def model(self):
        """return forward model
        """
        return self.__model

    def data(self): 
        """Return data
        """
        return self.__data

    def init_means(self):
        """Return initial means
        """
        return self.__init_means

    def n_params(self):
        """Return number of ode model parameters
        """
        return self.__n_params

    def m(self):
        """Return vector of means of MVN distribution
        """
        return self.__m
    
    def p(self):
        """Return precision matrix for MVN diistribution
        """
        return self.__p

    def Cov(self):
        """return Covariance matrix for MVN distribution
        """
        return np.linalg.inv(self.__p)

    def c(self):
        """return shape parameters for Gamma distribution
        """
        return self.__c 
    
    def s(self):
        """Return scale parameters for Gamma distribution
        """
        return self.__s

    def m0(self):
        """Return prior on means for MVN distribution
        """
        return self.__m0
    
    def p0(self):
        """Return prior on precision matrix for MVN distribution
        """
        return self.__p0

    def Cov0(self):
        """Return the prior covariance matrix for the MVN distribution
        """
        return np.linalg.inv(self.__p0)

    def c0(self):
        """Return the prior on the shape parameter for the Gamma distribution
        """
        return self.__c0
    
    def s0(self):
        """Return the prior on the scale parameters for the Gamma distribution
        """
        return self.__s0

    def F(self):
        """Return array of free energy for n iterations
        """
        return self.__F

    def set_params(self, params): 
        """Set the parameters of a VBModel

        Args:
            params : tuple
                     containing values for m, p, c and s. Can be obtained using
                     get_priors or a solution to infer for VBModel
        """
        self.__m, self.__p, self.__c, self.__s = params
        
    def set_priors(self, priors):
        """Set the priors of a VBModel

        Args: 
            priors : tuple
                     tuple containing values for m, p, c, s
        """
        self.__m0, self.__p0, self.__c0, self.__s0 = priors
    
    def get_priors(self):
        """Return tuple of priors

        Returns:
            priors: tuple
                    tuple of priors m0, p0, c0, and s0. 
        """
        return self.__m0, self.__p0, self.__c0, self.__s0 
    
    def set_F(self, F):
        """Set the value of F for the VBModel

        Args:
            F : array
                Array containing free energy for each iteration of VB update
        """
        self.__F = F


@singledispatch
def infer(ProbModel=None):
    """Base generic function for infer

    Args:
        ProbModel : Defaults to None.

    Raises:
        NotImplementedError: Make sure the corrext model type is entered
    """
    raise NotImplementedError("Implement Probablistic Model.")

@infer.register(VBModel)
def _(ProbModel, n=20):
    """Perform VB optimisation on VBModel

    Args:
        ProbModel : VBModel
                    class object implemented with VBModel 
                n : int
                    number of iterations for VB updates 

    Returns:
        Solution : VBModel
                   Optimised ProbModel object using Variational Bayes
    """
    sol, F =  vb(pm=ProbModel, n=n)
    pm = VBModel(model=ProbModel.model(), data=ProbModel.data())
    pm.set_params(sol)
    pm.set_priors(ProbModel.get_priors())
    pm.set_F(F)
    return pm