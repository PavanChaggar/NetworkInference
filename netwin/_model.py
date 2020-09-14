""" script containing class and functions for modelling, integrating etc
""" 
import netwin as nw
#from ._model_library import *
from ._infer import *

from scipy.integrate import odeint
from abc import ABC, abstractmethod

class Model(ABC): 
    """Model class for the easy implementation of network models
    """
    def __init__(self, network_path: str):
        """initialise class with network and model 
        args:
          filename : str
                     string to the location of a csv containing an adjacency matrix
        model_name : str 
                     string containing the model user wishes to initialised 
                     options: 
                        - 'network_diffusion' 
                           du = k*(L @ p)
        """
        self.filename = network_path

        self.A = nw.adjacency_matrix(self.filename)
        self.D = nw.degree_matrix(self.A)
        self.L = nw.graph_Laplacian(A=self.A, D=self.D)

        self.infer = None

    @abstractmethod
    def f(self):
        """Function set to be the model implemented
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
        raise NotImplementedError('This should be implemented')

    @abstractmethod
    def solve(self):
        """Function to use odeint to sovle network model in f
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
        raise NotImplementedError('This should be implemented')

    @abstractmethod
    def forward(self):
        """Function to describt ehf forward model
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
        raise NotImplementedError('This should be implemented')

    def set_infer(self,inference_scheme):
        """function to set inference class given by a particular inference scheme
        args : 
            inference scheme : str
                               string object with one of the following options: 
                               'mcmc' 
                               'vb'
                               'sbi'
        returns : 
                   inference : class 
                               returns class object corresponding to infernce scheme chosen
        """
        self.infer = set_inference(inference_scheme)
        #self.which_inference = inference_scheme
    