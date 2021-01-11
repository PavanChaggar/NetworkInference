""" script containing class and functions for modelling, integrating etc
""" 
from abc import ABC, abstractmethod

from ._networks import *

class Model(ABC): 
    """Abstract Base Class for implementing dynamical network models.

       Parameters:
       -----------
       network_path (str)
            path to a network array, .csv, or .graphml

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
        if type(network_path) == str:
            self.filename = network_path

        self.__A = adjacency_matrix(network_path)
        self.__D = degree_matrix(self.__A)
        self.__L = graph_Laplacian(A=self.__A, D=self.__D)

        self.t = None

    @abstractmethod
    def f(self):
        """Differential equation function to be implemented

        Parameters:
        -----------
        u0 (array)
            array containing intial conditions
        t (array)
            numpy array containing time steps at which to evaluate model
            e.g. t = np.linespace(0, 1, 100)
        params (array)
            array containing parameter values
            e.g. params = [L, k]
                L = Graph Laplacian
                k = diffusion coefficient
        """
        raise NotImplementedError('This should be implemented')

    @abstractmethod
    def solve(self):
        """Function to use odeint to sovle network model in f

        args: 
        -----
        u0 (array)
            array containing intial conditions
        t (array)
            array containing time steps at which to evaluate model
        params (array)
           array containing parameter values
        """
        raise NotImplementedError('This should be implemented')

    @abstractmethod
    def forward(self):
        """
        Function to describt ehf forward model

        args:
        -----
        u0 (array)
            array containing intial conditions
        t (array)
            arrau containing time steps at which to evaluate model
        params (array)
            array containing parameter values
        """
        raise NotImplementedError('This should be implemented')
    
    def L(self):
        """Return the Laplacian matrix associated with the model.
        """
        return self.__L

    def A(self):
        """Return the adjency matrix associated with the model.
        """
        return self.__A
    
    def D(self):
        """
        Return the degree matrix associated with the model.
        """
        return self.__D 

