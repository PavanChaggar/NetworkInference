""" script containing class and functions for modelling, integrating etc
""" 
from abc import ABC, abstractmethod

from ._networks import *

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
        if type(network_path) == str:
            self.filename = network_path

        self.__A = adjacency_matrix(network_path)
        self.__D = degree_matrix(self.__A)
        self.__L = graph_Laplacian(A=self.__A, D=self.__D)

        self.t = None

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
    
    def L(self):
        """Return the Laplacian matrix associated with the model.

        Returns:
            L : numpy array
                n x n Laplacian Matrix
        """
        return self.__L

    def A(self):
        """Return the adjency matrix associated with the model.

        Returns:
            A : numpy array
                n x n adjacency Matrix
        """
        return self.__A 
    
    def D(self):
        """Return the degree matrix associated with the model.

        Returns:
            D : numpy array
                n x n Degree Matrix
        """
        return self.__D 

