""" script containing class and functions for modelling, integrating etc
""" 
import netwin as nw
from netwin._inference import * 
from ._model_library import *
from scipy.integrate import odeint

class Model(object): 
    """Model class for the easy implementation of network models
    """
    def __init__(self, network_path: str, model_name: str):
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
        self.which_model = model_name

        self.A = nw.adjacency_matrix(self.filename)
        self.D = nw.degree_matrix(self.A)
        self.L = nw.graph_Laplacian(A=self.A, D=self.D)

        self.f, self.which_model = self._models(model_choice = self.which_model)

    def _models(self, model_choice: str): 
        """HoF to set model variable based on user choice 
           Presently only network diffusion is implemented 
        args: 
        model choice : str 
                       string containg the moder user wishes to initialise
        returns:
                   f : function
                       function imported from model library implementing desired network model
        model_choice : str 
                       return string containing the function set to f
        """
        if model_choice == 'network_diffusion': 
            f = network_diffusion
            return f, model_choice
        elif model_choice == 'fkpp':
            f = network_fkpp
            return f, model_choice 

    def simulate(self, u0, t, params):
        """Function to use odeint to sovle network models 
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
        return odeint(self.f, u0, t, args=(params,))

