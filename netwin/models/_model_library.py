""" Scipt containing different network models 
"""
import numpy as np
from netwin import Model
from scipy.integrate import odeint


class NetworkDiffusion(Model):
    """Network Diffusion model instantiation of the Model class

    Args
    ----
    Model (class)
        abstract base class for forward models
    Path (str)
        string to the csv corresponding to the network A matrix.
    """
    def f(self, p, t, theta):
        """Differential equation model for network diffusion

        Args
        ----
        p (array)
            protein concentration
        t (array)
            array of time points to evlulate at t
        theta (array)
            model parameters
        """
        k = theta
        du = k * (-self.L() @ p)
        return du

    def solve(self, p, theta):
        """numerical integration of ode model

        Args
        ----
        p (array)
            protein concentration vector
        theta (array)
            model parameters
        """
        return odeint(self.f, p, self.t, args=(theta,))

    def forward(self, u0): 
        """forward model taking in free parameters and returning simulated
        data

        Args
        ----

        u0 (array)
            initial values for protein concentration vector and model parameters

        """
        p = np.exp(u0[:-1])
        theta = u0[-1]

        u = self.solve(p,theta) 
        return u

class NetworkFKPP(Model):
    """Network FKPP model instantiation of the Model class

    Args
    ----
    Model (class)
        abstract base class for forward models
    Path (str)
        string to the csv corresponding to the network A matrix.
    """
    def f(self, p, t, theta):
        """Differential equation model for network FKPP
        returns array of values for the ode model
        Args:
        -----
        p (array)
            protein concentration
        t (array)
            array of time points to evlulate at
        theta (array)
            model parameters

        """
        k, a = theta
        du = k * (-self.L() @ p) + (a * p) * (1 - p)
        return du

    def solve(self, p, theta):
        """numerical integration of ode model

        Args
        ----
        p (array)
            protein concentration vector
        theta (array)
            model parameters

        """
        return odeint(self.f, p, self.t, args=(theta,))

    def forward(self, u0): 
        """forward model taking in free parameters and returning simulated
        data

        Args
        ----

        u0 (array)
            initial values for protein concentration vector and model parameters

        """
        p = np.exp(u0[:-2])
        theta = np.exp(u0[-2:])
        
        u = self.solve(p, theta) 
        return u
    
    