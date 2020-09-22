""" Scipt containing different network models 
"""
import numpy as np
from netwin import Model
from scipy.integrate import odeint


class NetworkDiffusion(Model):

    def f(self, p, t, theta):
        k = theta
        du = k * (-self.L() @ p)
        return du

    def solve(self, p, theta):
        return odeint(self.f, p, self.t, args=(theta,))

    def forward(self, u0): 
        p = np.exp(u0[:-1])
        theta = u0[-1]

        u = self.solve(p,theta) 
        return u.T  

class NetworkFKPP(Model):
    def f(self, p, t, theta):
        k, a = theta
        du = k * (-self.L() @ p) + (a * p) * (1 - p)
        return du

    def solve(self, p, theta):
        return odeint(self.f, p, self.t, args=(theta,))

    def forward(self, u0): 
        p = np.exp(u0[:-2])
        theta = u0[-2:]
        
        u = self.solve(p, theta) 
        return u.T 
    
    