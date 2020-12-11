"""File containing test for classes/functions in _model.py
"""

import unittest
import os
import numpy as np

import netwin as nw
from netwin import Model

class TestModel(unittest.TestCase):
    """Class to test the Model class in _model.py
    """
    # set root directory and path to example network
    root_dir = os.path.split(os.path.dirname(__file__))[0]
    network_path = os.path.join(root_dir, 'data/brain_networks/scale1.csv')


    def test_init(self):
        """Test initialisation of model class
        """
        # instantiate classw with example network and network diffusion model
        # check the correct model has been assigned and that a function is returned
        class test_model(Model):
            def f(self):
                pass
            def solve(self):
                pass
            def forward(self):
                pass
        
        m = test_model(self.network_path)

        assert isinstance(m, Model) == True
        
        assert callable(m.f) == True
        assert callable(m.solve) == True
        assert callable(m.forward) == True

        n = len(m.A())
        
        # test the matrices of m are all square (adjacency, degree and Laplacian) 
        assert m.A().shape == (n,n)
        assert m.D().shape == (n,n)
        assert m.L().shape == (n,n)
    
    
    def test_network_diffusion(self): 
        """Test _models to assign model to f
        """
        # Instantiate class 
        m = nw.NetworkDiffusion(self.network_path)
        n = len(m.A())

        m.t = np.linspace(0, 1, 100)
        u0 = np.ones((n))
        k = 10.0
        
        # run simulation using f
        u = m.f(u0, m.t, k)

        # check that f returns an array as expected
        assert u.shape == (n,)
        assert np.all(u!=u0)

        # test forward model, including ode integration using scipy
        u0 = np.append(u0, k)

        sol = m.forward(u0)
        
        # test the shape of the solution and solution is as expected
        assert sol.shape == (len(m.t),len(m.A()))
        assert sol[0,:].all() == sol[-1,:].all()

        # set up a different problem with non-uniform initial conditions
        u0_2 = np.ones((len(m.A())))
        u0_2[30] = 10.0 

        u0_2 = np.append(u0_2, k)

        sol_2 = m.forward(u0_2)
        
        # test the solution for the two solutions are different as expected
        assert np.all(sol[0,:]==sol_2[0,:]) == False
        assert np.all(sol_2[0,:]!=sol_2[-1,:])

    def test_network_fkpp(self): 

        m = nw.NetworkFKPP(self.network_path)
        
        n = len(m.A())

        m.t = np.linspace(0, 1, 100)
        u0 = np.ones((n))
        params = 10.0, 5.0

        u = m.f(u0, m.t, params)

        # check that f returns an array as expected
        assert u.shape == (n,)
        assert np.all(u!=u0)

        u0 = np.append(u0, params)
        sol = m.forward(u0)
        
        # test the shape of the solution and solution is as expected
        assert sol.shape == (len(m.t),len(m.A()))
        assert sol[0,:].all() == sol[-1,:].all()

        # set up a different problem with non-uniform initial conditions
        u0_2 = np.zeros((n))
        u0_2[30] = 10.0 
        u0_2 = np.append(u0_2, params)
        sol_2 = m.forward(u0_2)
        
        # test the solution for the two solutions are different as expected
        assert np.all(sol[0,:]==sol_2[0,:]) == False

if __name__ == '__main__':
    unittest.main()
