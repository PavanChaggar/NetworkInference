"""File containing test for classes/functions in _model.py
"""

import unittest
import os
import numpy as np

import netwin as nw

class TestModel(unittest.TestCase):
    """Class to test the Model class in _model.py
    """
    # set root directory and path to example network
    root_dir = os.path.split(os.path.dirname(__file__))[0]
    network_path = os.path.join(root_dir, 'data/brain_networks/scale1.csv')
    
    # set time steps, intial condition and diffusion constant for network diffusion model
    t = np.linspace(0, 1, 100)
    u0 = np.ones((83))
    k = 1.0 
    a = 1.0

    def test_init(self):
        """Test initialisation of model class
        """
        # instantiate classw with example network and network diffusion model
        m = nw.Model(network_path = self.network_path, model_name='network_diffusion')
        
        # check the correct model has been assigned and that a function is returned
        assert m.which_model == 'network_diffusion'
        assert callable(m.f) == True
        
        n = len(m.A)
        
        # test the matrices of m are all square (adjacency, degree and Laplacian) 
        assert m.A.shape == (n,n)
        assert m.D.shape == (n,n)
        assert m.L.shape == (n,n)
    
    def test_models(self): 
        """Test _models to assign model to f
        """
        # Instantiate class 
        m = nw.Model(network_path = self.network_path, model_name='network_diffusion')
        n = len(m.A)

        params = m.L, self.k
        
        # run simulation using f
        u = m.f(self.u0, self.t, params)

        # check that f returns an array as expected
        assert u.shape == (n,)
        assert np.all(u!=self.u0)

        m = nw.Model(network_path = self.network_path, model_name='fkpp')
        assert m.which_model == 'fkpp' 
        
        params = m.L, self.k, self.a
        
        u2 = m.f(self.u0, self.t, params)

        # check that f returns an array as expected
        assert u2.shape == (n,)
        assert np.all(u2!=self.u0)


    def test_solve_diffusion(self): 
        """Test solver for network diffusion model
        """
        #Instantiate class
        m = nw.Model(network_path = self.network_path, model_name='network_diffusion')
        # pack parameters and solve for initial values
        params = m.L, self.k
        sol = m.solve(self.u0, self.t, params)
        
        # test the shape of the solution and solution is as expected
        assert sol.shape == (len(self.t),len(m.A))
        assert sol[0,:].all() == sol[-1,:].all()

        # set up a different problem with non-uniform initial conditions
        u0_2 = np.ones((len(m.A)))
        u0_2[30] = 10.0 
        sol_2 = m.solve(u0_2, self.t, params)
        
        # test the solution for the two solutions are different as expected
        assert np.all(sol[0,:]==sol_2[0,:]) == False
        assert np.all(sol_2[0,:]!=sol_2[-1,:])

    def test_solve_fkpp(self): 
        """Test solver for network fkpp model
        """
        #Instantiate class
        m = nw.Model(network_path = self.network_path, model_name='fkpp')
        # pack parameters and solve for initial values
        params = m.L, self.k, self.a
        sol = m.solve(self.u0, self.t, params)
        
        # test the shape of the solution and solution is as expected
        assert sol.shape == (len(self.t),len(m.A))
        assert sol[0,:].all() == sol[-1,:].all()

        # set up a different problem with non-uniform initial conditions
        u0_2 = np.ones((len(m.A)))
        u0_2[30] = 10.0 
        sol_2 = m.solve(u0_2, self.t, params)
        
        # test the solution for the two solutions are different as expected
        assert np.all(sol[0,:]==sol_2[0,:]) == False
        assert np.all(sol_2[0,:]!=sol_2[-1,:])

if __name__ == '__main__':
    unittest.main()
