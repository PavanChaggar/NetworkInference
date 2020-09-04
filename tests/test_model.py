"""File containing test for classes/functions in _model.py
"""

import unittest
import os
import numpy as np

import netwin as nw

class TestModel(unittest.TestCase):

    root_dir = os.path.split(os.path.dirname(__file__))[0]
    network_path = os.path.join(root_dir, 'data/brain_networks/scale1.csv')
    
    t = np.linspace(0, 1, 100)
    u0 = np.ones((83))
    k = 1.0 

    def test_init(self):
        m = nw.Model(network_path = self.network_path, model_name='network_diffusion')
        
        assert m.which_model == 'network_diffusion'
        assert callable(m.f) == True
        
        n = len(m.A)

        assert m.A.shape == (n,n)
        assert m.A.shape == (n,n)
        assert m.A.shape == (n,n)

        assert callable(m.f) == True
    
    def test_models(self): 
    
        m = nw.Model(network_path = self.network_path, model_name='network_diffusion')
        n = len(m.A)

        params = m.L, self.k

        u = m.f(self.u0, self.t, params)

        assert u.shape == (n,)
        assert np.all(u!=self.u0)

    def test_solve(self): 
        m = nw.Model(network_path = self.network_path, model_name='network_diffusion')
        params = m.L, self.k
        sol = m.solve(self.u0, self.t, params)
        
        assert sol.shape == (len(self.t),len(m.A))
        assert sol[0,:].all() == sol[-1,:].all()

        u0_2 = np.ones((len(m.A)))
        u0_2[30] = 10.0 
        sol_2 = m.solve(u0_2, self.t, params)
        print(sol)
        print(sol_2)
        
        assert np.all(sol[0,:]==sol_2[0,:]) == False
        assert np.all(sol_2[0,:]!=sol_2[-1,:])

if __name__ == '__main__':
    unittest.main()
