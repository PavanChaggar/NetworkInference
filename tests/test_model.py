"""File containing test for classes/functions in _model.py
"""

import unittest
import os
import numpy as np

import netwin as nw

class TestModel(unittest.TestCase):

    root_dir = os.path.split(os.path.dirname(__file__))[0]
    network_path = os.path.join(root_dir, 'data/brain_networks/scale1.csv')
    
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

        t = np.linspace(0, 1, 100)
        u0 = np.zeros((n))
        k = 1.0 
        params = m.L, k

        u = m.f(u0, t, params)

        assert u.shape == (n,)
if __name__ == '__main__':
    unittest.main()
