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

if __name__ == '__main__':
    unittest.main()
