"""File containg test for classes/functions in src/networks.py
"""

import unittest
import os
import netwin as nw


class Test_Network(unittest.TestCase):
    root_dir = os.path.split(os.path.dirname(__file__))[0]
    graph_path = os.path.join(root_dir, 'data/brain_networks/scale1.csv')

    def test_adjacency_matrix(self):
        assert self.graph_path.endswith('.csv')
         
        A = nw.adjacency_matrix(self.graph_path)
        n = len(A) 
        assert A.shape == (n,n)

    def test_degree_matrix(self): 
        pass
