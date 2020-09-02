"""File containg test for classes/functions in src/networks.py
"""

import unittest
import os
import numpy as np 

import netwin as nw


class Test_Network(unittest.TestCase):
    root_dir = os.path.split(os.path.dirname(__file__))[0]
    graph_path = os.path.join(root_dir, 'data/brain_networks/scale1.csv')
    A = nw.adjacency_matrix(graph_path)
    D = nw.degree_matrix(adjacency_matrix=A)

    def test_adjacency_matrix(self):
        assert self.graph_path.endswith('.csv')
        n = len(self.A) 
        assert self.A.shape == (n,n)

    def test_degree_matrix(self): 
        n = len(self.D)
        assert self.D.shape == (n,n) 

        assert np.count_nonzero(self.D - np.diag(np.diagonal(self.D))) == 0
        
    def test_graph_Laplacian(self):
        L = nw.graph_Laplacian(A=self.A) 
        L_d = nw.graph_Laplacian(A=self.A, D=self.D)
        n = len(L)
        assert L.shape == (n, n)