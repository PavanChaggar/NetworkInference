"""File containg test for classes/functions in src/networks.py
"""

import unittest
import os
import numpy as np 

import netwin as nw


class Test_Network(unittest.TestCase):
    """Class to test functions in _networks.py
    """
    # get example brain network
    root_dir = os.path.split(os.path.dirname(__file__))[0]
    graph_path = os.path.join(root_dir, 'data/brain_networks/scale1.csv')
    
    # Read adjacency matrix
    A = nw.adjacency_matrix(graph_path)
    # Generate Degree matrix from A
    D = nw.degree_matrix(adjacency_matrix=A)

    def test_adjacency_matrix(self):
        """Test adjacency matrix function for reading csv
        """
        # make sure path ends with .csv
        assert self.graph_path.endswith('.csv')
        # test the matrix is square
        n = len(self.A) 
        assert self.A.shape == (n,n)
        # test the diagonal entries are zero
        assert np.diagonal(self.A).all() == 0

    def test_degree_matrix(self): 
        """Test degrer matrix function for generating the degree matrix from A
        """ 
        # test degree matrix is square
        n = len(self.D)
        assert self.D.shape == (n,n) 
        # test degree matrix is only non-zero at the diagonal 
        assert np.count_nonzero(np.diagonal(self.D)) > 0
        assert np.count_nonzero(self.D - np.diag(np.diagonal(self.D))) == 0
        # test degree matrix has less than or equal to non-zero components than dimension of matrix
        assert np.count_nonzero(np.diagonal(self.D)) <= n

        
    def test_graph_Laplacian(self):
        """Test the graph Laplacian function for generating L from A and D, both with and without D. 
        """
        # create L using only A
        L = nw.graph_Laplacian(A=self.A) 
        # create L using A and D
        L_d = nw.graph_Laplacian(A=self.A, D=self.D)
        
        # test matrix is square
        n = len(L)
        assert L.shape == (n, n)
        
        # test sum is 0 
        assert np.sum(L) == 0
        # test the sum of the off-diagonals is negative
        assert np.sum(L - np.diag(np.diagonal(L))) < 0
        # test the sum of the off diagonals equals the sum of A
        assert -np.sum(L - np.diag(np.diagonal(L))) == np.sum(self.A)

        # repeat tests for L_d
        n = len(L_d)

        assert L_d.shape == (n, n)
        assert np.sum(L_d) == 0
        assert np.sum(L_d - np.diag(np.diagonal(L_d))) < 0
        assert -np.sum(L_d - np.diag(np.diagonal(L_d))) == np.sum(self.A)