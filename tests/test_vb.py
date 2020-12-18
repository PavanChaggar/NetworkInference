"""File containg tests for the implementation of variational Bayes inference
"""

import unittest
import numpy as np
from math import isclose

from netwin import Model, VBModel, infer
from netwin.models import NetworkFKPP

class TestVBInference(unittest.TestCase):
    """Test class for performing inference using analytic variational Bayes 
    """

    A = np.reshape(np.random.normal(5, 2, 25), (5,5))
    np.fill_diagonal(A, 0)
    A = A / np.max(A)

    m = NetworkFKPP(A)
    
    m.t = np.linspace(0,1,50)
    p = np.random.uniform(0,0.5,5)
    u0 = np.append(p, [2.0, 3.0])

    sim = m.forward(np.log(u0))

    def test_classinit(self):
        """test the class has initialised as expected
        """
        assert np.all(self.A.diagonal()  == np.zeros(5))

        assert isinstance(self.m, Model) == True

        assert np.all(self.p < 0.5)
        assert np.all(self.p > 0.0)

        assert len(self.u0) == 7

    def test_inference(self):
        p0 = np.ones([5])
        u_guess = np.append(p0, [1.0,1.0])

        ProbModel = VBModel(model=self.m, data=self.sim, init_means=u_guess)

        assert isinstance(ProbModel, VBModel) == True

        pm = infer(ProbModel, n=20)
    

        assert len(pm.F()) == 20

        means = np.exp(pm.m())
    
        assert len(means) == 7

        for i in range(len(means)):
            assert isclose(means[i],self.u0[i],abs_tol=1e-2) == True
        
        assert ProbModel.get_priors() == pm.get_priors()

        assert isinstance(pm.model(), Model) == True