import unittest
import autograd.numpy as np
from src.optimizer import Optimizer
from src.tests.make_test import make_test
from src.functions import rosenbrock


INITIAL_GUESS = (5., 5.)
PRECSISION = 1e-8


class TestRosenbrock(unittest.TestCase):
	def test_gradient_descent(self):
		x = make_test("gradient_descent", rosenbrock, INITIAL_GUESS, PRECSISION)
		assert np.linalg.norm(x - np.array((0., 0.))) < PRECSISION

if __name__ == '__main__':
    unittest.main()
