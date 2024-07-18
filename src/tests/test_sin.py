import unittest
import autograd.numpy as np
from src.optimizer import Optimizer
from src.tests.make_test import make_test


INITIAL_GUESS_ZERO = (-0.3 - np.pi / 2, 0.3 - np.pi / 2)
INITIAL_GUESS_ONE = 0
PRECSISION = 1e-8


class TestSin(unittest.TestCase):
	def test_dichotomie(seld):
		x = make_test("dichotomie", np.sin, INITIAL_GUESS_ZERO, PRECSISION)
		assert abs(x + np.pi / 2) < PRECSISION

	def test_golden_ratio(self):
		x = make_test("golden_ratio", np.sin, INITIAL_GUESS_ZERO, PRECSISION)
		assert abs(x + np.pi / 2) < PRECSISION

	def test_gradient_descent(self):
		x = make_test("gradient_descent", np.sin, INITIAL_GUESS_ONE, PRECSISION)
		assert abs(x + np.pi / 2) < PRECSISION



if __name__ == '__main__':
    unittest.main()
