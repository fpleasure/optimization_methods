"""This module contains test of sin function for:
- Dichotomie
- Golden ratio
- GD with constant step
"""


import unittest
import autograd.numpy as np
from src.tests.make_test import make_test

# Initial point for zero order methods
INITIAL_GUESS_ZERO = (-0.3 - np.pi / 2, 0.3 - np.pi / 2)

# Initial point for more order methods
INITIAL_GUESS_ONE = 0

PRECSISION = 1e-8


class TestSin(unittest.TestCase):
	"""Unittest class"""
	
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
