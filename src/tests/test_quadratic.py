"""This module contains test of 1D qudratic function for:
- Dichotomie
- Golden ratio
- GD with constant step
"""

import unittest
from src.tests.make_test import make_test


def quadratic_function(x):
	return (x - 2) ** 2

# Initial point for zero order methods
INITIAL_GUESS_ZERO = (-2, 2)

# Initial point for more order methods
INITIAL_GUESS_ONE = 10

PRECSISION = 1e-6


class TestQuadratic(unittest.TestCase):
	"""Unittest class"""
	
	def test_dichotomie(seld):
		x = make_test("dichotomie", quadratic_function, INITIAL_GUESS_ZERO, PRECSISION)
		assert abs(x - 2) < PRECSISION

	def test_golden_ratio(self):
		x = make_test("golden_ratio", quadratic_function, INITIAL_GUESS_ZERO, PRECSISION)
		assert abs(x - 2) < PRECSISION

	def test_gradient_descent(self):
		x = make_test("gradient_descent", quadratic_function, INITIAL_GUESS_ONE, PRECSISION)
		assert abs(x - 2) < PRECSISION

if __name__ == '__main__':
    unittest.main()
