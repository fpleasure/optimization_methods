import unittest
import autograd.numpy as np
from src.optimizer import Optimizer

class TestDichotomie(unittest.TestCase):
	def test_quadratic(self):
		def objective_function(x):
			return (x - 2) ** 2

		initial_guess = (-2, 2)
		precsision = 1e-6
		
		opt = Optimizer("dichotomie")
		x = opt.optimize(objective_function, initial_guess, precsision, callback=False)

		assert abs(x - 2) < precsision

	def test_diff_func(self):
		def objective_function(x):
			return np.sin(x)

		initial_guess = (-0.3 - np.pi / 2, 0.3 - np.pi / 2)
		precsision = 1e-8
		
		opt = Optimizer("dichotomie")
		x = opt.optimize(objective_function, initial_guess, precsision, callback=False)

		assert abs(x + np.pi / 2) < precsision

class TestGoldenRatio(unittest.TestCase):
	def test_quadratic(self):
		def objective_function(x):
			return (x - 2) ** 2

		initial_guess = (-2, 2)
		precsision = 1e-6
		
		opt = Optimizer("golden_ratio")
		x = opt.optimize(objective_function, initial_guess, precsision, callback=False)

		assert abs(x - 2) < precsision

	def test_diff_func(self):
		def objective_function(x):
			return np.sin(x)

		initial_guess = (-0.3 - np.pi / 2, 0.3 - np.pi / 2)
		precsision = 1e-8
		
		opt = Optimizer("golden_ratio")
		x = opt.optimize(objective_function, initial_guess, precsision, callback=False)

		assert abs(x + np.pi / 2) < precsision


if __name__ == '__main__':
    unittest.main()