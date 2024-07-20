"""This module contains optimizer class."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.methods import gradient_descent, one_dimensional


class Optimizer:
	"""Class to collect all optimizers

	Attributes
	----------
	_optimizer : child class BaseOptimizer
		optimizer realization

	Methods
	----------
	optimize(objective_function, initial_guess, precsision=1e-8, callback=False)
		Find minimum of objective function
	"""
    
	def __init__(self, optimizer_name, max_iter=1000):
		"""
		Parameters
		----------
		optimizer_name : str
			name of optimizer algorithm

		max_iter : int
			max iterations, if achieved, it generates an error
		"""
		
		if optimizer_name == "dichotomie":
			self._optimizer = one_dimensional.Dichotomie(max_iter)
		elif optimizer_name == "golden_ratio":
			self._optimizer = one_dimensional.GoldenRatio(max_iter)
		elif optimizer_name == "gradient_descent":
			self._optimizer = gradient_descent.GradientDescent(max_iter)
		elif optimizer_name == "gradient_descent_steepest":
			self._optimizer = gradient_descent.GradientDescentSteepest(max_iter)
		elif optimizer_name == "gradient_descent_crash_step":
			self._optimizer = gradient_descent.GradientDescentCrashStep(max_iter)
		else:
			raise ValueError(f"Optimizer with name '{optimizer_name}' does not exist!")

	def optimize(self, objective_function, initial_guess, precsision=1e-8, callback=False):
		"""Return minimum point if callback=False
		or return data about algorithm steps if callback=True

		Parameters
		----------
		objective_function : Callable
			function to minimize

		initial_guess : Union[Sequence[float], np.ndarray, float]
			starting point of algorithm

		precsision : float
			point location accuracy

		callback : bool
			information about optimization 
			algorithm steps"""
		
		return self._optimizer.optimize(objective_function, initial_guess, precsision, callback)