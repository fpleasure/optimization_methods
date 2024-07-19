"""This module contains gradient descent (GD) algorithms:
- GD with constant step
- GD with crash step
- Steepest Descent
"""

from src.methods.base import BaseOptimizer
from src.methods.supporting_functions import to_float
from src.methods.one_dimensional import GoldenRatio

import autograd.numpy as np
from autograd import grad


class GradientDescent(BaseOptimizer):
	"""Gradient descent with constant step

	Attributes
    ----------
	learning_rate : float
		gradient descent tuning parametr
	max_iter : int
		max iterations, if achieved, it generates an error
	callback_data : dict
		information about optimization algorithm steps

	Methods
	----------
	optimize(objective_function, initial_guess, precsision=1e-8, callback=False)
		Find minimum of objective function
	
	_initialize_callback(initial_guess, objective_function)
		Initialize data structures for optimization 
		algorithm steps

	_set_callback_on_step(objective_function, x)
		Add information about optimization 
		algorithm step
	"""
	
	def __init__(self, max_iter=1000):
		self.max_iter = max_iter
		self.callback_data = dict()

	def optimize(self, objective_function, initial_guess, precsision=1e-8, callback=False, learning_rate=0.1):
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
			algorithm steps

		Raises
        ------
		RuntimeError
			If iteration limit has been exceeded
		"""

		x = to_float(initial_guess)
		self._initialize_callback(initial_guess, objective_function)
		
		for _ in range(self.max_iter):
			gradient = grad(objective_function)(x)
			x = x - learning_rate * gradient
			self._set_callback_on_step(objective_function, x)
			
			if np.linalg.norm(gradient) < precsision:
				return self.callback_data if callback else x

		raise RuntimeError("Iteration limit has been exceeded")
	
	def _initialize_callback(self, initial_guess, objective_function):
		"""
		Parameters
        ----------
		objective_function : Callable
			function to minimize

		initial_guess : Union[Sequence[float], np.ndarray, float]
			starting point of algorithm

		Information about callback_data
		----------
		callback_data["solution"] : Union[Sequence[float], np.ndarray, float]
			minimum  point of objective function

		callback_data["points"] : List[Tuple]
			Algorithm path, list consists of elements
			of the form: (point, objective_function(point))
		
		callback_data["count_of_grad_eval"] : int
			Count of evaluated gradients
		"""

		self.callback_data["solution"] = initial_guess
		self.callback_data["points"] = [(initial_guess, objective_function(initial_guess))]
		self.callback_data["count_of_grad_eval"] = 0
		
	def _set_callback_on_step(self, objective_function, x):
		"""
		Parameters
        ----------
		objective_function : Callable
			function to minimize

		x : Union[Sequence[float], np.ndarray, float]
			step of algorithm
		"""

		self.callback_data["solution"] = x
		self.callback_data["points"].append((x, objective_function(x)))
		self.callback_data["count_of_grad_eval"] += 1


class GradientDescentSteepest(BaseOptimizer):
	"""Steepest descent

	Attributes
    ----------
	max_iter : int
		max iterations, if achieved, it generates an error
	callback_data : dict
		information about optimization algorithm steps

	Methods
	----------
	optimize(objective_function, initial_guess, precsision=1e-8, callback=False, alpha_boundaries=(0, 10))
		Find minimum of objective function
	
	_initialize_callback(initial_guess, objective_function)
		Initialize data structures for optimization 
		algorithm steps

	_set_callback_on_step(objective_function, x)
		Add information about optimization 
		algorithm step
	"""

	def __init__(self, max_iter=1000):
		self.max_iter = max_iter
		self.callback_data = dict()

	def optimize(self, objective_function, initial_guess, precsision=1e-8, callback=False, alpha_boundaries=(0, 10)):
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
			return information about optimization 
			algorithm stes or not

		alpha_boundaries : tuple[Union[int, float]]
			search limits for the gradient extension parameter

		Raises
        ------
		RuntimeError
			If iteration limit has been exceeded
		"""

		x = to_float(initial_guess)
		self._initialize_callback(initial_guess, objective_function)
		optimizer = GoldenRatio()

		for _ in range(self.max_iter):
			gradient = grad(objective_function)(x)
			alpha = optimizer.optimize(lambda a: objective_function(x - a * gradient), alpha_boundaries, precsision)
			x = x - alpha * gradient
			self._set_callback_on_step(objective_function, x)

			if np.linalg.norm(gradient) < precsision:
				return self.callback_data if callback else x

		raise RuntimeError("Iteration limit has been exceeded")
	
	def _initialize_callback(self, initial_guess, objective_function):
		"""
		Parameters
        ----------
		objective_function : Callable
			function to minimize

		initial_guess : Union[Sequence[float], np.ndarray, float]
			starting point of algorithm

		Information about callback_data
		----------
		callback_data["solution"] : Union[Sequence[float], np.ndarray, float]
			minimum  point of objective function

		callback_data["points"] : List[Tuple]
			Algorithm path, list consists of elements
			of the form: (point, objective_function(point))
		
		callback_data["count_of_grad_eval"] : int
			Count of evaluated gradients
		"""

		self.callback_data["solution"] = initial_guess
		self.callback_data["points"] = [(initial_guess, objective_function(initial_guess))]
		self.callback_data["count_of_grad_eval"] = 0
		
	def _set_callback_on_step(self, objective_function, x):
		"""
		Parameters
        ----------
		objective_function : Callable
			function to minimize

		x : Union[Sequence[float], np.ndarray, float]
			step of algorithm
		"""
		self.callback_data["solution"] = x
		self.callback_data["points"].append((x, objective_function(x)))
		self.callback_data["count_of_grad_eval"] += 1


class GradientDescentCrashStep(BaseOptimizer):
	"""Gradient descent with crash step

	Attributes
    ----------
	max_iter : int
		max iterations, if achieved, it generates an error
	callback_data : dict
		information about optimization algorithm steps

	Methods
	----------
	optimize(objective_function, initial_guess, precsision=1e-8, callback=False, 
	coefficient=0.2, step=0.2, first_step=0.1)
		Find minimum of objective function
	
	_initialize_callback(initial_guess, objective_function)
		Initialize data structures for optimization 
		algorithm steps

	_set_callback_on_step(objective_function, x)
		Add information about optimization 
		algorithm step
	"""

	def __init__(self, max_iter=1000):
		self.max_iter = max_iter
		self.callback_data = dict()

	def optimize(self, objective_function, initial_guess, precsision=1e-8, callback=False, coefficient=0.1, step=0.95, first_step=1):
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
			return information about optimization 
			algorithm stes or not

		coefficient : float
			constant coefficient
		
		step : float
			crash step constant

		first_step : float
			first step value

		Raises
        ------
		RuntimeError
			If iteration limit has been exceeded
		"""

		x = to_float(initial_guess)
		self._initialize_callback(initial_guess, objective_function)
		optimizer = GoldenRatio()
		alpha = first_step
		for _ in range(self.max_iter):
			gradient = grad(objective_function)(x)
			
			while objective_function(x - alpha * gradient) >= \
            objective_function(x) - coefficient * alpha * (gradient ** 2):
				alpha *= step

			x = x - alpha * gradient
			self._set_callback_on_step(objective_function, x)

			if np.linalg.norm(gradient) < precsision:
				return self.callback_data if callback else x

		raise RuntimeError("Iteration limit has been exceeded")
	
	def _initialize_callback(self, initial_guess, objective_function):
		"""
		Parameters
        ----------
		objective_function : Callable
			function to minimize

		initial_guess : Union[Sequence[float], np.ndarray, float]
			starting point of algorithm

		Information about callback_data
		----------
		callback_data["solution"] : Union[Sequence[float], np.ndarray, float]
			minimum  point of objective function

		callback_data["points"] : List[Tuple]
			Algorithm path, list consists of elements
			of the form: (point, objective_function(point))
		
		callback_data["count_of_grad_eval"] : int
			Count of evaluated gradients
		"""

		self.callback_data["solution"] = initial_guess
		self.callback_data["points"] = [(initial_guess, objective_function(initial_guess))]
		self.callback_data["count_of_grad_eval"] = 0
		
	def _set_callback_on_step(self, objective_function, x):
		"""
		Parameters
        ----------
		objective_function : Callable
			function to minimize

		x : Union[Sequence[float], np.ndarray, float]
			step of algorithm
		"""
		self.callback_data["solution"] = x
		self.callback_data["points"].append((x, objective_function(x)))
		self.callback_data["count_of_grad_eval"] += 1