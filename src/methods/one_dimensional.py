"""This module contains zero order optimization algorithms:
- Dichotomie
- Golden ratio
"""

from src.methods.base import BaseOptimizer
from src.methods.supporting_functions import to_float

import autograd.numpy as np


class Dichotomie(BaseOptimizer):
	"""Dichotomie algorithm

	Attributes
    ----------
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
			return information about optimization 
			algorithm stes or not

		Raises
        ------
		ValueError
			If initial guess does not have sequence type
		
		RuntimeError
			If iteration limit has been exceeded
		"""
		
		if not isinstance(initial_guess, (tuple, list, np.ndarray)):
			raise ValueError("Initital guess must be have sequence type")
		
		# TO-DO: проверка на правильно введенные данные
		a, b = to_float(initial_guess)
		self._initialize_callback((a + b)/2, objective_function)

		for _ in range(self.max_iter):
			x = (a + b)/2
			x_1 = x - precsision
			x_2 = x + precsision
			if objective_function(x_1) < objective_function(x_2):
				b = x
			else:
				a = x
			self._set_callback_on_step(objective_function, x)
			
			if abs(b - a) < precsision:
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
		
		callback_data["count_of_fubc_eval"] : int
			Count of evaluated fubctions
		"""

		self.callback_data["solution"] = initial_guess
		self.callback_data["points"] = [(initial_guess, objective_function(initial_guess))]
		self.callback_data["count_of_func_eval"] = 0
		
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
		self.callback_data["count_of_func_eval"] += 2


class GoldenRatio(BaseOptimizer):
	"""GoldenRatio algorithm

	Attributes
    ----------
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
			return information about optimization 
			algorithm stes or not

		Raises
        ------
		ValueError
			If initial guess does not have sequence type
		
		RuntimeError
			If iteration limit has been exceeded
		"""
		
		if not isinstance(initial_guess, (tuple, list, np.ndarray)):
			raise ValueError("Initital guess must be have sequence type")
		
		# TO-DO: проверка на правильно введенные данные
		a, b = to_float(initial_guess)
		phi = (1 + np.sqrt(5))/2
		self._initialize_callback((a + b)/2, objective_function)

		for _ in range(self.max_iter):
			x = (a + b)/2
			x_1 = b - (b - a)/phi
			x_2 = a + (b - a)/phi

			if objective_function(x_1) > objective_function(x_2):
				a = x_1
				x_1 = x_2
				x_2 = b - (x_1 - a)
			else:
				b = x_2
				x_2 = x_1
				x_1 = a + (b - x_2)

			self._set_callback_on_step(objective_function, x)
		
			if abs(b - a) < precsision:
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
		
		callback_data["count_of_fubc_eval"] : int
			Count of evaluated fubctions
		"""

		self.callback_data["solution"] = initial_guess
		self.callback_data["points"] = [(initial_guess, objective_function(initial_guess))]
		self.callback_data["count_of_func_eval"] = 0
		
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
		self.callback_data["count_of_func_eval"] += 2