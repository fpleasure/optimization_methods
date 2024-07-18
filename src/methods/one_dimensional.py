from src.methods.base import BaseOptimizer
from src.methods.supporting_functions import to_float

import autograd.numpy as np


class Dichotomie(BaseOptimizer):
	def __init__(self, max_iter=1000):
		self.max_iter = max_iter
		self.callback_data = dict()

	def optimize(self, objective_function, initial_guess, precsision=1e-8, callback=False):
		if not isinstance(initial_guess, tuple):
			raise ValueError("Initital guess must be have tuple type")
		# TO-DO: проверка на правильно введенные данные
		a, b = to_float(initial_guess)
		min_x = max_x = 0.
		self._initialize_callback((a + b) / 2)

		for _ in range(self.max_iter):
			x = (a + b) / 2
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
	
	def _initialize_callback(self, initial_guess):
		self.callback_data["solution"] = initial_guess
		self.callback_data["points"] = list()
		self.callback_data["count_of_func_eval"] = 0
		
	def _set_callback_on_step(self, objective_function, x):
		self.callback_data["solution"] = x
		self.callback_data["points"].append((x, objective_function(x)))
		self.callback_data["count_of_func_eval"] += 2

class GoldenRatio(BaseOptimizer):
	def __init__(self, max_iter=1000):
		self.max_iter = max_iter
		self.callback_data = dict()

	def optimize(self, objective_function, initial_guess, precsision=1e-8, callback=False):
		if not isinstance(initial_guess, tuple):
			raise ValueError("Initital guess must be have tuple type")
		
		# TO-DO: проверка на правильно введенные данные
		a, b = to_float(initial_guess)
		phi = (1 + np.sqrt(5)) / 2
		self._initialize_callback((a + b) / 2)

		for _ in range(self.max_iter):
			x = (a + b) / 2
			x_1 = b - (b - a) / phi
			x_2 = a + (b - a) / phi

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
	
	def _initialize_callback(self, initial_guess):
		self.callback_data["solution"] = initial_guess
		self.callback_data["points"] = list()
		self.callback_data["count_of_func_eval"] = 0
		
	def _set_callback_on_step(self, objective_function, x):
		self.callback_data["solution"] = x
		self.callback_data["points"].append((x, objective_function(x)))
		self.callback_data["count_of_func_eval"] += 2