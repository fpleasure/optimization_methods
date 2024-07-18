from src.methods.base import BaseOptimizer
from src.methods.supporting_functions import to_float

import autograd.numpy as np
from autograd import grad


class GradientDescent(BaseOptimizer):
	def __init__(self, max_iter=1000, learning_rate=0.1):
		self.learning_rate = learning_rate
		self.max_iter = max_iter
		self.callback_data = dict()

	def optimize(self, objective_function, initial_guess, precsision=1e-8, callback=False):
		x = to_float(initial_guess)
		self._initialize_callback(initial_guess)

		for _ in range(self.max_iter):
			gradient = grad(objective_function)(x)
			# print(x, np.linalg.norm(gradient))
			x = x - self.learning_rate * gradient
			self._set_callback_on_step(objective_function, x)

			if np.linalg.norm(gradient) < precsision:
				return self.callback_data if callback else x

		raise RuntimeError("Iteration limit has been exceeded")
	
	def _initialize_callback(self, initial_guess):
		self.callback_data["solution"] = initial_guess
		self.callback_data["points"] = list()
		self.callback_data["count_of_grad_eval"] = 0
		
	def _set_callback_on_step(self, objective_function, x):
		self.callback_data["solution"] = x
		self.callback_data["points"].append((x, objective_function(x)))
		self.callback_data["count_of_grad_eval"] += 1