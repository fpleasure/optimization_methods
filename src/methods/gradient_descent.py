from .base import BaseOptimizer
from .calculus import gradient

class GradientDescent(BaseOptimizer):
	def __init__(self, max_iter=1000, learning_rate=0.1):
		self.learning_rate = learning_rate
		self.max_iter = max_iter
		self.callback_data = dict()

	def optimize(self, objective_function, initial_guess, precsision=1e-8, callback=False):
		x = initial_guess
		self._initialize_callback(initial_guess)

		for _ in range(self.max_iter):
			grad = gradient(objective_function, x, precsision)
			x = x - self.learning_rate * grad
			self._set_callback_on_step(objective_function, x)
			if abs(grad) < precsision:
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