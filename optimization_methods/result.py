import numpy as np


class OptimizationResult:
	"""
	Класс для хранения результатов оптимизации.
	"""

	def __init__(self, x_initial: np.ndarray) -> None:
		if not isinstance(x_initial, np.ndarray):
			raise ValueError("Wrong point type! All points must have numpy array type.")
		self.solution = None
		self.num_iterations = 0
		self.trace = [x_initial, ]
		self.status = "Create solution object."
		self.error = None

	def add_iteration(self, point: np.ndarray, gradient: np.ndarray=None, hesse: np.ndarray=None, num_func_call: int=None):
		if not isinstance(point, np.ndarray):
			raise ValueError("Wrong point type! All points must have numpy array type.")
		
		self.trace.append(point)
		self.num_iterations += 1
		self.status = f"Make {self.num_iterations} iteration."

	def stop_iterations(self, status: str="success", error_message: str=None, one_dimensional: bool=False):
		if status == "success":
			if not one_dimensional:
				self.solution = self.trace[-1]
			else:
				a, b = self.trace[-1]
				self.solution = (a + b) / 2
			self.status = f"Optimization successfully stoped."
		elif status == "error":
			self.status = f"Error in optimization."
			self.error = error_message
		else:
			raise ValueError("Wrong status! Status has 2 state: success and error.")

	def __repr__(self):
		return (f"OptimizationResult(solution={self.solution}, "
				f"iterations={self.num_iterations}, trace={self.trace}, "
				f"status={self.status})")

	def __str__(self):
		return (f"Optimization result:\nsolution: {self.solution}\n"
				f"iterations: {self.num_iterations}\n"
				f"trace: {self.trace}\nstatus: {self.status}")
