"""This module contains abstract class for optimizer."""

from abc import ABC, abstractmethod


class BaseOptimizer(ABC):
	"""Abstract class for optimizers

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

	@abstractmethod
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
		"""
		pass

	@abstractmethod
	def _initialize_callback(self, initial_guess, objective_function):
		"""
		Parameters
        ----------
		objective_function : Callable
			function to minimize

		initial_guess : Union[Sequence[float], np.ndarray, float]
			starting point of algorithm
		"""
		pass

	@abstractmethod
	def _set_callback_on_step(self, objective_function, x):
		"""
		Parameters
		----------
		objective_function : Callable
			function to minimize

		x : Union[Sequence[float], np.ndarray, float]
			step of algorithm
		"""
		pass