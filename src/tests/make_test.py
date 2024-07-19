"""This module contains helper 
functions for tests"""

from src.optimizer import Optimizer


def make_test(optimizer_name, objective_function, initial_guess, precsision):
	"""Return minimun of objective function
	
	Parameters
	----------
	objective_function : Callable
		function to minimize

	initial_guess : Union[Sequence[float], np.ndarray, float]
		starting point of algorithm

	precsision : float
		point location accuracy
	"""
	opt = Optimizer(optimizer_name)
	x = opt.optimize(objective_function, initial_guess, precsision, callback=False)

	return x