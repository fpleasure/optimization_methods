"""This module contains helper 
functions for methods."""

import autograd.numpy as np
from autograd import grad


def to_float(input_data):
	"""If input data is not float, return 
	float input data if input data is int or
	return float np.array if input data is sequence
	
	Parameters
    ----------
	input_data : Union[int, List[int], Tuple[int]]
		data to floatting
	"""
	if isinstance(input_data, (list, tuple, np.ndarray)):
		return np.array(input_data, dtype=float)
	else:
		return float(input_data)
	
