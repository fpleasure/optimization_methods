import autograd.numpy as np
from autograd import grad

def to_float(input_data):
	if isinstance(input_data, (list, tuple, np.ndarray)):
		return np.array(input_data, dtype=float)
	else:
		return float(input_data)
	
