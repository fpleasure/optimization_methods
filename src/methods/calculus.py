import numpy as np


def derivative(objective_function, x, precsision=1e-8):
	return (objective_function(x + precsision) - objective_function(x -  precsision)) / (2 * precsision)

def gradient(objective_function, x, precsision=1e-8):
	if isinstance(x, (int, float)):
		return derivative(objective_function, x, precsision)
	
	vector = np.array([0.] * len(x))

	for i in range(len(x)):
		def function(pt):
			axis_vector = [0] * len(x)
			axis_vector[i] = pt
			
			return objective_function(axis_vector)[i]
		
		vector[i] = derivative(function, x[i], precsision)

	return vector