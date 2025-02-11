import numpy as np
from optimization_methods.optimizers.one_dimensional import GoldenRatio


def f(x):
	return x ** 2

optimizer = GoldenRatio()
result = optimizer.minimize(f, np.array([-1, 1]), tolerance=1e-1, max_iterations=300)
print(result)