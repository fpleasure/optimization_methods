import numpy as np
from optimization_methods.optimizers.multidimensional import Newton
from optimization_methods.plotter import OptimizationPlotter


optimizer = Newton()

def f(x):
	return 3 * x[0] ** 2 + 5 * x[1] ** 2

result = optimizer.minimize(f, np.array([-5.0, 2.0]), tolerance=1e-2)
print(result)
plot = OptimizationPlotter(result, f).plot()
plot.show()