import numpy as np
from optimization_methods.optimizers.multidimensional import CyclicCoordinateDescent
from optimization_methods.plotter import OptimizationPlotter


optimizer = CyclicCoordinateDescent()
def f(x):
	return x[0] ** 2 + x[1] ** 2

result = optimizer.minimize(f, np.array([-15.0, 35.0]), tolerance=1e-2)
plot = OptimizationPlotter(result, f).plot()
plot.show()