import numpy as np
from typing import Callable
from .base import Optimizer
from ..result import OptimizationResult
from ..exceptions import ConvergenceError


# ============================================
#       Методы одномерной минимизации
# ============================================


class OneDimensionalOptimizer(Optimizer):
	"""
	Базовый класс оптимизатора для
	алгоритмов одномерной оптимизации
	"""

	def _validate_args(self, func: Callable[[float], float], a: float, b: float, tolerance: float=1e-6, max_iterations: int=1000) -> None:
		if a >= b:
			raise ValueError("Wrong a and b. Left bound must be < right bound.")


class GoldenRatio(OneDimensionalOptimizer):
	"""
	Метод золотого сечения.
	"""

	def _minimize_implementation(self, func: Callable[[float], float], a: float, b: float, tolerance: float=1e-6, max_iterations: int=1000) -> OptimizationResult:
		results = OptimizationResult(np.array([a, b]))
		resphi = (np.sqrt(5) - 1) / 2.
		
		while abs(b - a) > tolerance:
			x1 = b - resphi * (b - a)
			x2 = a + resphi * (b - a)
			f1, f2 = func(x1), func(x2)
			
			if f1 < f2:
				b, x2, f2 = x2, x1, f1
				x1 = a + resphi * (b - x2)
			else:
				a, x1, f1 = x1, x2, f2
				x2 = b - resphi * (b - x1)

			if results.num_iterations > max_iterations:
				results.stop_iterations(status="error", error_message=repr(ConvergenceError()),
				one_dimensional=True)
				raise ConvergenceError()
			
			results.add_iteration(np.array([a, b]))
		
		results.stop_iterations(status="success", error_message=None,
				one_dimensional=True)
		return results


class Dichotomie(OneDimensionalOptimizer):
	"""
	Метод Дихотомии.
	"""

	def _minimize_implementation(self, func: Callable[[float], float], a: float, b: float, tolerance: float=1e-6, max_iterations: int=1000) -> OptimizationResult:
		results = OptimizationResult(np.array([a, b]))
		
		while abs(b - a) > tolerance:
			x = (a + b) / 2.
			x1 = x - tolerance
			x2 = x + tolerance
			f1, f2 = func(x1), func(x2)
			
			if f1 < f2:
				b = x
			else:
				a = x

			if results.num_iterations > max_iterations:
				results.stop_iterations(status="error", error_message=repr(ConvergenceError()),
				one_dimensional=True)
				raise ConvergenceError()
			
			results.add_iteration(np.array([a, b]))
		
		results.stop_iterations(status="success", error_message=None,
				one_dimensional=True)
		return results
