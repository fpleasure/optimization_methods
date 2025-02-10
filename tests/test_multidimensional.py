from abc import ABC
import unittest
import numpy as np
from optimization_methods.optimizers.multidimensional import *
from optimization_methods.functions import rosenbrock


class BaseTestsForMultidimensional(unittest.TestCase, ABC):
	optimizer = None

	@classmethod
	def setUpClass(cls):
		"""
		Пропуск тестов в абстрактном базовом классе
		"""
		if cls is BaseTestsForMultidimensional:
			raise unittest.SkipTest("Пропуск тестов в абстрактном базовом классе")

	def test_on_quadratic_function_two_dimensional(self):
		"""
		Тест на квадратичной функции x^2 + y^2.
		"""
		tolerance = 1e-4

		def quadratic(x):
			return x[0] ** 2 + x[1] ** 2

		result = self.optimizer.minimize(quadratic, np.array([-2.0, 1.0]), tolerance=tolerance)
		self.assertTrue(abs(quadratic(result.solution)) < tolerance, f"{result.solution}")

	def test_on_quadratic_function_three_dimensional(self):
		"""
		Тест на квадратичной функции x^2 + y^2 + z^2.
		"""
		tolerance = 1e-4

		def quadratic(x):
			return x[0] ** 2 + x[1] ** 2 + x[2] ** 2

		result = self.optimizer.minimize(quadratic, np.array([-2.0, 1.0, -1.]), tolerance=tolerance)
		self.assertTrue(abs(quadratic(result.solution)) < tolerance, f"{result.solution}")

	def test_with_different_initial_points(self):
		"""
		Проверка работы алгоритма с разными начальными точками.
		"""
		tolerance = 1e-4

		def quadratic(x):
			return x[0] ** 2 + x[1] ** 2

		start_points = [np.array([5.0, -3.0]), np.array([-1.0, 4.0]), np.array([10.0, 10.0])]

		for start in start_points:
			with self.subTest(start=start):
				result = self.optimizer.minimize(quadratic, start, tolerance=tolerance)
				self.assertTrue(abs(quadratic(result.solution)) < tolerance, f"Failed at {start}")


class GradientDescentTest(BaseTestsForMultidimensional):
    optimizer = GradientDescent()


class SteepestDescentTest(BaseTestsForMultidimensional):
    optimizer = SteepestDescent()


class NewtonTest(BaseTestsForMultidimensional):
    optimizer = Newton()


class ModificatedNewtonTest(BaseTestsForMultidimensional):
    optimizer = ModificatedNewton()


class BFGSTest(BaseTestsForMultidimensional):
    optimizer = BFGS()


class PawellTest(BaseTestsForMultidimensional):
	optimizer = Pawell()


class DFPTest(BaseTestsForMultidimensional):
	optimizer = DFP()


class ConjugateGradient(BaseTestsForMultidimensional):
	optimizer = ConjugateGradient()


class PolakRibiere(BaseTestsForMultidimensional):
	optimizer = PolakRibiere()


class CyclicCoordinateDescent(BaseTestsForMultidimensional):
	optimizer = CyclicCoordinateDescent()


class HookeJeeves(BaseTestsForMultidimensional):
	optimizer = HookeJeeves()


class Rosenbrock(BaseTestsForMultidimensional):
	optimizer = Rosenbrock()


if __name__ == "__main__":
    unittest.main()
