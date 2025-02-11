from abc import ABC
import unittest
import numpy as np
from optimization_methods.optimizers.one_dimensional import GoldenRatio, Dichotomie


class BaseTestsForOneDimensional(unittest.TestCase, ABC):
	optimizer = None

	@classmethod
	def setUpClass(cls):
		"""
		Пропуск тестов в абстрактном базовом классе
		"""
		if cls is BaseTestsForOneDimensional:
			raise unittest.SkipTest("Пропуск тестов в абстрактном базовом классе")

	def test_on_quadratic_function(self):
		"""Тест на квадратичной функции f(x) = x^2"""
		tolerance = 1e-3
		def quadratic(x):
			return x ** 2

		result = self.optimizer.minimize(quadratic, np.array([-1, 1]), tolerance=tolerance)

		self.assertTrue(abs(quadratic(result.solution)) < tolerance)

	def test_on_cubic_function(self):
		"""Тест на кубической функции f(x) = (x-1)^3"""
		tolerance = 1e-3
		def cubic(x):
			return (x - 1) ** 3

		result = self.optimizer.minimize(cubic, np.array([0, 2]), tolerance=tolerance)

		self.assertAlmostEqual(result.solution, 0, delta=tolerance)

	def test_with_different_interval(self):
		"""Тест с другим интервалом (-5, 5)"""
		tolerance = 1e-3
		def quadratic(x):
			return (x - 2) ** 2

		result = self.optimizer.minimize(quadratic, np.array([-5, 5]), tolerance=tolerance)

		self.assertAlmostEqual(result.solution, 2, delta=tolerance)

	def test_convergence_with_lower_tolerance(self):
		"""Тест на сходимость при уменьшении tolerance"""
		tolerance_high = 1e-5
		tolerance_low = 1e-2

		def quadratic(x):
			return x ** 2

		result_low = self.optimizer.minimize(quadratic, np.array([-2, 1]), tolerance=tolerance_low)
		result_high = self.optimizer.minimize(quadratic, np.array([-2, 1]), tolerance=tolerance_high)

		self.assertTrue(abs(quadratic(result_high.solution)) < tolerance_high)
		self.assertTrue(abs(quadratic(result_low.solution)) < tolerance_low)
		self.assertLess(abs(result_high.solution), abs(result_low.solution))


class GoldenRatioTest(BaseTestsForOneDimensional):
    optimizer = GoldenRatio()


class DichotomieTest(BaseTestsForOneDimensional):
	optimizer = Dichotomie()


if __name__ == "__main__":
    unittest.main()
