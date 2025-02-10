import numpy as np
from autograd import grad, jacobian
from typing import Callable
from abc import abstractmethod

from .base import Optimizer
from ..result import OptimizationResult
from ..exceptions import ConvergenceError, DimensionalError
from .one_dimensional import GoldenRatio


class MultidimensionalOptimizer(Optimizer):
	"""
	Базовый класс оптимизатора для
	алгоритмов многомерной оптимизации.
	"""

	def _validate_args(self, func: Callable[[np.ndarray], float], x0: np.ndarray, lr: float=0.1, tolerance: float=1e-6, max_iterations: int=1000) -> None:
		if len(x0) < 2:
			raise DimensionalError()


# ============================================
#             Градиентные методы
# ============================================


class GradientDescent(MultidimensionalOptimizer):
	"""
	Метод градиентного спуска.
	"""

	def _minimize_implementation(self, func: Callable[[np.ndarray], float], x_init: np.ndarray, lr: float=0.1, tolerance: float=1e-6, max_iterations: int=1000) -> OptimizationResult:
		results = OptimizationResult(x_init)
		x = np.array(x_init, dtype=np.float64)

		for _ in range(max_iterations):
			gradient = grad(func)(x)
			if np.linalg.norm(gradient) < tolerance:
				results.stop_iterations(status="success", error_message=None)
				return results
			x = x - lr * gradient
			results.add_iteration(x)

		results.stop_iterations(status="error", error_message=repr(ConvergenceError()))
		raise ConvergenceError()


class SteepestDescent(MultidimensionalOptimizer):
	"""
	Метод наискорейшего спуска.
	"""

	def _minimize_implementation(self, func: Callable[[np.ndarray], float], x_init: np.ndarray, tolerance: float=1e-6, max_iterations: int=1000, parametr_range: np.ndarray=[0, 10]) -> OptimizationResult:
		results = OptimizationResult(x_init)
		x = np.array(x_init, dtype=np.float64)
		optimizer = GoldenRatio()

		for _ in range(max_iterations):
			gradient = grad(func)(x)
			if np.linalg.norm(gradient) < tolerance:
				results.stop_iterations(status="success", error_message=None)
				return results
			
			try:
				alpha = optimizer.minimize(lambda a: func(x - a * gradient), *parametr_range, tolerance=tolerance, max_iterations=max_iterations).solution
			except:
				alpha = 1
			
			x = x - alpha * gradient
			results.add_iteration(x)

		results.stop_iterations(status="error", error_message=repr(ConvergenceError()))
		raise ConvergenceError()

# ============================================
#        Методы сопряженных направлений
# ============================================


class ConjugateGradient(MultidimensionalOptimizer):
	"""
	Метод сопряженных градиентов (Флетчера-Ривса).
	"""

	def _beta(self, gradient_new: np.ndarray, gradient: np.ndarray) -> float:
		return np.dot(gradient_new, gradient_new) / np.dot(gradient, gradient)

	def _minimize_implementation(self, func: Callable[[np.ndarray], float], x_init: np.ndarray, tolerance: float=1e-6, max_iterations: int=1000, parametr_range: np.ndarray=[0, 10]) -> OptimizationResult:
		results = OptimizationResult(x_init)
		x = np.array(x_init, dtype=np.float64)
		optimizer = GoldenRatio()
		
		gradient = grad(func)(x)
		direction = -gradient
		
		for _ in range(max_iterations):
			
			try:
				alpha = optimizer.minimize(lambda a: func(x + a * direction), *parametr_range, tolerance=tolerance).solution
			except:
				alpha = 1
			
			x_new = x + alpha * direction
			results.add_iteration(x_new)
			
			gradient_new = grad(func)(x_new)
			
			if np.linalg.norm(gradient_new) < tolerance:
				results.stop_iterations(status="success", error_message=None)
				return results
			
			direction = -gradient_new + self._beta(gradient_new, gradient) * direction
			
			x, gradient = x_new, gradient_new
		
		results.stop_iterations(status="error", error_message=repr(ConvergenceError()))
		raise ConvergenceError()


class PolakRibiere(ConjugateGradient):
	"""
	Метод Полака-Рибьера.
	"""

	def _beta(self, gradient_new: np.ndarray, gradient: np.ndarray) -> float:
		return max(0, np.dot(gradient_new, gradient_new - gradient) / np.dot(gradient, gradient))



# ============================================
#               Методы Ньютона
# ============================================


class Newton(MultidimensionalOptimizer):
	"""
	Метод Ньютона.
	"""

	def _minimize_implementation(self, func: Callable[[np.ndarray], float], x_init: np.ndarray, tolerance: float=1e-6, max_iterations: int=1000) -> OptimizationResult:
		results = OptimizationResult(x_init)
		x = np.array(x_init, dtype=np.float64)

		for _ in range(max_iterations):
			gradient_general = grad(func)
			gradient = gradient_general(x)

			if np.linalg.norm(gradient) < tolerance:
				results.stop_iterations(status="success", error_message=None)
				return results

			hesse_matrix = jacobian(gradient_general)(x)
			
			try:
				direction = np.linalg.solve(hesse_matrix, gradient)
			except np.linalg.LinAlgError:
				hesse_matrix += np.eye(len(hesse_matrix)) * tolerance
				direction = np.linalg.pinv(hesse_matrix).dot(gradient)
			
			x = x - direction
			results.add_iteration(x)

		results.stop_iterations(status="error", error_message=repr(ConvergenceError()))
		raise ConvergenceError()


class ModificatedNewton(MultidimensionalOptimizer):
	"""
	Метод Ньютона с одномерной минимизацией.
	"""
	def _minimize_implementation(self, func: Callable[[np.ndarray], float], x_init: np.ndarray, tolerance: float=1e-6, max_iterations: int=1000, parametr_range: np.ndarray=[0, 10]) -> OptimizationResult:
		results = OptimizationResult(x_init)
		x = np.array(x_init, dtype=np.float64)
		optimizer = GoldenRatio()

		for _ in range(max_iterations):
			gradient_general = grad(func)
			gradient = gradient_general(x)
			if np.linalg.norm(gradient) < tolerance:
				results.stop_iterations(status="success", error_message=None)
				return results
			hesse_matrix = jacobian(gradient_general)(x)
			
			try:
				direction = np.linalg.solve(hesse_matrix, gradient)
			except np.linalg.LinAlgError:
				hesse_matrix += np.eye(len(hesse_matrix)) * tolerance
				direction = np.linalg.pinv(hesse_matrix).dot(gradient)
			
			try:
				alpha = optimizer.minimize(lambda a: func(x - a * direction), *parametr_range, tolerance=tolerance, max_iterations=max_iterations).solution
			except:
				alpha = 1
			
			x = x - alpha * direction
			results.add_iteration(x)

		results.stop_iterations(status="error", error_message=repr(ConvergenceError()))
		raise ConvergenceError()


# ============================================
#          Квази-Ньютоновские методы
# ============================================


class QuasiNewton(MultidimensionalOptimizer):
	"""
	Базовый класс оптимизатора для
	Квази-Ньютоновских методов.
	"""
	@abstractmethod
	def _get_direction_matrix(self, s: np.ndarray, y: np.ndarray, B: np.ndarray, tolerance: float) -> np.ndarray:
		pass

	def _minimize_implementation(self, func: Callable[[np.ndarray], float], x_init: np.ndarray, tolerance: float = 1e-6, max_iterations: int = 1000, parametr_range: np.ndarray=[0, 10]) -> OptimizationResult:
		results = OptimizationResult(x_init)
		x = np.array(x_init, dtype=np.float64)
		n = len(x)
		B = np.eye(n)
		optimizer = GoldenRatio()

		for _ in range(max_iterations):
			gradient_func = grad(func)
			gradient = gradient_func(x)
			if np.linalg.norm(gradient) < tolerance:
				results.stop_iterations(status="success", error_message=None)
				return results

			try:
				direction = - np.linalg.solve(B, gradient)
			except np.linalg.LinAlgError:
				B += np.eye(len(B)) * tolerance
				direction = - np.linalg.pinv(B).dot(gradient)

			try:
				alpha = optimizer.minimize(lambda a: func(x + a * direction), *parametr_range, tolerance=tolerance, max_iterations=max_iterations).solution
			except:
				alpha = 1

			x_new = x + alpha * direction
			gradient_new = gradient_func(x_new)

			s = x_new - x
			y = gradient_new - gradient
			B = self._get_direction_matrix(s, y, B, tolerance)
			x, gradient = x_new, gradient_new
			results.add_iteration(x)

		results.stop_iterations(status="error", error_message=repr(ConvergenceError()))
		raise ConvergenceError()


class BFGS(QuasiNewton):
	"""
	Метод BFGS.
	"""

	def _get_direction_matrix(self, s: np.ndarray, y: np.ndarray, B: np.ndarray, tolerance: float) -> np.ndarray:
		if np.abs(np.dot(s, y)) > tolerance:
			Bs = B @ s
			B += np.outer(y, y) / np.dot(y, s) - np.outer(Bs, Bs) / np.dot(s, Bs)
		return B


class Pawell(QuasiNewton):
	"""
	Метод Пауэлла.
	"""

	def _get_direction_matrix(self, s: np.ndarray, y: np.ndarray, B: np.ndarray, tolerance: float) -> np.ndarray:
		Bsy = s + np.dot(B, y)
		if np.abs(np.dot(Bsy, y)) > tolerance:
			B -= np.outer(Bsy, Bsy.T) / np.dot(Bsy, y)
		return B


class DFP(QuasiNewton):
	"""
	Метод ДФП.
	"""

	def _get_direction_matrix(self, s: np.ndarray, y: np.ndarray, B: np.ndarray, tolerance: float) -> np.ndarray:
		By = np.dot(B, y)
		if np.abs(np.dot(y, s)) > tolerance and np.dot(y, By) > tolerance:
			B += np.outer(s, s) / np.dot(y, s) - np.outer(By, np.dot(y, B)) / np.dot(y, By)
		return B


# ============================================
#           Методы прямого поиска
# ============================================

class CyclicCoordinateDescent(MultidimensionalOptimizer):
    """
    Метод циклического покоординатного спуска.
    """
    
    def _minimize_implementation(self, func: Callable[[np.ndarray], float], x_init: np.ndarray, tolerance: float=1e-6, max_iterations: int=1000, parametr_range: np.ndarray=[-10, 10]) -> OptimizationResult:
        results = OptimizationResult(x_init)
        x = np.array(x_init, dtype=np.float64)
        n = len(x)
        
        for _ in range(max_iterations):
            for i in range(n):
                direction = np.zeros(n)
                direction[i] = 1.0
                optimizer = GoldenRatio()
                alpha = optimizer.minimize(lambda a: func(x + a * direction), *parametr_range, tolerance=tolerance).solution
                x = x + alpha * direction
                results.add_iteration(x)
                
            if np.linalg.norm(grad(func)(x)) < tolerance:
                results.stop_iterations(status="success", error_message=None)
                return results
        
        results.stop_iterations(status="error", error_message=repr(ConvergenceError()))
        raise ConvergenceError()


class HookeJeeves(MultidimensionalOptimizer):
    """
    Метод Хука-Дживса.
    """
    
    def _minimize_implementation(self, func: Callable[[np.ndarray], float], x_init: np.ndarray, step_size: float=1.0, tolerance: float=1e-6, max_iterations: int=1000) -> OptimizationResult:
        results = OptimizationResult(x_init)
        x = np.array(x_init, dtype=np.float64)
        base_x = np.copy(x)
        
        while step_size > tolerance:
            for i in range(len(x)):
                new_x = np.copy(base_x)
                new_x[i] += step_size
                if func(new_x) < func(base_x):
                    base_x = new_x
                else:
                    new_x[i] -= 2 * step_size
                    if func(new_x) < func(base_x):
                        base_x = new_x
            
            if np.array_equal(base_x, x):
                step_size /= 2
            x = np.copy(base_x)
            results.add_iteration(x)
            
            if np.linalg.norm(grad(func)(x)) < tolerance:
                results.stop_iterations(status="success", error_message=None)
                return results
        
        results.stop_iterations(status="error", error_message=repr(ConvergenceError()))
        raise ConvergenceError()


class Rosenbrock(MultidimensionalOptimizer):
    """
    Метод Розенброка.
    """
    
    def _minimize_implementation(self, func: Callable[[np.ndarray], float], x_init: np.ndarray, tolerance: float=1e-6, max_iterations: int=1000, parametr_range: np.ndarray=[-10, 10]) -> OptimizationResult:
        results = OptimizationResult(x_init)
        x = np.array(x_init, dtype=np.float64)
        n = len(x)
        directions = np.eye(n)
        
        for _ in range(max_iterations):
            for i in range(n):
                optimizer = GoldenRatio()
                alpha = optimizer.minimize(lambda a: func(x + a * directions[i]), *parametr_range, tolerance=tolerance).solution
                x = x + alpha * directions[i]
                results.add_iteration(x)
            
            directions = np.linalg.qr(directions)[0]
            
            if np.linalg.norm(grad(func)(x)) < tolerance:
                results.stop_iterations(status="success", error_message=None)
                return results
        
        results.stop_iterations(status="error", error_message=repr(ConvergenceError()))
        raise ConvergenceError()