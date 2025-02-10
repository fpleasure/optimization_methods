from abc import ABC, abstractmethod
from ..result import OptimizationResult


class Optimizer(ABC):
	"""
	Абстрактный базовый класс для оптимизаторов.
	"""

	def minimize(self, *args, **kwargs) -> OptimizationResult:
		"""
		Определяет общий процесс минимизации,
		а конкретные шаги делегирует подклассам.
		"""
		self._validate_args(*args, **kwargs)
		return self._minimize_implementation(*args, **kwargs) 

	@abstractmethod
	def _validate_args(self, *args, **kwargs) -> None:
		"""
		Общий метод валидации.
		"""
		pass

	@abstractmethod
	def _minimize_implementation(self, *args, **kwargs) -> OptimizationResult:
		"""
		Реализация минимизации в конкретных подклассах.
		"""
		pass
