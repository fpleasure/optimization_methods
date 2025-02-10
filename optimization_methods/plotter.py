import numpy as np
import matplotlib.pyplot as plt
from typing import Callable
from .result import OptimizationResult


class OptimizationPlotter:
    """
    Класс для визуализации шагов двумерной оптимизации.
    """

    def __init__(self, optimization_result: OptimizationResult, function: Callable[[np.ndarray], float]):
        self.optimization_result = optimization_result
        self.function = function

    def _get_trace_data(self):
        """
        Извлекает координаты и значения функции по траектории оптимизации.
        """
        trace = np.array(self.optimization_result.trace)
        x_vals, y_vals = trace[:, 0], trace[:, 1]
        z_vals = np.array([self.function(np.array([x, y])) for x, y in zip(x_vals, y_vals)])
        return x_vals, y_vals, z_vals

    def _get_plot_bounds(self, x_vals: np.ndarray, y_vals: np.ndarray):
        """
        Рассчитывает границы для графика с учетом небольшого отступа.
        """
        x_margin = (max(x_vals) - min(x_vals)) * 0.1
        y_margin = (max(y_vals) - min(y_vals)) * 0.1
        x_min, x_max = min(x_vals) - x_margin, max(x_vals) + x_margin
        y_min, y_max = min(y_vals) - y_margin, max(y_vals) + y_margin
        return x_min, x_max, y_min, y_max

    def _generate_meshgrid(self, x_min: float, x_max: float, y_min: float, y_max: float, grid_size: int = 100):
        """
        Генерирует сетку для контурного графика.
        """
        x = np.linspace(x_min, x_max, grid_size)
        y = np.linspace(y_min, y_max, grid_size)
        X, Y = np.meshgrid(x, y)
        return X, Y

    def _plot_contours(self, X: np.ndarray, Y: np.ndarray, z_vals: np.ndarray):
        """
        Строит контуры функции на основе сетки и значений функции.
        """
        plt.contour(X, Y, self.function([X, Y]), levels=z_vals, cmap='viridis')

    def _plot_trajectory(self, x_vals: np.ndarray, y_vals: np.ndarray):
        """
        Строит траекторию оптимизации.
        """
        plt.plot(x_vals, y_vals, marker='o', linestyle='-', color='r', markersize=5, label='Траектория', zorder=2)

    def _plot_initial_and_final_points(self, x_vals: np.ndarray, y_vals: np.ndarray):
        """
        Отображает начальную и конечную точки оптимизации.
        """
        plt.scatter(x_vals[0], y_vals[0], color='darkgreen', marker='s', s=150, label='Начальная точка', zorder=3)
        plt.scatter(x_vals[-1], y_vals[-1], color='orange', marker='*', s=150, label='Конечная точка', zorder=3)

    def plot(self):
        """
        Основной метод для построения графика.
        """
        # Получаем данные траектории
        x_vals, y_vals, z_vals = self._get_trace_data()

        # Создаем фигуру для графика
        plt.figure(figsize=(8, 6))

        # Рассчитываем границы для графика
        x_min, x_max, y_min, y_max = self._get_plot_bounds(x_vals, y_vals)

        # Генерируем сетку
        X, Y = self._generate_meshgrid(x_min, x_max, y_min, y_max)

        # Строим контуры функции
        self._plot_contours(X, Y, sorted(z_vals))

        # Строим траекторию и точки
        self._plot_trajectory(x_vals, y_vals)
        self._plot_initial_and_final_points(x_vals, y_vals)

        # Настройки графика
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.grid(True)
        
        return plt
