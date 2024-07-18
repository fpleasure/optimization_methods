import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.methods import gradient_descent, one_dimensional


class Optimizer:
    def __init__(self, optimizer_name: str, max_iter=1000, learning_rate=0.1, interval=(0, 1)):
        if optimizer_name == "gradient_descent":
            self._optimizer = gradient_descent.GradientDescent(max_iter, learning_rate)
        elif optimizer_name == "dichotomie":
            self._optimizer = one_dimensional.Dichotomie(max_iter)
        elif optimizer_name == "golden_ratio":
            self._optimizer = one_dimensional.GoldenRatio(max_iter)
        else:
            raise ValueError(f"Optimizer with name '{optimizer_name}' does not exist!")

    def optimize(self, objective_function, initial_guess, precsision=1e-8, callback=False):
        return self._optimizer.optimize(objective_function, initial_guess, precsision, callback)