from methods import gradient_descent

class Optimizer:
    def __init__(self, optimizer_name: str, max_iter=1000, learning_rate=0.1, interval=(0, 1)):
        if optimizer_name == "gradient_descent":
            self._optimizer = gradient_descent.GradientDescent(max_iter, learning_rate)
        else:
            raise ValueError(f"Optimizer with name {optimizer_name} does not exist!")
        

    def optimize(self, objective_function, initial_guess, precsision=1e-8, callback=False):
        return self._optimizer.optimize(objective_function, initial_guess, precsision, callback)