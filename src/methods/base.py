from abc import ABC, abstractmethod

class BaseOptimizer(ABC):
    @abstractmethod
    def optimize(self, objective_function, initial_guess, precsision=1e-8, callback=False):
        pass
    
    @abstractmethod
    def _initialize_callback(self, initial_guess):
        pass
    
    @abstractmethod
    def _set_callback_on_step(self, objective_function, x):
        pass