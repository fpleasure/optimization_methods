class OptimizationError(Exception):
    """
    Общий класс ошибок в оптимизации
    """
    pass


class ConvergenceError(OptimizationError):
    """
    Ошибка, связанная с неудачной сходимостью.
    """
    
    def __init__(self, message="Not convergence."):
        super().__init__(message)
        

class DimensionalError(OptimizationError):
    """
    Ошибка, связанная с неправильной размерностью.
    """
    
    def __init__(self, message="Wrong dimension in args."):
        super().__init__(message)
