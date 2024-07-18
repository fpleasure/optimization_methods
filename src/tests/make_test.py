from src.optimizer import Optimizer

def make_test(optimizer_name, objective_function, initial_guess, precsision):
	opt = Optimizer(optimizer_name)
	x = opt.optimize(objective_function, initial_guess, precsision, callback=False)

	return x