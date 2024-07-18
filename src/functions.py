def rosenbrock(x, a=1, b=100):
	# min in (1, 1) = 0
	return (a - x[0] ** 2) + b * (x[1] - x[0] ** 2) ** 2

def beale(x, a=1.5, b=2.25, c=2.625):
	# min in (3, 0.5) = 0
	return (a - x[0] + x[0] * x[1]) ** 2 + (b - x[0] + x[0] * (x[1] ** 2)) ** 2 + (c - x[0] + x[0] * (x[1] ** 3)) ** 2

def matyas(x, a=0.26, b=0.48):
	# min in (0, 0) = 0
	return a * (x[0] ** 2 + x[1] ** 2) - b * x[0] * x[1]
