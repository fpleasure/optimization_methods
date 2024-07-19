"""This module contains base test functions."""


def rosenbrock(x, a=1, b=100):
	"""
	For a=1, b=100:
		Minimum point: (1, 1)
		Minimum value: 0
	"""
	
	return (a - x[0]**2) + b*(x[1] - x[0]**2)**2


def beale(x, a=1.5, b=2.25, c=2.625):
	"""
	For a=1.5, b=2.25, c=2.625:
		Minimum point: (3, 0.5)
		Minimum value: 0
	"""

	return (a - x[0] + x[0]*x[1])**2 + (b - x[0] + x[0]*(x[1]**2))**2 + (c - x[0] + x[0]*(x[1]**3))**2


def matyas(x, a=0.26, b=0.48):
	"""
	For a=0.26, b=0.48:
		Minimum point: (0, 0)
		Minimum value: 0
	"""

	return a*(x[0]**2 + x[1]**2) - b*x[0]*x[1]
