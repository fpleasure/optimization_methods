"""This module contains functions to
vizualize optimization functions
two arguments."""

import matplotlib.pyplot as plt
import autograd.numpy as np


def get_points_from_callback(callback, objective_function):
	"""Return x, y, z coordinates from
	callback_data optimizer

	Parameters
	----------
	callback : dict
		callback_data from optimizer

	objective_function : Callable
		function to minimize
	"""

	points = callback['points']
	x = [pt[0][0] for pt in points]
	y = [pt[0][1] for pt in points]
	z = sorted([objective_function(pt[0]) for pt in points])
	return x, y, z


def get_mesh_from_points(x, y, objective_function, margins=0.15, num=1000):
	"""Return mesh for contour lines
	x, y recieved from get_points_from_callback

	Parameters
	----------
	x : List
		x coordinate list from get_points_from_callback

	y : List
		y coordinate list from get_points_from_callback

	objective_function : Callable
		function to minimize

	margins : float
		margins for matplotlib

	num : int
		number of segments divisions
	"""

	x_mesh = np.linspace(min(x) - margins, max(x) + margins, num)
	y_mesh = np.linspace(min(y) - margins, max(y) + margins, num)
	x_mesh, y_mesh = np.meshgrid(x_mesh, y_mesh)
	z_mesh = objective_function([x_mesh, y_mesh])
	return x_mesh, y_mesh, z_mesh


def make_dots(x, y, linestyle='solid', color='red'):
	"""Draw path of algoritm

	Parameters
	----------
	x : List
		x coordinate list from get_points_from_callback

	y : List
		y coordinate list from get_points_from_callback

	linestyle : str
		linestyle for matplotlib

	color : str
		color for matplotlib
	"""

	plt.plot(x, y, linestyle=linestyle, c=color)


def make_contour(x_mesh, y_mesh, z_mesh, z, zorder=0):
	"""Draw a contour lines

	Parameters
	----------
	x_mesh : np.ndarray
		x coordinate mesh from get_mesh_from_points

	y_mesh : np.ndarray
		y coordinate mesh from get_mesh_from_points

	z_mesh : np.ndarray
		z coordinate mesh from get_mesh_from_points

	z : List
		z coordinate sorted list from get_points_from_callback

	zorder : str
		zorder for matplotlib
	"""

	plt.contour(x_mesh, y_mesh, z_mesh, levels=z, zorder=zorder)


def set_plot_settings(xlabel='x', ylabel='y', fontsize=10, labelpad=6):
	"""Set plot settings

	Parameters
	----------

	xlabel : str
		xlavel for matplotlib

	ylabel : str
		ylavel for matplotlib

	fontsize: Union[float, int]
		fontsize for matplotlib

	labelpad: Union[float, int]
		labelpad for matplotlib
	"""

	plt.axis('scaled')
	plt.xlabel(xlabel, fontsize=fontsize, labelpad=labelpad)
	plt.ylabel(ylabel, fontsize=fontsize, labelpad=labelpad)
	plt.xticks(fontsize=fontsize)
	plt.yticks(fontsize=fontsize)


def plot(callback, objective_function, margins=0.15, num=1000, color='red', linestyle='solid', xlabel='x', ylabel='y', fontsize=10, labelpad=6, zorder=0):
	"""Make plot in 2D

	Parameters
	----------
	callback : dict
		callback_data from optimizer

	objective_function : Callable
		function to minimize

	margins : float
		margins for matplotlib

	num : int
		number of segments divisions

	linestyle : str
		linestyle for matplotlib

	color : str
		color for matplotlib

	zorder : str
		zorder for matplotlib

	xlabel : str
		xlavel for matplotlib

	ylabel : str
		ylavel for matplotlib

	fontsize: Union[float, int]
		fontsize for matplotlib

	labelpad: Union[float, int]
		labelpad for matplotlib
	"""

	x, y, z = get_points_from_callback(callback, objective_function)
	x_mesh, y_mesh, z_mesh = get_mesh_from_points(x, y, objective_function, margins, num)
	make_contour(x_mesh, y_mesh, z_mesh, z, zorder)
	make_dots(x, y, linestyle, color)
	set_plot_settings(xlabel, ylabel, fontsize, labelpad)
	return plt