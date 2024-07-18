import matplotlib.pyplot as plt
import autograd.numpy as np

def get_points_from_callback(callback, objective_function):
	points = callback['points']
	x = [pt[0][0] for pt in points]
	y = [pt[0][1] for pt in points]
	z = sorted([objective_function(pt[0]) for pt in points])
	return x, y, z

def get_mesh_from_points(x, y, objective_function, margins=0.15, num=1000):
	x_mesh = np.linspace(min(x) - margins, max(x) + margins, num)
	y_mesh = np.linspace(min(y) - margins, max(y) + margins, num)
	x_mesh, y_mesh = np.meshgrid(x_mesh, y_mesh)
	z_mesh = objective_function([x_mesh, y_mesh])
	return x_mesh, y_mesh, z_mesh

def make_dots(x, y, linestyle, color):
	plt.plot(x, y, linestyle=linestyle, c=color)

def make_contour(x_mesh, y_mesh, z_mesh, z, zorder):
	plt.contour(x_mesh, y_mesh, z_mesh, levels=z, zorder=zorder)

def set_plot_settings(xlabel, ylabel, fontsize, labelpad):
	plt.axis('scaled')
	plt.xlabel(xlabel, fontsize=fontsize, labelpad=labelpad)
	plt.ylabel(ylabel, fontsize=fontsize, labelpad=labelpad)
	plt.xticks(fontsize=fontsize)
	plt.yticks(fontsize=fontsize)

def plot(callback, objective_function, margins=0.15, num=1000, color='red', linestyle='solid', xlabel='x', ylabel='y', fontsize=10, labelpad=6, zorder=0):
	x, y, z = get_points_from_callback(callback, objective_function)
	x_mesh, y_mesh, z_mesh = get_mesh_from_points(x, y, objective_function, margins, num)
	make_contour(x_mesh, y_mesh, z_mesh, z, zorder)
	make_dots(x, y, linestyle, color)
	set_plot_settings(xlabel, ylabel, fontsize, labelpad)
	return plt