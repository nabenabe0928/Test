import csv
import numpy as np
from numba import jit, f8, i8
import matplotlib.pyplot as plt

def open_csv(name):
	mat = []
	with open("data/" + name + ".csv", newline = "") as f:
		reader = csv.reader(f, delimiter = ",", quotechar = '"')

		for row in reader:
			mat.append([])
			for col in row:
				mat[-1].append(float(col))

	return np.array(mat)

def plot_heatmap(x, y, fs, level, name):
#def plot_heatmap(x, y, fs, name):
	def _imp(i):
		plt.figure()
		contour = plt.contourf(x, y, fs[i], level)
		#contour = plt.contourf(x, y, fs[i], 100)
		plt.colorbar(contour)
		plt.savefig("magnetic/{}/{}_figure{}.png".format(name, name, i))

	return _imp

def plot_vector(x, y, fxs, fys):
	def _imp(i):	
		plt.figure()
		plt.quiver(x, y, fxs[i], fys[i], color = "blue", angles = "xy", scale_units = "xy")
		plt.savefig("magnetic/density/density_figure{}.png".format(i))
	return _imp

@jit(f8[:,:](f8[:],i8,i8))
def get_f(vec, W, H):
	f = np.reshape(vec, (H, W))
	f_return = np.array([[0] * (W + 2)]) 
	
	for f_i in f:
		f_i = np.append(f_i, 0)
		f_i = np.insert(f_i, 0, 0)
		f_return = np.append(f_return, [f_i], axis = 0)

	f_return = np.append(f_return, [[0] * (W + 2)], axis = 0)
	
	return f_return


if __name__ == "__main__":
	mat = open_csv("mag")
	mat1 = open_csv("jx")
	mat2 = open_csv("jy")

	W = 50
	H = 50
	X = 0.1
	Y = 0.1
	x, y = np.meshgrid(np.linspace(0, X, W + 2), np.linspace(0, Y, H + 2))
	fs = []
	fxs = []
	fys = []

	
	for m in mat:
		fs.append(get_f(m, W, H))
	v_max = np.asarray(fs).max()
	v_min = np.asarray(fs).min()
	level = np.linspace(v_min, v_max, 1000)
	
	animation_f = plot_heatmap(x, y, fs, level, "magnetic")
	#animation_f = plot_heatmap(x, y, fs, "magnetic")
	#animation_f = plot_vector(x, y, fxs, fys)
	
	for t in range(len(mat)):
		animation_f(t)
		print(t)
	
	
	for m1, m2 in zip(mat1, mat2):
		fxs.append(get_f(m1, W, H))
		fys.append(get_f(m2, W, H))
	
	#animation_f = plot_heatmap(x, y, fs, level, "magnetic")
	#animation_f = plot_heatmap(x, y, fs, "magnetic")
	animation_f = plot_vector(x, y, fxs, fys)
	for t in range(len(mat1)):
		animation_f(t)
		print(t)