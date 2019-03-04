import numpy as np
from numba import jit, f8, i8, b1, void
import time
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


@jit(f8[:,:](i8,i8,f8,f8,f8[:,:],f8[:],f8[:]))
def get_laplacian_inv(W, H, dx, dy, f_bound, phi, psi):
	L = np.zeros((W * H, W * H))
	dx_inv2 = 1.0 / (dx ** 2)
	dy_inv2 = 1.0 / (dy ** 2)
	c = - 2.0 * dx_inv2 - 2.0 * dy_inv2

	
	for x in range(W):
		for y in range(H):
			alpha = y * W + x
			L[alpha][alpha] = c
			psi[alpha] = phi[alpha]

			if 0 < x < W - 1:
				L[alpha][alpha - 1] = dx_inv2
				L[alpha][alpha + 1] = dx_inv2
			elif x == 0:
				L[alpha][alpha + 1] = dx_inv2
				psi[alpha] -= dx_inv2 * f_bound[0][0]
			elif x == W - 1:
				L[alpha][alpha - 1] = dx_inv2
				psi[alpha] -= dx_inv2 * f_bound[0][1]
			if 0 < y < H - 1:
				L[alpha][alpha - W] = dy_inv2
				L[alpha][alpha + W] = dy_inv2
			elif y == 0:
				L[alpha][alpha + W] = dy_inv2
				psi[alpha] -= dx_inv2 * f_bound[1][0]
			elif y == H - 1:
				L[alpha][alpha - W] = dy_inv2
				psi[alpha] -= dx_inv2 * f_bound[1][1]
	
	return np.linalg.inv(L)

def plot_heatmap(x, y, f):

	contour = plt.contourf(x, y, f, 100)
	plt.colorbar(contour)
	plt.axis('equal')
	plt.tight_layout(pad=0.1)
	plt.savefig('figure.png')
	plt.show()

@jit(f8[:,:](f8[:,:],i8,i8,f8[:,:]))
def get_f(f, W, H, f_bound):
	f = np.reshape(f, (H, W))
	f_return = np.array([[f_bound[1][0]] * (W + 2)]) 
	
	for f_yi in f:
		f_yi = np.append(f_yi, f_bound[0][1])
		f_yi = np.insert(f_yi, 0, f_bound[0][0])
		f_return = np.append(f_return, [f_yi], axis = 0)

	f_return = np.append(f_return, [[f_bound[1][1]] * (W + 2)], axis = 0)
	
	return f_return

if __name__ == "__main__":
	W = 50
	H = 50
	l = 0.1
	x, y = np.meshgrid(np.linspace(0, l, W + 2), np.linspace(0, l, H + 2))
	dx = float( l / (W + 1) )
	dy = float( l / (H + 1) )
	f_bound = np.array([[0.0, 0.0], [100.0, 100]])
	psi = np.zeros(W * H)
	phi = np.zeros(W * H)
	L_inv = get_laplacian_inv(W, H, dx, dy, f_bound, phi, psi)
	
	f = get_f(np.dot(L_inv, psi), W, H, f_bound)
	
	plot_heatmap(x, y, f)
