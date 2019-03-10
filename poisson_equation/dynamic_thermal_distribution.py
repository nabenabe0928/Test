import numpy as np
from numba import jit, f8, i8, b1, void, u1
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation

@jit(f8[:,:](i8,i8,f8,f8,f8,f8[:,:],f8[:],f8[:]))
def get_Q_inv(W, H, dx, dy, coef, f_bound, phi, psi):
	Q = np.zeros((W * H, W * H))
	dx_inv2 = 1.0 / (dx ** 2)
	dy_inv2 = 1.0 / (dy ** 2)
	c = - 2.0 * dx_inv2 - 2.0 * dy_inv2

	
	for x in range(W):
		for y in range(H):
			alpha = y * W + x
			Q[alpha][alpha] = c
			psi[alpha] = phi[alpha]

			if 0 < x < W - 1:
				Q[alpha][alpha - 1] = dx_inv2
				Q[alpha][alpha + 1] = dx_inv2
			elif x == 0:
				Q[alpha][alpha + 1] = dx_inv2
				psi[alpha] -= dx_inv2 * f_bound[0][0]
			elif x == W - 1:
				Q[alpha][alpha - 1] = dx_inv2
				psi[alpha] -= dx_inv2 * f_bound[0][1]
			if 0 < y < H - 1:
				Q[alpha][alpha - W] = dy_inv2
				Q[alpha][alpha + W] = dy_inv2
			elif y == 0:
				Q[alpha][alpha + W] = dy_inv2
				psi[alpha] -= dy_inv2 * f_bound[1][0]
			elif y == H - 1:
				Q[alpha][alpha - W] = dy_inv2
				psi[alpha] -= dy_inv2 * f_bound[1][1]

	psi *= coef
	
	return np.linalg.inv(coef * Q + np.identity(W * H))

def plot_heatmap(x, y, fs, element):
	def _imp(i):
		print(i)
		plt.figure()
		contour = plt.contourf(x, y, fs[i], 100)
		plt.colorbar(contour)
		plt.savefig("dynamic_{}_img/dynamic_figure{}.png".format(element, i))

	return _imp
	

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

def main(T_c, W, H, lmd, thrm_cap, dt, dx, dy, f_bound, itr, x, y):
	for element in T_c.keys():
		T = np.zeros(W * H)
		psi = np.zeros(W * H)
		phi = np.zeros(W * H)
		c = - lmd[element] / thrm_cap[element] * dt
		Q_inv = get_Q_inv(W, H, dx, dy, c, f_bound, phi, psi)

		get_T(itr, Q_inv, T, psi, W, H,f_bound, x, y, element)	
	

@jit(void(i8,f8[:,:],f8[:],f8[:],i8,i8,f8[:,:],f8[:],f8[:],u1))
def get_T(itr, Q_inv, T, psi, W, H, f_bound, x, y, element):
	Ts = [T]
	T_plot = [get_f(np.dot(Q_inv, Ts[-1] + psi), W, H, f_bound)]

	for t in range(itr):
		Ts.append(np.dot(Q_inv, Ts[-1] + psi))
		T_plot.append(get_f(Ts[-1], W, H, f_bound))

		if t % 100 == 0:
			print(t)
	
	animation_f = plot_heatmap(x, y, T_plot, element)
	for t in range(itr):
		animation_f(t)
	
	#fig = plt.figure()
	#ani = animation.FuncAnimation(fig, animation_f, itr, interval = 100, blit = False)
	#ani.save("dynamic_thermal_transition.gif", writer = "imagemagick")
	#plt.show()

if __name__ == "__main__":
	itr = 500
	W = 50
	H = 50
	X = 0.1
	Y = 0.1
	dt = 0.1
	x, y = np.meshgrid(np.linspace(0, X, W + 2), np.linspace(0, Y, H + 2))
	lmd = {"Au":316.0, "Hg": 8.6, "Cu": 399, "Fe": 77}
	thrm_cap = {"Au":129.0 * 19320.0, "Hg": 139.0 * 13546.0, "Cu": 385.0 * 8960.0, "Fe": 444.0 * 7874.0}
	f_bound = np.array([[0.0, 0.0], [100.0, 100]])

	dx = float( X / (W + 1) )
	dy = float( Y / (H + 1) )
	T_c = {"Au": [], "Hg": [], "Cu": [], "Fe": []}

	main(T_c, W, H, lmd, thrm_cap, dt, dx, dy, f_bound, itr, x, y)