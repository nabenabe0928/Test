import numpy as np
from numba import jit, f8, i8, b1, void
import time
import matplotlib.pyplot as plt

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

def main(T_c, W, H, lmd, thrm_cap, dt, dx, dy, f_bound, itr):
	for element in T_c.keys():
		T = np.zeros(W * H)
		psi = np.zeros(W * H)
		phi = np.zeros(W * H)
		c = - lmd[element] / thrm_cap[element] * dt
		Q_inv = get_Q_inv(W, H, dx, dy, c, f_bound, phi, psi)

		T_c[element] = get_T(itr, Q_inv, T, psi, W, H)

		plt.plot(T_c[element], label = element)
	plt.xlabel("elapsed time[0.1s]")
	plt.ylabel("temperature[K]")
	plt.legend(loc = "best")
	plt.tight_layout(pad=0.1)
	plt.savefig('figure_dynamic.png')
	plt.show()	
	

@jit(f8[:](i8,f8[:,:],f8[:],f8[:],i8,i8))
def get_T(itr, Q_inv, T, psi, W, H):
	T_c = np.array([])
	W_half = int(W / 4)
	H_half = int(H / 2)
	
	for t in range(itr):
		T = np.dot(Q_inv, T + psi)
		T_c = np.append(T_c, T[H_half * W + W_half])

		if t % 100 == 0:
			print(t)
	
	return T_c

if __name__ == "__main__":
	itr = 5000
	W = 50
	H = 50
	X = 0.1
	Y = 0.1
	dt = 0.1
	lmd = {"Au":316.0, "Hg": 8.6, "Cu": 399, "Fe": 77}
	thrm_cap = {"Au":129.0 * 19320.0, "Hg": 139.0 * 13546.0, "Cu": 385.0 * 8960.0, "Fe": 444.0 * 7874.0}
	f_bound = np.array([[0.0, 0.0], [100.0, 100]])

	dx = float( X / (W + 1) )
	dy = float( Y / (H + 1) )
	T_c = {"Au": [], "Hg": [], "Cu": [], "Fe": []}

	main(T_c, W, H, lmd, thrm_cap, dt, dx, dy, f_bound, itr)
