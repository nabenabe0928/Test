import numpy as np
from numba import jit, f8, i8, b1, void
import time

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

if __name__ == "__main__":
	itr = 10000
	W = 3
	H = 3
	X = 0.1
	Y = 0.1
	dt = 1.0e-1
	lmd = 316.0
	thrm_cap = 129.0 * 19320.0
	phi = np.zeros(W * H)
	f_bound = np.array([[0.0, 100.0], [0.0, 100]])

	dx = float( X / (W + 1) )
	dy = float( Y / (H + 1) )
	c = - lmd / thrm_cap * dt
	T = np.zeros(W * H)
	psi = np.zeros(W * H)
	Q_inv = get_Q_inv(W, H, dx, dy, c, f_bound, phi, psi)
	
	for t in range(itr):
		print(T)
		T = np.dot(Q_inv, T + psi)