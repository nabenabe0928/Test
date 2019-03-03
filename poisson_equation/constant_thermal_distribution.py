import numpy as np
from numba import jit, f8, i8, b1, void
import time

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

if __name__ == "__main__":
	W = 3
	H = 3
	dx = float( 0.1 / (W + 1) )
	dy = float( 0.1 / (H + 1) )
	f_bound = np.array([[0.0, 100.0], [0.0, 100]])
	psi = np.zeros(W * H)
	phi = np.zeros(W * H)
	L_inv = get_laplacian_inv(W, H, dx, dy, f_bound, phi, psi)
	
	print(np.dot(L_inv, psi))