import numpy as np
from numba import jit, f8, i8, b1, void, u1
import time
import sys
import csv
import matplotlib.pyplot as plt
import matplotlib.animation as animation

@jit(f8[:,:](i8,i8,f8,f8,f8,f8))
def get_surface_potential(W, H, dx, dy, d, mu0):
	sur_mag = np.zeros((W * H, W * H))
	coef = mu0 * d * dx * dy / 4.0 / np.pi

	for i1 in range(W):
		for j1 in range(H):
			alpha1 = j1 * W + i1
			r1 = np.array([i1 * dx, j1 * dy, 0])

			for i2 in range(W):
				for j2 in range(H):
					alpha2 = j2 * W + i2
					r2 = np.array([i2 * dx, j2 * dy, d / 2.0])

					sur_mag[alpha1][alpha2] =  coef / ( np.linalg.norm(r1 - r2) ** 3 ) 
	
	return sur_mag

@jit(f8[:,:](i8,i8,f8,f8,f8))
def get_delta(W, H, dx, dy, eta):
	delta = np.zeros((W * H, W * H))
	dx_inv2 = eta * 1.0 / (dx ** 2)
	dy_inv2 = eta * 1.0 / (dy ** 2)
	c = - 2.0 * dx_inv2 - 2.0 * dy_inv2

	
	for i in range(W):
		for j in range(H):
			alpha = j * W + i
			delta[alpha][alpha] = c
			
			if 0 < i < W - 1:
				delta[alpha][alpha - 1] = dx_inv2
				delta[alpha][alpha + 1] = dx_inv2
			elif i == 0:
				delta[alpha][alpha + 1] = dx_inv2
			elif i == W - 1:
				delta[alpha][alpha - 1] = dx_inv2
			if 0 < j < H - 1:
				delta[alpha][alpha - W] = dy_inv2
				delta[alpha][alpha + W] = dy_inv2
			elif j == 0:
				delta[alpha][alpha + W] = dy_inv2
			elif j == H - 1:
				delta[alpha][alpha - W] = dy_inv2
	
	return delta

@jit(f8[:,:](i8,i8,f8))
def get_nablax(W, H, dx):
	dx_inv = 1.0 / dx / 2.0
	nablax = np.zeros((W * H, W * H))

	for i in range(W):
		for j in range(H):
			alpha = j * W + i
			
			if 0 < i < W - 1:
				nablax[alpha][alpha - 1] = - dx_inv
				nablax[alpha][alpha + 1] = dx_inv
			elif i == 0:
				nablax[alpha][alpha + 1] = dx_inv
			elif i == W - 1:
				nablax[alpha][alpha - 1] = - dx_inv

	return nablax

@jit(f8[:,:](i8,i8,f8))
def get_nablay(W, H, dy):
	dy_inv = 1.0 / dy / 2.0
	nablay = np.zeros((W * H, W * H))

	for i in range(W):
		for j in range(H):
			alpha = j * W + i
			
			if 0 < j < H - 1:
				nablay[alpha][alpha - W] = - dy_inv
				nablay[alpha][alpha + W] = dy_inv
			elif j == 0:
				nablay[alpha][alpha + W] = dy_inv
			elif j == H - 1:
				nablay[alpha][alpha - W] = - dy_inv

	return nablay

@jit(void(f8[:,:],f8[:,:],f8[:,:],f8[:,:],i8,f8,i8,i8,f8))
def main(mat1, mat2, nablax, nablay, itr, f, W, H, dt):
	coef = 2 * np.pi * f
	mag = np.zeros(W * H)
	Jx = []
	Jy = []
	M = []


	for t in range(itr):
		B = coef * np.sin(coef * t * dt)
		mag = mat1 @ ( mat2 @ mag - np.full(W * H, B) )
		M.append(mag[:])
		Jx.append(nablay @ mag)
		Jy.append(- nablax @ mag)
		print(t)

	write_csv(M, "mag")
	write_csv(Jx, "jx")
	write_csv(Jy, "jy")

def write_csv(M, name):
	with open("data/" + name + ".csv", "w", newline = "") as f:
		writer = csv.writer(f, delimiter = ",", quotechar = '"')

		for array in M:
			writer.writerow(array)

if __name__ == "__main__":
	f = 1.0
	itr = 50
	W = 50
	H = 50
	X = 0.1
	Y = 0.1
	d = 0.01
	eta = 2.0 * 1.0e-8
	mu0 = 4.0 * np.pi * 1.0e-7
	dt = 0.1
	x, y = np.meshgrid(np.linspace(0, X, W + 2), np.linspace(0, Y, H + 2))
	
	dx = float( X / (W + 1) )
	dy = float( Y / (H + 1) )

	sur_mag = get_surface_potential(W, H, dx, dy, d, mu0)
	delta = get_delta(W, H, dx, dy, eta)
	mag_coef = mu0 * np.identity(W * H)
	nablax = get_nablax(W, H, dx)
	nablay = get_nablay(W, H, dy)

	mat1 = np.linalg.inv( (mag_coef - sur_mag) / dt - delta )
	mat2 = (mag_coef - sur_mag) / dt

	main(mat1, mat2, nablax, nablay, itr, f, W, H, dt)