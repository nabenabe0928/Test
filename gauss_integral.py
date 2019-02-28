import time
import numpy as np
import scipy.special

def measure_time(func, lower, upper, mu, sgm, dx_div):
	start = time.time()
	func(lower, upper, mu, sgm, dx_div)
	finish = time.time()
	print("elapsed time[s]:",finish - start)

def naive_integral(lower, upper, mu, sgm, dx_div):

	dx = float(1 / dx_div * (upper - lower))
	x = np.linspace(lower, upper, int(dx_div))

	c1 = 1.0 / np.sqrt(2 * np.pi) / sgm
	idx = - 0.5 * ((x - mu) / sgm) ** 2
	

	print("value:",np.sum(dx * c1 * np.exp(idx)))

def scipy_lib(lower, upper, mu, sgm, dx_div):
	
	idx_l = (lower - mu) / np.sqrt(2) / sgm
	idx_u = (upper - mu) / np.sqrt(2) / sgm

	print("value:",0.5 * ( scipy.special.erf(idx_u) - scipy.special.erf(idx_l) ) )

def monte_carlo(lower, upper, mu, sgm, dx_div):
	samples = np.random.normal(
		loc = mu, 
		scale = sgm, 
		size = int(dx_div))
	print("value:",np.mean((lower <= samples) & (samples <= upper)))


lower = -2
upper = 2
mu = 0
sgm = 1
dx_div = 1e+3

functions = [
			"naive_integral", 
			"scipy_lib", 
			"monte_carlo"]

for func in functions:
	print("function",func)
	measure_time(
		eval(func),
		lower, 
		upper, 
		mu, 
		sgm, 
		dx_div
		)
	print("")
