import time
import numpy as np
import scipy.special

def measure_time(func, mus, sgms, dx_div):
	start = time.time()
	func(mus, sgms, dx_div)
	finish = time.time()
	print("elapsed time[s]:",finish - start)

def normal_cdf(x, mu, sgm):
	idx = (x - mu) / np.sqrt(2) / sgm
	return 0.5 * (1 + scipy.special.erf(idx))

def normal_pdf(x, mu, sgm):
	c = 1.0 / np.sqrt(2 * np.pi) / sgm
	idx = - 0.5 * ((x - mu) / sgm) ** 2
	return c * np.exp(idx)

def integral(mus, sgms, dx_div):
	lower = min([mu - 3 * sgm for mu, sgm in zip(mus, sgms)])
	upper = min([mu + 3 * sgm for mu, sgm in zip(mus, sgms)])
	dx = float(1 / dx_div * (upper - lower))
	x = np.linspace(lower, upper, int(dx_div))

	F1 = normal_cdf(x, mus[0], sgms[0])
	f2 = normal_pdf(x, mus[1], sgms[1]) 	
	F3 = normal_cdf(x, mus[2], sgms[2])

	print("value:", np.sum(dx * F1 * f2 * (1 - F3)))


def monte_carlo(mus, sgms, dx_div):
	samples = [np.random.normal(loc = mu,scale = sgm,size = int(dx_div)) for mu, sgm in zip(mus, sgms)]
	print("value:",np.mean((samples[0] <= samples[1]) & (samples[1] <= samples[2])))


mus = np.array([1.1,1.2,1.3])
sgms = np.array([0.1,0.1,0.1]) 
dx_div = 1e+6

functions = [
			"integral", 
			"monte_carlo"]

for func in functions:
	print("function",func)
	measure_time(
		eval(func),
		mus, 
		sgms, 
		dx_div
		)
	print("")
