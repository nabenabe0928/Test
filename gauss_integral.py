import numpy as np

def f(x):
	return 1.0 / np.sqrt(2 * np.pi) * np.exp(- 0.5 * x ** 2 )

lower = -2
upper = 2

# Here, you can specify how many digit you would like to get precisely.
# The bigger you set the parameter, the more precise you can get the value. 
dx_div = int(1e+5)

dx = float(1 / dx_div * (upper - lower))
domain = np.linspace(lower, upper, dx_div)

S = np.sum(dx * f(domain))

print(S)
