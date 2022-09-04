import numpy as np
import matplotlib.pyplot as plt
"""
This file plots the M-P Law. You can modify the following parameters to plot different digures.
Please notice that N <= T, that is, 0 < c <= 1.
"""

# parameters
N = 1000
T = 10000
c = N / T
var = .1

# summon the random matrix and calculate the eigenvalues
x = np.random.normal(0.0, np.sqrt(var), (N, T))
s = np.matmul(x, x.T) / T
eigs = np.linalg.eigvals(s)
# calculate the theoretical density curve
a = var * np.square(1 - np.sqrt(c))
b = var * np.square(1 + np.sqrt(c))
x = np.linspace(a, b, 100)
y = np.sqrt((b - x) * (x - a)) / (2 * np.pi * x * c * var)

# draw the figure
plt.hist(eigs, bins=N // 20, density=True)
plt.plot(x, y)
plt.show()
