import numpy as np
import matplotlib.pyplot as plt


N = 1000
T = 10000
c = N / T
var = .1

x = np.random.normal(0.0, np.sqrt(var), (N, T))
s = np.matmul(x, x.T) / T
eigs = np.linalg.eigvals(s)

a = var * np.square(1 - np.sqrt(c))
b = var * np.square(1 + np.sqrt(c))
x = np.linspace(a, b, 100)
y = np.sqrt((b-x)*(x-a)) / (2*np.pi*x*c*var)

plt.hist(eigs, bins=N//20, density=True)
plt.plot(x, y)
plt.show()
