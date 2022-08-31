import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import unitary_group

# parameters
N = 400
T = 2000
c = N / T
L = 4
r = np.power(1-c, L/2)

# draw inner and outer circles
an = np.linspace(0, 2*np.pi, 100)
x = np.cos(an)
y = np.sin(an)
plt.plot(x, y, linestyle='--', color='b')
x = r * x
y = r * y
plt.plot(x, y, linestyle='--', color='b')

# draw eigenvalues on the complex plane
X = np.random.normal(0.0, 1.0, (N, T))
sv = np.diag(np.linalg.svd(X)[1])
Z = np.eye(N)
for _ in range(L):
    U, V = unitary_group.rvs(N), unitary_group.rvs(N)
    Xu = np.matmul(np.matmul(U, sv), V)
    Z = np.matmul(Z, Xu)
Z1: np.ndarray = Z / np.power(T, L/2)
ei = np.linalg.eigvals(Z1)
msr = np.mean(np.abs(ei))

print(msr)

plt.scatter(ei.real, ei.imag, 2, 'r')
plt.axis('equal')
plt.show()
