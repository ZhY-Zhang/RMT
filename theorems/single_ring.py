import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import unitary_group
"""
This file plots the Single Ring Law. You can modify the following parameters to plot different digures.
Please notice that N <= T, that is, 0 < c <= 1.
"""

# parameters
N = 400
T = 2000
L = 1
c = N / T
r = np.power(1 - c, L / 2)

# calculate eigenvalues of the product matrix
X = np.random.normal(0.0, 1.0, (N, T))
sv = np.diag(np.linalg.svd(X)[1])
Z = np.eye(N)
for _ in range(L):
    U, V = unitary_group.rvs(N), unitary_group.rvs(N)
    Xu = np.matmul(np.matmul(U, sv), V)
    Z = np.matmul(Z, Xu)
Z1: np.ndarray = Z / np.power(T, L / 2)
ei = np.linalg.eigvals(Z1)
# calculate the mean spectrum radius (MSR)
msr = np.mean(np.abs(ei))
# calculate the theretical envelope circles and MSR circle
an = np.linspace(0, 2 * np.pi, 100)
x0, y0 = np.cos(an), np.sin(an)                # outer envelope circle
x1, y1 = r * x0, r * y0                        # inner envelope circle
x2, y2 = msr * x0, msr * y0                    # MSR circle

# draw the figure
plt.plot(x0, y0, linestyle='--', color='b')
plt.plot(x1, y1, linestyle='--', color='b')
plt.scatter(ei.real, ei.imag, 2, 'r', label="eigenvalues")
plt.plot(x2, y2, linestyle='--', color='g', label="MSR")
plt.axis('equal')
plt.legend()
plt.title("Single Ring Theorem - Eigenvalue Distribution")
plt.show()
