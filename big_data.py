from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import unitary_group


def argumented_matrix_generator(Dg: np.ndarray, Df: np.ndarray, m: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """The shape of status matrix "Dg" is (N1, T).
    The shape of factor matrix "Df" is (N2, T).
    The amplitude of noise "m" is 1.0 by default."""
    # check the shape of status and factor matrices
    if Dg.shape[1] != Df.shape[1]:
        raise ValueError("The numbers of samplings of status and factors are different.")
    # step 1: duplicate the factor matrix to match the size of status matrix
    k = Dg.shape[0] // Df.shape[0]
    Dc = np.tile(Df, (k, 1))
    # step 2: add random noise to lower the relativity of the elements of the duplicated matrix
    N = np.random.normal(0.0, 1.0, size=Dc.shape)
    Ef = Dc + m * N
    # calculate signal to noise ratio if necessary
    # s2n_ratio = np.trace(np.matmul(Ef, Ef.T)) / np.trace(np.matmul(N, N.T)) / (m * m)
    # step 3: generate the argumented matrix "A" and reference argumented matrix "An"
    A = np.concatenate((Dg, Ef), axis=0)
    An = np.concatenate((Dg, N), axis=0)
    return A, An


def single_ring(X: np.ndarray, L: int = 1) -> float:
    """The parameter "X" is the data matrix, whose shape is (N, T).
    The parameter "L" is the number of singular value equivalent matrix (matrices).
    The function returns the Mean Spectrum Radius (MSR) of the product matrix Z."""
    N, T = X.shape[0], X.shape[1]
    c = N / T
    if c > 1:
        raise ValueError("The shape of data matrix must satisfy N <= T.")
    # step 1: normalize the data matrix X row by row
    # Here I use transpose function for readability.
    X1 = X.T
    X2 = (X1 - np.mean(X1, axis=0)) / np.std(X1, axis=0)
    Xn = X2.T
    # step 2: calculate the singular values of the normalized data matrix Xn and arrange them as a diagonal matrix
    sv = np.linalg.svd(Xn)[1]
    sv = np.diag(sv)
    # step 3: calculate the matrix Z, which is the product of SVE matrices of the normalized data matrix Xn
    Z = np.eye(N)
    for _ in range(L):
        U, V = unitary_group.rvs(N), unitary_group.rvs(N)
        Xu = np.matmul(np.matmul(U, sv), V)
        Z = np.matmul(Z, Xu)
    # step 4: normalize the product matrix Z
    Zn = (Z - np.mean(Z)) / np.std(Z) / np.sqrt(N)
    # step 5: find the spectrum of normalized product matrix Zn and calculate the MSR
    ei = np.linalg.eigvals(Zn)
    msr = np.mean(np.abs(ei))
    return msr


if __name__ == "__main__":
    X = np.random.normal(12.34, 65.43, size=(400, 2000))
    msr = single_ring(X, 4)
    print(msr)
