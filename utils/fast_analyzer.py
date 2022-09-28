from typing import Tuple

import numpy as np
from scipy.stats import unitary_group
"""
This file contains the speed boosted version of some analysis functions. Some
of them lack type checking and may cause unexpected errors. Check the input
parameters carefully before using them.
"""


def single_ring(X: np.ndarray, U: np.ndarray) -> float:
    """
    This function is a speed boosted version of the "single_ring" function. It
    doesn't check the size of input matrices and doesn't support multiple
    singular value equivalent matrices.
    """
    # step 1: normalize the data matrix X row by row
    X1 = X.T
    X2 = (X1 - np.mean(X1, axis=0)) / np.std(X1, axis=0)
    np.nan_to_num(X2, copy=False)
    Xn = X2.T
    # step 2: calculate the singular values of the Xn and arrange them as a diagonal matrix
    sv = np.diag(np.linalg.svd(Xn)[1])
    # step 3: calculate the SVE matrix of Xn
    Xu = np.matmul(sv, U)
    # step 4: normalize Xu
    Zn = (Xu - np.mean(Xu)) / np.std(Xu) / np.sqrt(X.shape[0])
    # step 5: calculate the spectrum of Xu and the MSR
    ei = np.linalg.eigvals(Zn)
    msr = np.mean(np.abs(ei))
    return msr, Xn


def get_msr_array(sync_data: np.ndarray, Tw: int, expected_size: int, amplitude: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    This function is a speed boosted version of the "get_msr_array" function.
    """
    # calculate necessary arguments
    num_windows = sync_data.shape[1] - Tw + 1
    msrs = np.empty(num_windows)
    dup_n = expected_size // sync_data.shape[0]
    dup_t = int(np.ceil(expected_size / Tw))
    N = dup_n * sync_data.shape[0]
    T = dup_t * Tw
    U = unitary_group.rvs(N)
    noise = np.random.normal(0.0, 1.0, size=(N, T))
    unified_data = np.empty((num_windows, N, T))
    # output matrix size for analysis
    # TODO: print("Matrix size for analysis: ({}, {})".format(N, T))
    for i in range(num_windows):
        windowed_data = sync_data[:, i:i + Tw]
        dup_data = np.tile(windowed_data, (dup_n, dup_t))
        noise_data = dup_data + amplitude * noise
        msrs[i], unified_data[i] = single_ring(noise_data, U)
    return msrs, unified_data
