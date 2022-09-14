from typing import Tuple
from tqdm import tqdm

import numpy as np
from scipy.stats import unitary_group
"""
This file contains the core functions to analyze the data by RMT. Generally,
the only function you need to import is the "get_msr_array" or "get_msr_arrays"
function. Remember to input a proper window size to get better performance.
"""


def get_window(sync_data: np.ndarray, T: int) -> np.ndarray:
    """
    This function (generator) is used to generate the windowed data. The shape
    of synchronized data "sync_data" is (N, N_SAMPLES). "T" is the length of
    the window. It generates totally (N_SAMPLES - T + 1) windows.
    """
    num_windows = sync_data.shape[1] - T + 1
    if num_windows <= 0:
        raise ValueError("The synchronized data samples are not enough to analyze or T is too big.")
    for i in range(num_windows):
        yield sync_data[:, i:i + T]


def get_argumented_matrices(Dg: np.ndarray, Df: np.ndarray, dup_g: int, dup_c: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    The shape of status matrix "Dg" is (N1, T). The shape of factor matrix "Df"
    is (N2, T). The "dup_g" and "dup_c" are used to control how many times the
    raw matrices are duplicated.
    """
    # check the shape of status and factor matrices
    if Dg.shape[1] != Df.shape[1]:
        raise ValueError("The numbers of samplings of status and factors are different.")
    # step 1: duplicate and add random noise to the status matrix
    # the random noise is used to lower the relativity of matrix elements
    Dg = np.tile(Dg, (dup_g, 1))
    Noise = np.random.normal(0.0, 1.0, size=Dg.shape)
    Dg = Dg + 1.0 * Noise
    # step 2: duplicate and add random noise to the factor matrix
    Dc = np.tile(Df, (dup_c, 1))
    Noise = np.random.normal(0.0, 1.0, size=Dc.shape)
    Ef = Dc + 1.0 * Noise
    # calculate signal to noise ratio if needed
    # s2n_ratio = np.trace(np.matmul(Ef, Ef.T)) / np.trace(np.matmul(N, N.T)) / (m * m)
    # step 4: generate the argumented matrix "Ad" and reference argumented matrix "Ar"
    Ad = np.concatenate((Dg, Ef), axis=0)
    Ar = np.concatenate((Dg, Noise), axis=0)
    return Ad, Ar


def single_ring(X: np.ndarray, L: int = 1) -> float:
    """
    The parameter "X" is the data matrix, whose shape is (N, T). The parameter
    "L" is the number of singular value equivalent matrix (matrices). The
    function returns the Mean Spectrum Radius (MSR) of the product matrix Z.
    """
    N, T = X.shape[0], X.shape[1]
    c = N / T
    if c > 1:
        raise ValueError("The shape of data matrix must satisfy N <= T.")
    # step 1: normalize the data matrix X row by row
    # Here I use transpose function for readability.
    X1 = X.T
    X2 = (X1 - np.mean(X1, axis=0)) / np.std(X1, axis=0)
    np.nan_to_num(X2, copy=False)
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


def get_msr_array(sync_data: np.ndarray, expected_size: int, Tw: int) -> np.ndarray:
    """
    This function is used to calculate the MSR array of the synchronized data
    matrix. Choose a porper expected size for better performance.

    :param sync_data: the synchronized data matrix, a 2-D numpy array
    :param expected_size: the expected size of the duplicated matrix
    :param Tw: the window length
    :returns: the MSR array of the raw data, a 1-D numpy array
    """
    # calculate necessary arguments
    window_generator = get_window(sync_data, Tw)
    num_windows = sync_data.shape[1] - Tw + 1
    msrs = np.empty(num_windows)
    dup_n = expected_size // sync_data.shape[0]
    dup_t = int(np.ceil(expected_size / Tw))
    # output matrix size for analysis
    print("Matrix size for analysis: ({}, {})".format(dup_n * sync_data.shape[0], dup_t * Tw))
    # display the progress bar
    with tqdm(total=num_windows) as bar:
        # get data and move the window
        for i, windowed_data in enumerate(window_generator):
            dup_data = np.tile(windowed_data, (dup_n, dup_t))
            noise = np.random.normal(0.0, 1.0, size=dup_data.shape)
            noise_data = dup_data + 1.0 * noise
            msrs[i] = single_ring(noise_data)
            bar.update()
    return msrs


def get_msr_arrays(sync_data: np.ndarray, Nw: int, Tw: int, idx_g: np.ndarray,
                   idx_f: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    This function is used to calculate the MSR array of the data matrix and the
    reference matrix, which are generated based on the synchronized data, that
    is, "sync_data". Remember to input proper window size (Nw, Tw) for better
    performance.
    """
    # calculate necessary arguments
    window_generator = get_window(sync_data, Tw)
    num_windows = sync_data.shape[1] - Tw + 1
    msrs_dat = np.empty(num_windows)
    msrs_arg = np.empty(num_windows)
    dup_g = Nw // len(idx_g) // 2
    dup_c = Nw // len(idx_f) // 2
    # output matrix size
    print("Argumented matrix size: ({}, {})".format(dup_g * len(idx_g) + dup_c * len(idx_f), Tw))
    # display the progress bar
    with tqdm(total=num_windows) as bar:
        # get data and move the window
        for i, windowed_data in enumerate(window_generator):
            # split the windowed data matrix into the status matrix and the factor matrix
            Dg = windowed_data[idx_g, :]
            Df = windowed_data[idx_f, :]
            # generate the argumented matrix and the reference argumented matrix
            Ad, Ar = get_argumented_matrices(Dg, Df, dup_g, dup_c)
            # calculate the MSR of the two matrices
            msrs_dat[i] = single_ring(Ad)
            msrs_arg[i] = single_ring(Ar)
            # update the progress bar
            bar.update()
    return msrs_dat, msrs_arg
