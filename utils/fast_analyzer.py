from tqdm import tqdm

import numpy as np
"""
This file contains the speed boosted version of some analysis functions. Some
of them lack type checking and may cause unexpected errors. Check the input
parameters carefully before using them.
"""


def fast_single_ring(X: np.ndarray) -> float:
    """
    This function is a speed boosted version of the "single_ring" function. It
    doesn't check the size of input matrices and doesn't support singular value
    equivalent matrices.
    """
    # step 1: normalize the data matrix X row by row
    X1 = X.T
    X2 = (X1 - np.mean(X1, axis=0)) / np.std(X1, axis=0)
    np.nan_to_num(X2, copy=False)
    Xn = X2.T
    # step 2: calculate the singular values of Xn and arrange them as a diagonal matrix
    Xu = np.diag(np.linalg.svd(Xn)[1])
    # step 3: normalize Xu
    Zn = (Xu - np.mean(Xu)) / np.std(Xu) / np.sqrt(X.shape[0])
    # step 4: calculate the spectrum of Xu and the MSR
    ei = np.linalg.eigvals(Zn)
    msr = np.mean(np.abs(ei))
    return msr


def fast_msr_array(sync_data: np.ndarray, Tw: int) -> np.ndarray:
    """
    This function is a speed boosted version of the "get_msr_array" function.
    """
    # calculate necessary arguments
    num_windows = sync_data.shape[1] - Tw + 1
    msrs = np.empty(num_windows)
    # output matrix size for analysis
    print("Matrix size for analysis: ({}, {})".format(sync_data.shape[0], Tw))
    # display the progress bar
    with tqdm(total=num_windows) as bar:
        # get data and move the window
        for i in range(num_windows):
            windowed_data = sync_data[:, i:i + Tw]
            msrs[i] = fast_single_ring(windowed_data)
            bar.update()
    return msrs
