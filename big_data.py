from pathlib import Path
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import unitary_group
"""
This file draws the relativity of the arguments (MSR) in the data file.
"""

# parameters
# file
FILE_PATH = Path("D:\\work\\YuCai\\projects\\data\\transformer_1.csv")
COL_NAMES = [
    "T_env", "date_T_env", "M_env", "date_M_env", "I_1A", "date_I_1A", "I_1B", "date_I_1B", "I_1C", "date_I_1C", "I_2A",
    "date_I_2A", "I_2B", "date_I_2B", "I_2C", "date_I_2C", "T_oil", "date_T_oil", "M_breather", "date_M_breather", "T_infrared",
    "date_T_infrared", "T_A", "date_T_A", "T_B", "date_T_B", "T_C", "date_T_C", "T_breather", "date_T_breather", "H2",
    "date_H2", "CH4", "date_CH4", "C2H6", "date_C2H6", "C2H4", "date_C2H4", "C2H2", "date_C2H2", "CO", "date_CO", "CO2",
    "date_CO2"
]
COLS = len(COL_NAMES)
# data process
START_DATE = pd.Timestamp(2020, 6, 1)
STOP_DATE = pd.Timestamp(2020, 6, 5)
TIME_STEP = pd.Timedelta(hours=1)
N = 22
T = 48


def synchronize_data(data: pd.DataFrame, start_date: pd.Timestamp, stop_date: pd.Timestamp,
                     time_step: pd.Timedelta) -> np.ndarray:
    pass


def get_window(sync_data: np.ndarray, N: int, T: int) -> np.ndarray:
    pass


def argumented_matrix_generator(Dg: np.ndarray, Df: np.ndarray, m: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    The shape of status matrix "Dg" is (N1, T). The shape of factor matrix "Df" is (N2, T).
    Generally, N1 >> N2. The amplitude of noise "m" is 1.0 by default.
    """
    # check the shape of status and factor matrices
    if Dg.shape[1] != Df.shape[1]:
        raise ValueError("The numbers of samplings of status and factors are different.")
    # step 1: duplicate the factor matrix to match the size of status matrix
    k = Dg.shape[0] // Df.shape[0]
    Dc = np.tile(Df, (k, 1))
    # step 2: add random noise to lower the relativity of the elements of the duplicated matrix
    Noise = np.random.normal(0.0, 1.0, size=Dc.shape)
    Ef = Dc + m * Noise
    # calculate signal to noise ratio if needed
    # s2n_ratio = np.trace(np.matmul(Ef, Ef.T)) / np.trace(np.matmul(N, N.T)) / (m * m)
    # step 3: generate the argumented matrix "A" and reference argumented matrix "An"
    A = np.concatenate((Dg, Ef), axis=0)
    An = np.concatenate((Dg, Noise), axis=0)
    return A, An


def single_ring(X: np.ndarray, L: int = 1) -> float:
    """
    The parameter "X" is the data matrix, whose shape is (N, T).
    The parameter "L" is the number of singular value equivalent matrix (matrices).
    The function returns the Mean Spectrum Radius (MSR) of the product matrix Z.
    """
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
    # read the csv file, and parse the date format
    raw_data = pd.read_csv(FILE_PATH, parse_dates=list(range(1, COLS, 2)), infer_datetime_format=True)
    sync_data = synchronize_data(raw_data, START_DATE, STOP_DATE, TIME_STEP)
    # reduce memory use
    del raw_data
    # get windowed data matrix and calculate msr
    msr_list = []
    num_samples = int(np.ceil((STOP_DATE - START_DATE) / TIME_STEP))
    for i in range(num_samples):
        data_matrix = get_window(sync_data, N, T)
        msr = single_ring(data_matrix)
        msr_list.append(msr)
    # draw the figure
    plt.plot(msr_list)
    plt.show()

    # tmps
    '''
    Xg = np.random.normal(12.34, 65.43, size=(80, 200))
    Xf = np.random.normal(9.876, 6.543, size=(6, 200))
    ar, ra = argumented_matrix_generator(Xg, Xf)
    msr_ar = single_ring(ar, 1)
    msr_ra = single_ring(ra, 1)
    print(msr_ar, msr_ra)
    '''

    # plt.hist(data.iloc[:, list(range(1, COLS, 2))], bins=100, stacked=True)
    # plt.show()

    # plt.plot(data.iloc[:, list(range(1, COLS, 2))], data.iloc[:, list(range(0, COLS, 2))], label=COL_NAMES[0:COLS:2])
    # plt.legend()
    # plt.show()
