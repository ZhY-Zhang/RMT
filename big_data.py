import warnings
from pathlib import Path
from typing import Tuple, List
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import unitary_group
"""
This file draws the relativity of the arguments (MSR) in the data file.
"""
# TODO: 试试在 csv 文件读取得到的 raw_data 里加上噪音
# TODO: 做一个矩阵色块的动效

warnings.filterwarnings('ignore')

# parameters
# file
FILE_DIR = Path("D:\\work\\YuCai\\projects\\data")
MEASUREMENTS = [
    "T_env", "M_env", "I_1A", "I_1B", "I_1C", "I_2A", "I_2B", "I_2C", "T_oil", "M_breather", "T_infrared", "T_A", "T_B", "T_C",
    "T_breather", "H2", "CH4", "C2H6", "C2H4", "C2H2", "CO", "CO2"
]
# data process
START_TIME = pd.Timestamp(2020, 6, 1)
STOP_TIME = pd.Timestamp(2021, 7, 1)
WINDOW_PERIOD = pd.Timedelta(days=14)
STATUS_LIST = ["T_infrared", "T_A", "T_B", "T_C"]
FACTOR_LIST = ["T_env"]
DUP_STATUS: int = 8
# calculate parameters
STATUS_INDICES = np.array([MEASUREMENTS.index(m) for m in STATUS_LIST])
FACTOR_INDICES = np.array([MEASUREMENTS.index(m) for m in FACTOR_LIST])
WINDOW_LENGTH = 2 * DUP_STATUS * len(STATUS_INDICES)
TIME_STEP = WINDOW_PERIOD / WINDOW_LENGTH
N_SAMPLES = int((STOP_TIME - START_TIME) / TIME_STEP + 1)
print("Window Length:", WINDOW_LENGTH)


def synchronize_data(file_directory: Path, file_names: List[str], start_time: pd.Timestamp, stop_time: pd.Timestamp,
                     num_samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    This function is used to synchronize the sensor data of the transformer. It
    applies linear interpolation to the data. It returns a 2-D numpy array. The
    rows represent different sensors while the columns represent different time.
    """
    sync_data = []
    time_seq = pd.date_range(start_time, stop_time, num_samples)
    time_int_seq = np.array(time_seq, dtype=np.int64)
    for name in file_names:
        file_path = file_directory.joinpath("{}.csv".format(name))
        raw_data = pd.read_csv(file_path, parse_dates=[1], infer_datetime_format=True)
        t, v = np.array(raw_data['datetime'], dtype=np.int64), np.array(raw_data['data'], dtype=float)
        data_seq = np.interp(time_int_seq, t, v)
        sync_data.append(data_seq)
    sync_matrix = np.array(sync_data)
    return time_seq, sync_matrix


def get_window(sync_data: np.ndarray, T: int) -> np.ndarray:
    """
    This function (generator) is used to generate the windowed data. The shape
    of synchronized data "sync_data" is (N, N_SAMPLES). "T" is the length of
    the window. It generates totally (N_SAMPLES - T + 1) windows.
    """
    num_windows = sync_data.shape[1] - T + 1
    if num_windows <= 0:
        raise ValueError("The synchronized data samples are not enough to analyze. \
Please change STOP_DATE, START_DATE, TIME_STEP or WINDOW_LENGTH.")
    for i in range(num_windows):
        yield sync_data[:, i:i + T]


def get_argumented_matrices(Dg: np.ndarray, Df: np.ndarray, m: float = 1.0, dup_g: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    The shape of status matrix "Dg" is (N1, T). The shape of factor matrix "Df"
    is (N2, T). Generally, N1 >> N2. The amplitude of noise "m" is 1.0 by
    default.
    """
    # check the shape of status and factor matrices
    if Dg.shape[1] != Df.shape[1]:
        raise ValueError("The numbers of samplings of status and factors are different.")
    # step 0: duplicate the status matrix to satisfy that N1 >> N2 and add noise
    Dg = np.tile(Dg, (dup_g, 1))
    Noise = np.random.normal(0.0, 1.0, size=Dg.shape)
    Dg = Dg + m * Noise
    # step 1: duplicate the factor matrix to match the size of status matrix
    k = Dg.shape[0] // Df.shape[0]
    Dc = np.tile(Df, (k, 1))
    # step 2: add random noise to lower the relativity of the elements of the duplicated matrix
    Noise = np.random.normal(0.0, 1.0, size=Dc.shape)
    Ef = Dc + m * Noise
    # calculate signal to noise ratio if needed
    # s2n_ratio = np.trace(np.matmul(Ef, Ef.T)) / np.trace(np.matmul(N, N.T)) / (m * m)
    # step 3: generate the argumented matrix "Ad" and reference argumented matrix "Ar"
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


def get_msr(sync_data: np.ndarray,
            window_length: int,
            status_indices: np.ndarray,
            factor_indices: np.ndarray,
            dup_g: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    This function is used to calculate the msr array from the synchronized data.
    """
    msr_ad_list, msr_ar_list = [], []
    window_generator = get_window(sync_data, window_length)
    # display the progress bar
    num_windows = sync_data.shape[1] - window_length + 1
    with tqdm(total=num_windows) as bar:
        # get data and move the window
        for windowed_data in window_generator:
            # split the windowed data matrix into the status matrix and the factor matrix
            Dg = windowed_data[status_indices, :]
            Df = windowed_data[factor_indices, :]
            # generate the argumented matrix and the reference argumented matrix
            Ad, Ar = get_argumented_matrices(Dg, Df, dup_g=dup_g)
            # calculate the MSR of the two matrices
            msr_ad = single_ring(Ad)
            msr_ad_list.append(msr_ad)
            msr_ar = single_ring(Ar)
            msr_ar_list.append(msr_ar)
            # update the progress bar
            bar.update()
    # convert lists to numpy arrays
    msr_ad_array = np.array(msr_ad_list)
    msr_ar_array = np.array(msr_ar_list)
    return msr_ad_array, msr_ar_array


if __name__ == "__main__":
    # read the csv files and synchoronize data
    time_seq, sync_data = synchronize_data(FILE_DIR, MEASUREMENTS, START_TIME, STOP_TIME, N_SAMPLES)
    print("\033[32mLoaded all csv files successfully.\033[0m")
    # get windowed data matrix and calculate msr
    msr_ad_array, msr_ar_array = get_msr(sync_data, WINDOW_LENGTH, STATUS_INDICES, FACTOR_INDICES, DUP_STATUS)
    print("\033[32mCalculated the msr array successfully.\033[0m")
    # deal with MSR of the argumented matrix and the reference matrix
    msr_difference = msr_ar_array - msr_ad_array

    # draw the grand figure
    time_seq_msr = time_seq[WINDOW_LENGTH - 1:]
    """
    plt.subplot(4, 6, 1)
    plt.title("MSR difference")
    # plt.plot(time_seq_msr, msr_ad_array, linewidth=1, label='real data')
    # plt.plot(time_seq_msr, msr_ar_array, linewidth=1, label='reference')
    plt.plot(time_seq_msr, msr_difference, linewidth=1)
    plt.xticks(time_seq_msr[[0, -1]])
    for i, title in enumerate(MEASUREMENTS):
        plt.subplot(4, 6, i + 2)
        plt.title(title)
        plt.plot(time_seq, sync_data[i])
        plt.xticks(time_seq[[0, -1]])
    plt.show()
    """

    # draw the MSR figure
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    # ax2.plot(time_seq, sync_data[10] - sync_data[0], linewidth=1, label='delta_T', color='b', linestyle=':')
    ax2.plot(time_seq, sync_data[0], linewidth=1, label='T_env', color='c', linestyle=':')
    ax2.plot(time_seq, sync_data[10], linewidth=1, label='T_infrared', color='b', linestyle=':')
    ax2.plot(time_seq, sync_data[11], linewidth=1, label='T_A', color='y', linestyle=':')
    ax2.plot(time_seq, sync_data[12], linewidth=1, label='T_B', color='g', linestyle=':')
    ax2.plot(time_seq, sync_data[13], linewidth=1, label='T_C', color='r', linestyle=':')
    ax2.set_ylabel("Temperature")
    # ax1.plot(time_seq_msr, msr_ad_array, linewidth=1, label='real data')
    # ax1.plot(time_seq_msr, msr_ar_array, linewidth=1, label='reference')
    ax1.plot(time_seq_msr, msr_difference, linewidth=1, label='MSR difference', color='g')
    ax1.set_xlabel("Time")
    ax1.set_ylabel("MSR Difference")
    ax1.grid()
    fig.legend(loc='upper right')
    plt.show()
