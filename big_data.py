import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from utils.file_loader import synchronize_data
from utils.analyzer import get_msr
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
FACTOR_LIST = ["M_env"]
N_WINDOW: int = 16
# calculate parameters
STATUS_INDICES = np.array([MEASUREMENTS.index(m) for m in STATUS_LIST])
FACTOR_INDICES = np.array([MEASUREMENTS.index(m) for m in FACTOR_LIST])
T_WINDOW = 2 * N_WINDOW * len(STATUS_INDICES)
TIME_STEP = WINDOW_PERIOD / T_WINDOW
N_SAMPLES = int((STOP_TIME - START_TIME) / TIME_STEP + 1)
print("Window Length:", T_WINDOW)

if __name__ == "__main__":
    # read the csv files and synchoronize data
    time_seq, sync_data = synchronize_data(FILE_DIR, MEASUREMENTS, START_TIME, STOP_TIME, N_SAMPLES)
    print("\033[32mLoaded all csv files successfully.\033[0m")
    # get windowed data matrix and calculate msr
    msr_ad_array, msr_ar_array = get_msr(sync_data, N_WINDOW, T_WINDOW, STATUS_INDICES, FACTOR_INDICES)
    print("\033[32mCalculated the msr array successfully.\033[0m")
    # deal with MSR of the argumented matrix and the reference matrix
    msr_difference = msr_ar_array - msr_ad_array

    # draw the grand figure
    time_seq_msr = time_seq[T_WINDOW - 1:]
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
    # ax2.plot(time_seq, sync_data[0], linewidth=1, label='T_env', color='c', linestyle=':')
    # ax2.plot(time_seq, sync_data[8], linewidth=1, label='T_oil', color='black', linestyle=':')
    ax2.plot(time_seq, sync_data[1], linewidth=1, label='M_env', color='c', linestyle=':')
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
