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
EXPECTED_SIZE: int = 60
# plot
AX2_LIST = ['T_env', 'T_oil', 'T_infrared', 'T_A', 'T_B', 'T_C']

if __name__ == "__main__":
    # calculate parameters
    idx_g = np.array([MEASUREMENTS.index(m) for m in STATUS_LIST])
    idx_f = np.array([MEASUREMENTS.index(m) for m in FACTOR_LIST])
    Tw = Nw = EXPECTED_SIZE
    time_step = WINDOW_PERIOD / Tw
    # read the csv files and synchoronize data
    time_seq, sync_data = synchronize_data(FILE_DIR, MEASUREMENTS, START_TIME, STOP_TIME, time_step)
    print("\033[32mLoaded all csv files successfully.\033[0m")
    # get windowed data matrix and calculate msr
    msrs_dat, msrs_arg = get_msr(sync_data, Nw, Tw, idx_g, idx_f)
    msr_difference = msrs_arg - msrs_dat
    print("\033[32mCalculated the msr array successfully.\033[0m")

    # draw the MSR figure
    time_seq_msr = time_seq[Tw - 1:]
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(time_seq_msr, msr_difference, linewidth=1, label='MSR difference', color='g')
    for m in AX2_LIST:
        ax2.plot(time_seq, sync_data[MEASUREMENTS.index(m)], label=m, linestyle=':', linewidth=1)
    ax1.set_xlabel("Time")
    ax1.set_ylabel("MSR Difference")
    ax1.grid()
    ax2.set_ylabel("Temperature")
    fig.legend(loc='upper right')
    plt.show()
    """
    # draw the grand figure
    plt.subplot(4, 6, 1)
    plt.title("MSR difference")
    # plt.plot(time_seq_msr, msrs_dat, linewidth=1, label='real data')
    # plt.plot(time_seq_msr, msrs_arg, linewidth=1, label='reference')
    plt.plot(time_seq_msr, msr_difference, linewidth=1)
    plt.xticks(time_seq_msr[[0, -1]])
    for i, title in enumerate(MEASUREMENTS):
        plt.subplot(4, 6, i + 2)
        plt.title(title)
        plt.plot(time_seq, sync_data[i])
        plt.xticks(time_seq[[0, -1]])
    plt.show()
    """
