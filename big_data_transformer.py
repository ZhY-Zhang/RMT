from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from utils.file_processor import synchronizer, M_TRANSFORMER
from utils.analyzer import get_msr_arrays
"""
This file analyzes the transformer data.
"""

# parameters
# file
FILE_DIR = Path("D:\\work\\YuCai\\projects\\data")
# data process
START_TIME = pd.Timestamp(2020, 6, 1)
STOP_TIME = pd.Timestamp(2021, 7, 1)
WINDOW_PERIOD = pd.Timedelta(days=7)
STATUS_LIST = ["T_infrared", "T_A", "T_B", "T_C"]
FACTOR_LIST = ["M_env"]
EXPECTED_SIZE = 64
# plot
AX2_LIST = ['T_env', 'T_oil', 'T_infrared', 'T_A', 'T_B', 'T_C']

if __name__ == "__main__":
    # calculate parameters
    idx_g = np.array([M_TRANSFORMER.index(m) for m in STATUS_LIST])
    idx_f = np.array([M_TRANSFORMER.index(m) for m in FACTOR_LIST])
    Tw = Nw = EXPECTED_SIZE
    time_step = WINDOW_PERIOD / Tw
    # read the csv files and synchoronize data
    time_seq, sync_data = synchronizer(FILE_DIR, M_TRANSFORMER, START_TIME, STOP_TIME, time_step)
    print("\033[32mLoaded all csv files successfully.\033[0m")
    # get windowed data matrix and calculate msr
    msrs_dat, msrs_arg = get_msr_arrays(sync_data, Nw, Tw, idx_g, idx_f)
    msr_difference = msrs_arg - msrs_dat
    print("\033[32mCalculated the msr array successfully.\033[0m")

    # draw the MSR figure
    time_seq_msr = time_seq[Tw - 1:]
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(time_seq_msr, msr_difference, linewidth=1, label='MSR difference', color='g')
    for m in AX2_LIST:
        ax2.plot(time_seq, sync_data[M_TRANSFORMER.index(m)], label=m, linestyle=':', linewidth=1)
    ax1.set_xlabel("Time")
    ax1.set_ylabel("MSR Difference")
    ax1.grid()
    ax2.set_ylabel("Temperature")
    fig.legend(loc='upper right')
    plt.show()
