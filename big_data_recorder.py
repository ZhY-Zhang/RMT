from pathlib import Path

import matplotlib.pyplot as plt

from utils.file_processor import simple_loader, M_RECORDER
from utils.analyzer import get_msr_array
"""
This file analyzes the fault recorder data.
"""

# parameters
# A-8, A-10, A-36
FILE_PATH = Path("D:\\work\\YuCai\\projects\\data\\A-8.csv")
EXPECTED_SIZE = 256
T_WINDOW = 64

if __name__ == "__main__":
    Nw = Tw = EXPECTED_SIZE
    # read the csv files and synchoronize data
    time_seq, sync_data = simple_loader(FILE_PATH)
    print("\033[32mLoaded all csv files successfully.\033[0m")
    # get windowed data matrix and calculate msr
    msr_array = get_msr_array(sync_data, EXPECTED_SIZE, T_WINDOW)
    print("\033[32mCalculated the msr array successfully.\033[0m")

    # plot the MSR curve
    time_seq_msr = time_seq[T_WINDOW - 1:]
    plt.subplot(3, 1, 1)
    plt.title("MSR")
    plt.plot(time_seq_msr, msr_array, label='MSR', linewidth=1)
    # plot the voltage curves
    plt.subplot(3, 1, 2)
    plt.title("Voltages")
    for m in ["VefA1", "VefB1", "VefC1"]:
        plt.plot(time_seq, sync_data[M_RECORDER.index(m)], label=m, linewidth=1)
    plt.legend(loc='upper right')
    # plot the current curves
    plt.subplot(3, 1, 3)
    plt.title("Currents")
    for m in ["IA1", "IB1", "IC1", "IZ1"]:
        plt.plot(time_seq, sync_data[M_RECORDER.index(m)], label=m, linewidth=1)
    plt.legend(loc='upper right')
    plt.show()
