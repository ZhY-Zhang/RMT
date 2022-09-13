from pathlib import Path

import matplotlib.pyplot as plt

from utils.file_processor import simple_loader, M_RECORDER
from utils.analyzer import get_msr_array
"""
This file analyzes the fault recorder data.
"""

# parameters
FILE_PATH = Path("D:\\work\\YuCai\\projects\\data\\1-13A.csv")
EXPECTED_SIZE = 64

if __name__ == "__main__":
    Nw = Tw = EXPECTED_SIZE
    # read the csv files and synchoronize data
    time_seq, sync_data = simple_loader(FILE_PATH)
    print("\033[32mLoaded all csv files successfully.\033[0m")
    # get windowed data matrix and calculate msr
    msr_array = get_msr_array(sync_data, Nw, Tw)
    print("\033[32mCalculated the msr array successfully.\033[0m")

    # plot the MSR curve
    time_seq_msr = time_seq[Tw - 1:]
    plt.subplot(3, 1, 1)
    plt.title("MSR")
    plt.plot(time_seq_msr, msr_array, label='MSR', linewidth=1)
    # plot the voltage curves
    plt.subplot(3, 1, 2)
    plt.title("Voltages")
    for m in ["VefA1", "VefB1", "VefC1"]:
        plt.plot(time_seq, sync_data[M_RECORDER.index(m)], label=m, linewidth=1)
    # plot the current curves
    plt.subplot(3, 1, 3)
    plt.title("Currents")
    for m in ["IA1", "IB1", "IC1", "IZ1"]:
        plt.plot(time_seq, sync_data[M_RECORDER.index(m)], label=m, linewidth=1)
    plt.show()
