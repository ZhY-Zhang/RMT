from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from utils.file_processor import simple_loader, M_RECORDER
from utils.fast_analyzer import fast_msr_array
"""
This file analyzes the fault recorder data.
"""

# parameters
FILE_DIR = Path("D:\\work\\YuCai\\projects\\data")
SAVE_DIR = Path("D:\\work\\YuCai\\figures\\fault_recorder")
# f = 50Hz, Ts = 244us
T_WINDOW = int(1 / 50 / 244e-6)

plt.rcParams['figure.figsize'] = (12.8, 7.2)


def analyzer(file_path: Path, save_path: Path, Tw: int, display: bool = False, save: bool = False) -> None:
    # read the csv files and synchoronize data
    time_seq, sync_data = simple_loader(file_path)
    print("\033[32mLoaded all csv files successfully.\033[0m")
    # get windowed data matrix and calculate msr
    msr_array = fast_msr_array(sync_data, T_WINDOW)
    print("\033[32mCalculated the msr array successfully.\033[0m")
    # plot the MSR curve
    name = file_path.stem
    time_seq_msr = time_seq[T_WINDOW - 1:]
    plt.subplot(3, 1, 1)
    plt.title("{} - MSR".format(name))
    plt.plot(time_seq_msr, msr_array, label='MSR', linewidth=1)
    plt.xlim(time_seq[0], time_seq[-1])
    # plot the voltage curves
    plt.subplot(3, 1, 2)
    plt.title("Voltage")
    for m in ["VefA1", "VefB1", "VefC1"]:
        plt.plot(time_seq, sync_data[M_RECORDER.index(m)], label=m, linewidth=1)
    plt.legend(loc='upper right')
    plt.xlim(time_seq[0], time_seq[-1])
    # plot the current curves
    plt.subplot(3, 1, 3)
    plt.title("Current")
    for m in ["IA1", "IB1", "IC1", "IZ1"]:
        plt.plot(time_seq, sync_data[M_RECORDER.index(m)], label=m, linewidth=1)
    plt.legend(loc='upper right')
    plt.xlim(time_seq[0], time_seq[-1])
    plt.tight_layout()
    if save:
        plt.savefig(save_path)
        print("\033[32mSaved the figure as \"{}\".\033[0m".format(save_path))
    if display:
        plt.show()
    plt.clf()


if __name__ == "__main__":
    wrong_files = []
    for file_path in FILE_DIR.glob("*.csv"):
        save_path = SAVE_DIR / "{}.png".format(file_path.stem)
        try:
            analyzer(file_path, save_path, T_WINDOW, save=True)
        except pd.errors.ParserError:
            wrong_files.append(file_path)
            print("\033[31mFile \"{}\" has wrong format.\033[0m".format(file_path))
    # print files with wrong format
    if len(wrong_files) > 0:
        print("\033[31mThe files with wrong format are listed below.\033[0m")
        for p in wrong_files:
            print(p)
