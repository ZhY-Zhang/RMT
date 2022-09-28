from pathlib import Path
from typing import Tuple
from multiprocessing import Pool

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from utils.file_processor import simple_loader, M_RECORDER
from utils.fast_analyzer import get_msr_array
"""
This file analyzes the fault recorder data.
"""

# parameters
FILE_DIR = Path("D:\\work\\YuCai\\projects\\data")
SAVE_DIR = Path("D:\\work\\YuCai\\figures\\fault_recorder")

T_WINDOW = int(1 / 50 / 244e-6)                # f = 50Hz, Ts = 244us
EXPECTED_SIZE = 80

PROCESSES = 8
SAVE_FIGURE = True
SAVE_VIDEO = False

plt.rcParams['figure.figsize'] = (12.8, 7.2)


def make_figure(sync_data: np.ndarray, msr_array: np.ndarray, time_seq: np.ndarray, file_name: str):
    # plot the MSR curve
    time_seq_msr = time_seq[T_WINDOW - 1:]
    plt.subplot(3, 1, 1)
    plt.title("{} - MSR".format(file_name))
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
    # adjust and save the figure
    plt.tight_layout()
    figure_path = SAVE_DIR / "{}.png".format(file_name)
    plt.savefig(figure_path)
    plt.clf()
    # TODO: print("\033[32mSaved the figure as \"{}\".\033[0m".format(figure_path))


def make_video(unified_data: np.ndarray, msr_array: np.ndarray, time_seq: np.ndarray, file_name: str):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    mat_data = ax.matshow(unified_data[0], vmin=-3, vmax=3)
    text_msr = ax.text(0, -1, "{}".format(msr_array[0]))
    fig.colorbar(mat_data)

    def update_ani(frame):
        mat_data.set_data(unified_data[frame])
        text_msr.set_text("{}".format(msr_array[frame]))
        return mat_data, text_msr,

    ani = FuncAnimation(fig, update_ani, frames=len(unified_data), interval=5, blit=True)
    video_path = SAVE_DIR / "{}.mp4".format(file_name)
    ani.save(video_path, fps=60, writer="ffmpeg")
    # TODO: print("\033[32mSaved the video as \"{}\".\033[0m".format(video_path))


def analyzer(file_path: Path) -> Tuple[str, bool]:
    # read the csv files and synchoronize data
    try:
        time_seq, sync_data = simple_loader(file_path)
    except pd.errors.ParserError:
        return file_path, False
    # TODO: print("\033[32mLoaded all csv files successfully.\033[0m")
    # get windowed data matrix and calculate msr
    msr_array, unified_data = get_msr_array(sync_data, T_WINDOW, EXPECTED_SIZE)
    # TODO: print("\033[32mCalculated the msr array successfully.\033[0m")
    # make and save the figure and video
    if SAVE_FIGURE:
        make_figure(sync_data, msr_array, time_seq, file_path.stem)
    if SAVE_VIDEO:
        make_video(unified_data, msr_array, time_seq, file_path.stem)
    return file_path, True


if __name__ == "__main__":
    file_paths = FILE_DIR.glob("*.csv")
    fail_paths = []
    with Pool(processes=PROCESSES) as pool:
        results = pool.imap_unordered(analyzer, file_paths)
        for file_path, finish in results:
            if finish:
                print("\033[32mFile \"{}\" is processed.\033[0m".format(file_path))
            else:
                fail_paths.append(file_path)
                print("\033[31mFile \"{}\" has wrong format.\033[0m".format(file_path))
    # print files with wrong format
    if len(fail_paths) > 0:
        print("\033[31mThe files with wrong format are listed below.\033[0m")
        for p in fail_paths:
            print(p)
