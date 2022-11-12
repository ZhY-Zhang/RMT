from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FILE_DIR = Path("D:\\work\\YuCai\\projects\\data")
FILE_NAME = "KB03-8_2022-11-04-20-26-16.csv"
NAMES = ["t", "P1", "P2"]

if __name__ == '__main__':
    # load data file
    file_path = FILE_DIR.joinpath(FILE_NAME)
    df = pd.read_csv(file_path, names=NAMES, header=0, index_col="t", parse_dates=[0], infer_datetime_format=True)
    # interpolate lost data
    df.interpolate(method='time', limit_direction='forward', inplace=True)
    # copy the time to a new column
    df["dt"] = df.index
    # calculate the difference of two continous rows
    df = df.diff()
    # convert the energy to power
    df["P1"] = df["P1"] / df["dt"].dt.total_seconds() * 1000
    df["P2"] = df["P2"] / df["dt"].dt.total_seconds() * 1000
    print(df)
    plt.plot(df["P1"])
    plt.plot(df["P2"])
    plt.show()
