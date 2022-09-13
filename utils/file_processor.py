from pathlib import Path
from typing import Tuple, List
from tqdm import tqdm

import numpy as np
import pandas as pd
"""
This file contains the all the functions to load files, process the raw data
and unify data format. Please read the comments carefully is you need to use
them. The outputs of the functions are the same, that is, a time sequence (1-D
array, T) and a data matrix (2-D array, (N, T)).
"""

M_TRANSFORMER = [
    "T_env", "M_env", "I_1A", "I_1B", "I_1C", "I_2A", "I_2B", "I_2C", "T_oil", "M_breather", "T_infrared", "T_A", "T_B", "T_C",
    "T_breather", "H2", "CH4", "C2H6", "C2H4", "C2H2", "CO", "CO2"
]
__M_RECORDER = ["time", "IA1", "IB1", "IC1", "IZ1", "VefA1", "VefB1", "VefC1", "unknown"]
M_RECORDER = ["IA1", "IB1", "IC1", "IZ1", "VefA1", "VefB1", "VefC1"]


def spliter(file_dir: Path, raw_file: str):
    """
    This function is designed for "主变设备状态数据".\n
    The function splits a complex csv file to several small ones, each of which
    represents the measurement of a single sensor.
    """
    file_path = file_dir / raw_file
    cols = 2 * len(M_TRANSFORMER)
    raw_data = pd.read_csv(file_path, parse_dates=list(range(1, cols, 2)), infer_datetime_format=True)
    with tqdm(M_TRANSFORMER) as bar:
        for i, measurement in enumerate(M_TRANSFORMER):
            split_data = raw_data.iloc[:, 2 * i:2 * i + 2]
            split_data.columns = ['data', 'datetime']
            split_data = split_data.dropna()
            file_path = file_dir / "{}.csv".format(measurement)
            split_data.to_csv(file_path, index=False)
            bar.update()


def synchronizer(file_directory: Path, file_names: List[str], start_time: pd.Timestamp, stop_time: pd.Timestamp,
                 time_step: pd.Timedelta) -> Tuple[np.ndarray, np.ndarray]:
    """
    This function is designed for "主变设备状态数据".\n
    The function loads multiple csv files whose data may not be synchronized or
    evenly sampled.It applies linear interpolation to the data. It returns a
    time sequence (1-D array, T) and a synchronized data matrix (2-D array, (N,
    T)).
    """
    sync_data = []
    num_samples = int((stop_time - start_time) / time_step + 1)
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


def simple_loader(file_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    This function is designed for "录波故障样例".\n
    This function is used to load a single csv file whose data are synchronized
    and evenly sampled. The function returns a time sequence (1-D array, T) and
    a synchronized data matrix (2-D array, (N, T)).
    """
    raw_data = pd.read_csv(file_path, names=__M_RECORDER, header=0)
    time_seq = np.array(raw_data['time'])
    sync_matrix = np.array(raw_data.drop(['time', 'unknown'], axis=1)).T
    return time_seq, sync_matrix
