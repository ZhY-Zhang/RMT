from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
"""
This file contains the all the functions to process the raw data. Read the
comments carefully is you need to use them.
"""


def synchronize_data(file_directory: Path, file_names: List[str], start_time: pd.Timestamp, stop_time: pd.Timestamp,
                     time_step: pd.Timedelta) -> Tuple[np.ndarray, np.ndarray]:
    """
    This function is used to synchronize the sensor data of the transformer. It
    applies linear interpolation to the data. It returns a 2-D numpy array. The
    rows represent different sensors while the columns represent different time.
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
