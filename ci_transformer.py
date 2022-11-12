from pathlib import Path
from typing import List
import numpy as np
import pandas as pd

from dowhy import CausalModel

import warnings
from sklearn.exceptions import DataConversionWarning
"""
This file analyzes the transformer data by doWhy.
"""

warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.filterwarnings(action='ignore', category=FutureWarning)

# parameters
# file
DATASET_PATH = Path("D:\\work\\YuCai\\projects\\data")
# data process
START_TIME = pd.Timestamp(2020, 6, 1)
STOP_TIME = pd.Timestamp(2021, 7, 1)
SAMPLE_PERIOD = pd.Timedelta(hours=4)
# causal model
TREATMENT = 'T_env'
OUTCOME = 'T_oil'

DATASET_FILES = [
    "T_env", "I_1A", "I_1B", "I_1C", "I_2A", "I_2B", "I_2C", "T_oil", "T_infrared", "T_A", "T_B", "T_C", "T_breather"
]
DATAFRAME_COLUMNS = [
    "J_time", "J_weekday", "T_env", "I_1A", "I_1B", "I_1C", "I_2A", "I_2B", "I_2C", "T_oil", "T_infrared", "T_A", "T_B", "T_C",
    "T_breather"
]
GML_GRAPH = ""


def causal_loader(dataset_path: Path, dataset_files: List[str], start_time: pd.Timestamp, stop_time: pd.Timestamp,
                  time_step: pd.Timedelta) -> pd.DataFrame:
    """
    The function loads multiple csv files whose data may not be synchronized or
    evenly sampled, as well as a GML graph string. It applies linear
    interpolation to the data. It returns a dataset containing the synchronized
    data, the causal graph and other necessary information.
    """
    sync_data = {}
    # acquire standard time sequence
    num_samples = int((stop_time - start_time) / time_step + 1)
    time_seq = pd.date_range(start_time, stop_time, num_samples)
    time_int_seq = np.array(time_seq, dtype=np.int64)
    # load dataset
    sync_data['J_time'] = time_seq.hour
    sync_data['J_weekday'] = time_seq.weekday
    for name in dataset_files:
        file_path = dataset_path.joinpath("{}.csv".format(name))
        raw_data = pd.read_csv(file_path, parse_dates=[1], infer_datetime_format=True)
        t, v = np.array(raw_data['datetime'], dtype=np.int64), np.array(raw_data['data'], dtype=float)
        data_seq = np.interp(time_int_seq, t, v)
        sync_data[name] = data_seq
    # construct pandas dataframe
    df = pd.DataFrame(data=sync_data)
    return df


if __name__ == "__main__":
    # TODO: improve dag format convertion
    # O. Convert graph format.
    with open("dag_transformer.txt") as f:
        s = f.readlines()
    s1 = "digraph {"
    for li in s:
        if '>' in li:
            l1 = li[0:-1] + ';'
            s1 += l1
    s1 += '}'
    GML_GRAPH = s1
    print(GML_GRAPH)

    # I. Create a causal model from the data and given graph.
    df = causal_loader(DATASET_PATH, DATASET_FILES, START_TIME, STOP_TIME, SAMPLE_PERIOD)
    print("\033[32mLoaded all csv files successfully.\033[0m")
    print(df)
    model = CausalModel(data=df, treatment=TREATMENT, outcome=OUTCOME, graph=GML_GRAPH)
    # model.view_model()

    # II. Identify causal effect and return target estimands.
    identified_estimand = model.identify_effect()
    print(identified_estimand)

    # III. Estimate the target estimand using a statistical method.
    estimate = model.estimate_effect(identified_estimand, method_name="backdoor.linear_regression")
    print(estimate)

    # IV. Refute the obtained estimate using multiple robustness checks.
    refute_results = model.refute_estimate(identified_estimand, estimate, method_name="random_common_cause")
    print(refute_results)
