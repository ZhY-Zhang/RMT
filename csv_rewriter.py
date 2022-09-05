from pathlib import Path
from tqdm import tqdm
import pandas as pd
"""
This program splits a big csv file into several small files.
"""

# parameters
FILE_DIR = Path("D:\\work\\YuCai\\projects\\data")
RAW_FILE = "transformer_1.csv"
MEASUREMENTS = [
    "T_env", "M_env", "I_1A", "I_1B", "I_1C", "I_2A", "I_2B", "I_2C", "T_oil", "M_breather", "T_infrared", "T_A", "T_B", "T_C",
    "T_breather", "H2", "CH4", "C2H6", "C2H4", "C2H2", "CO", "CO2"
]
COLS = 2 * len(MEASUREMENTS)

if __name__ == "__main__":
    file_path = FILE_DIR / RAW_FILE
    raw_data = pd.read_csv(file_path, parse_dates=list(range(1, COLS, 2)), infer_datetime_format=True)
    with tqdm(MEASUREMENTS) as bar:
        for i, measurement in enumerate(MEASUREMENTS):
            split_data = raw_data.iloc[:, 2 * i:2 * i + 2]
            split_data.columns = ['data', 'datetime']
            split_data = split_data.dropna()
            file_path = FILE_DIR / "{}.csv".format(measurement)
            split_data.to_csv(file_path, index=False)
            bar.update()
