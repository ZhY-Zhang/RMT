from pathlib import Path
from utils.file_processor import spliter
"""
This file splits the transformer data file to several small files.
"""

# parameters
FILE_DIR = Path("D:\\work\\YuCai\\projects\\data")
RAW_FILE = "transformer_1.csv"

if __name__ == "__main__":
    spliter(FILE_DIR, RAW_FILE)
