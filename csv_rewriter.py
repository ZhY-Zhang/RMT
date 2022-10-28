from pathlib import Path
from utils.file_processor import splitter
"""
This file splits the transformer data file to several small files.
"""

# parameters
FILE_DIR = Path("D:\\work\\YuCai\\projects\\data")
RAW_FILE = "transformer_1.csv"

if __name__ == "__main__":
    splitter(FILE_DIR, RAW_FILE)
