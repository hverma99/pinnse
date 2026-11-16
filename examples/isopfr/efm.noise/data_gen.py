"""
Generate I_S_data.xlsx and D_S_data.xlsx for this case (EFM outputs), with
the shared generator of ../common/data_gen.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # examples/isopfr

from common.data_gen import write_datasets

if __name__ == "__main__":
    write_datasets("EFM")
