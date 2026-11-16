"""
Isothermal PFR, EFM: compare the trained PINN with the first-principles
solution along a reactor profile (figures and compare.xlsx in compare/).
"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # examples/isopfr

from common.evaluate import compare_profile
from common.process import E, k, key_species, nu, species
from common.train import load_data, normalize
from phys_res import Physics

if __name__ == "__main__":
    I_S_data, D_S_data = load_data()
    _, _, I_S_metrics, D_S_metrics = normalize("EFM", I_S_data, D_S_data)
    physics = Physics(I_S_metrics, D_S_metrics, species, nu, key_species, k, E)

    compare_profile("EFM", I_S_data, D_S_data, I_S_metrics, D_S_metrics, physics)
