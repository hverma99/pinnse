"""
Isothermal PFR, effluent-flow formulation (EFM): train the PINN.

The data loaders, network, optimizer and outputs shared by the isothermal PFR
cases are in ../common/train.py; this file holds the settings of this case.
"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # examples/isopfr

from common.process import E, k, key_species, nu, species
from common.train import load_data, normalize, save_outputs, train
from phys_res import Boundary, Physics


def main():
    I_S_data, D_S_data = load_data()
    norm_I_S_data, norm_D_S_data, I_S_metrics, D_S_metrics = normalize(
        "EFM", I_S_data, D_S_data
    )

    physics = Physics(I_S_metrics, D_S_metrics, species, nu, key_species, k, E)
    boundary = Boundary(species)

    return train(
        norm_I_S_data,
        norm_D_S_data,
        physics,
        boundary,
        log_grad_norms=True,  # record gradient-norm histories for the plots
        adapt_wts=False,
    )


if __name__ == "__main__":
    save_outputs(main())
