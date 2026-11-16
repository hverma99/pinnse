"""
Inverse EFM PINN for the isothermal PFR: estimate the activation energies of
the three reactions from the labeled data while training the surrogate.

The data loaders, network, optimizer and outputs shared by the isothermal PFR
cases are in ../common/train.py; this file holds the settings of this case.
"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # examples/isopfr

from common.process import k, key_species, nu, species
from common.train import DEVICE, load_data, normalize, save_outputs, train
from phys_res import Boundary, Physics


def main():
    # ------------------- Inverse PINN Setup ----------------------------
    # -------------- Unknown Activation Energy (E) ----------------------
    # E = theta * 1e3, theta in kJ/mol is trainable parameter;
    # the true values are common.process.E = [105.2, 124.89, 89.66] kJ/mol
    theta = nn.Parameter(
        torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device=DEVICE)
    )
    # -------------------------------------------------------------------

    I_S_data, D_S_data = load_data()
    norm_I_S_data, norm_D_S_data, I_S_metrics, D_S_metrics = normalize(
        "EFM", I_S_data, D_S_data
    )

    physics = Physics(I_S_metrics, D_S_metrics, species, nu, key_species, k, theta)
    boundary = Boundary(species)

    return train(
        norm_I_S_data,
        norm_D_S_data,
        physics,
        boundary,
        lr=5e-4,
        theta=theta,  # optimized with the network
        log_grad_norms=True,  # record gradient-norm histories for the plots
        adapt_wts=False,
    )


if __name__ == "__main__":
    save_outputs(main())
