"""
Effluent-flow-based (EFM) PINN model for the isothermal PFR process, trained
against labeled data perturbed by measurement noise.

This is an example where labeled outputs are perturbed to emulate practical measurements,
while the collocation datasets are unaffected, since they carry no measured values.
Setting phys_weight and bnd_weight to 0.0 in train() gives a purely data-driven
baseline with the same data, architecture and seed.

The data loaders, network, optimizer and outputs shared by the isothermal PFR
cases are in ../common/train.py; this file holds the settings of this case.
"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
from pathlib import Path

from pinnse import Noise

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # examples/isopfr

from common.process import E, k, key_species, nu, species
from common.train import load_data, normalize, save_outputs, train
from phys_res import Boundary, Physics

# ----------------------------- Run configuration -----------------------------
NOISE_LEVEL = 0.05  # Noise magnitude; interpretation depends on NOISE_MODE
NOISE_MODE = "relative"  # "relative" | "proportional" | "absolute"
NOISE_SEED = 42  # Seed of the noise realization


def add_noise(D_S_data):
    """
    Perturb the labeled outputs to emulate measurements. Effluent flowrates
    are non-negative, so the perturbed values are clipped at zero.
    """
    D_S_noisy, noise_metrics = Noise.gaussian(
        data=D_S_data,
        noise_level=NOISE_LEVEL,
        mode=NOISE_MODE,
        clip_min=0.0,
        random_state=NOISE_SEED,
    )
    return D_S_noisy


def main():
    I_S_data, D_S_data = load_data()
    D_S_noisy = add_noise(D_S_data)

    # Normalization metrics are derived from the noisy dataset,
    # since that is the dataset available in practice.
    norm_I_S_data, norm_D_S_data, I_S_metrics, D_S_metrics = normalize(
        "EFM", I_S_data, D_S_noisy
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
