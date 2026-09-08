import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch, torch.nn as nn, numpy as np, pandas as pd
from pinnse import Normalization, Noise, Save
from pinnse import DataModule
from pinnse import ANN
from pinnse import Training
from pinnse import Plotter
from data_gen import species
from phys_res import Physics, Boundary

"""
Effluent-flow-based (EFM) PINN model for the isothermal PFR process, trained
against labeled data perturbed by measurement noise.

This is an example where labeled outputs are perturbed to emulate practical measurements,
while the collocation datasets are unaffected, since they carry no measured values.
Setting PHYSICS_ON to False sets the physics and boundary weights to zero,
which reduces the total loss to the supervised data loss and recovers a conventional neural network;
this provides a purely data-driven baseline trained on identical data, with an
identical architecture and seed, so that the contribution of the enforced physics is isolated.
"""

# ----------------------------- Run configuration -----------------------------
NOISE_LEVEL = 0.05  # Noise magnitude; interpretation depends on NOISE_MODE
NOISE_MODE = "relative"  # "relative" | "proportional" | "absolute"
NOISE_SEED = 42  # Seed of the noise realization


def main():
    torch.manual_seed(1234)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Process and reaction system specifications
    species = ["O2", "CO2", "H2O", "C6H6", "C4H2O3"]
    N_species = len(species)
    nu = np.array(
        [
            [-4.5, 2.0, 2.0, -1.0, 1.0],
            [-7.5, 6.0, 3.0, -1.0, 0.0],
            [-3.0, 4.0, 1.0, 0.0, -1.0],
        ],
        dtype=np.float32,
    )
    key_species = ["C6H6", "C6H6", "C4H2O3"]
    spec_index = {sp: i for i, sp in enumerate(species)}
    k = np.array([7.7e6, 6.31e7, 2.33e4], dtype=np.float32)  # 1/s
    E = np.array([105.2, 124.89, 89.66], dtype=np.float32) * 1e3  # J/mol
    R = 8.314  # J/mol/K

    folder_name = "logs"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    I_S_data = pd.read_excel(r"I_S_data.xlsx")
    D_S_data = pd.read_excel(r"D_S_data.xlsx")

    # Perturb the labeled outputs to emulate measurements.
    # Effluent flowrates are non-negative, so the perturbed values are clipped at zero.
    D_S_noisy, noise_metrics = Noise.gaussian(
        data=D_S_data,
        noise_level=NOISE_LEVEL,
        mode=NOISE_MODE,
        clip_min=0.0,
        random_state=NOISE_SEED,
    )

    dim_in, dim_ot = I_S_data.shape[1], D_S_noisy.shape[1]

    # Normalization metrics are derived from the noisy dataset,
    # since that is the dataset available in practice.
    norm_I_S_data, norm_D_S_data, I_S_metrics, D_S_metrics = Normalization.min_max_pfr(
        I_S_data=I_S_data, D_S_data=D_S_noisy, formulation="EFM", species=species
    )

    physics = Physics(I_S_metrics, D_S_metrics, species, nu, key_species, k, E)
    boundary = Boundary(species)

    N_C_P, N_C_B = 20000, 5000
    B_D, B_C_P, B_C_B = 500, 500, 1000

    data = DataModule(
        I_S_data=norm_I_S_data,
        D_S_data=norm_D_S_data,
        labeled_data_batch_size=B_D,
        physics_coll_data_size=N_C_P,
        physics_coll_batch_size=B_C_P,
        boundary_coll_data_size=N_C_B,
        boundry_coll_batch_size=B_C_B,
        test_frac=0.1,
        val_frac=0.1,
    )

    train_loader, val_loader, test_loader = data.labeled_data_loader()
    phys_coll_loader = data.phys_colloc_loader()
    bnd_coll_loader = data.bnd_colloc_loader()

    depth, width = 6, 64
    layer_size = [dim_in] + [width] * depth + [dim_ot]

    model = ANN(layer_size, activation=nn.Tanh).to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    training = Training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        phys_coll_loader=phys_coll_loader,
        bnd_coll_loader=bnd_coll_loader,
        phys_residual=physics,
        bnd_residual=boundary,
        ckpt_path="./logs/best_model.pth",
        phys_weight=0.0,
        bnd_weight=0.0,
        adapt_wts=False,
    )

    history = training.adam_step(epochs=75000, val_every=100, verbose=True)
    return history


if __name__ == "__main__":
    history = main()

    Save.excel(history)
    Save.csv(history)

    folder_name = "figures"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    plots = Plotter(history=history, val_every=100)
    plots.plot_everything(savepath=folder_name, scale=1000)
