"""
Training pipeline shared by the isothermal PFR cases.

A case's main.py loads and normalizes the labeled data, builds its physics and
boundary residuals, and calls train() with its own options. train() builds the
data loaders, the network and the optimizer, all identical across the cases,
and runs Adam. save_outputs() writes the training history and its figures.
"""

import os

import torch
import torch.nn as nn

import pandas as pd

from pinnse import ANN, DataModule, Normalization, Plotter, Save, Training

from common.process import nu, species

SEED = 1234
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Collocation set sizes and batch sizes
N_C_P, N_C_B = 20000, 5000
B_D, B_C_P, B_C_B = 500, 500, 1000

# Network: DEPTH hidden layers of WIDTH neurons
DEPTH, WIDTH = 6, 64

EPOCHS, VAL_EVERY = 75000, 100
CKPT_PATH = "./logs/best_model.pth"


def load_data():
    """Read the labeled datasets of the case folder (the working directory)."""
    I_S_data = pd.read_excel(r"I_S_data.xlsx")
    D_S_data = pd.read_excel(r"D_S_data.xlsx")
    return I_S_data, D_S_data


def normalize(formulation: str, I_S_data: pd.DataFrame, D_S_data: pd.DataFrame):
    """
    Centered min-max normalization of a PFR formulation.

    Returns norm_I_S_data, norm_D_S_data, I_S_metrics, D_S_metrics.
    """
    return Normalization.min_max_pfr(
        I_S_data=I_S_data,
        D_S_data=D_S_data,
        formulation=formulation,
        species=species,
        N_reaction=nu.shape[0],
    )


def train(
    norm_I_S_data: pd.DataFrame,
    norm_D_S_data: pd.DataFrame,
    physics,
    boundary,
    *,
    lr: float = 1e-4,
    theta: nn.Parameter | None = None,
    scheduler=None,
    **options,
):
    """
    Train a PINN on the normalized labeled data with the given residuals.

    Inputs
    ------
    norm_I_S_data, norm_D_S_data : pd.DataFrame
        Normalized labeled inputs and outputs.
    physics, boundary : callable
        Physics and boundary residuals, (x, y) -> residual tensor.
    lr : float, optional, default=1e-4
        Adam learning rate.
    theta : nn.Parameter, optional
        Trainable physical parameters (inverse problems), optimized with the
        network and recorded by Training.
    scheduler : callable, optional
        Builds the learning-rate scheduler from the optimizer.
    **options
        Further Training arguments of the case (adapt_wts, log_grad_norms, ...).

    Returns
    -------
    history : dict
        Training history of Training.adam_step.
    """
    torch.manual_seed(SEED)
    os.makedirs("logs", exist_ok=True)

    data = DataModule(
        I_S_data=norm_I_S_data,
        D_S_data=norm_D_S_data,
        labeled_data_batch_size=B_D,
        physics_coll_data_size=N_C_P,
        physics_coll_batch_size=B_C_P,
        boundary_coll_data_size=N_C_B,
        boundary_coll_batch_size=B_C_B,
        test_frac=0.1,
        val_frac=0.1,
    )

    train_loader, val_loader, test_loader = data.labeled_data_loader()
    phys_coll_loader = data.phys_colloc_loader()
    bnd_coll_loader = data.bnd_colloc_loader()

    dim_in, dim_ot = norm_I_S_data.shape[1], norm_D_S_data.shape[1]
    layer_size = [dim_in] + [WIDTH] * DEPTH + [dim_ot]

    model = ANN(layer_size, activation=nn.Tanh).to(DEVICE)
    loss_fn = nn.MSELoss()
    params = list(model.parameters()) + ([theta] if theta is not None else [])
    optimizer = torch.optim.Adam(params, lr=lr)

    training = Training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=DEVICE,
        scheduler=scheduler(optimizer) if scheduler is not None else None,
        phys_coll_loader=phys_coll_loader,
        bnd_coll_loader=bnd_coll_loader,
        phys_residual=physics,
        bnd_residual=boundary,
        ckpt_path=CKPT_PATH,
        phys_weight=1.0,
        bnd_weight=1.0,
        theta=theta,
        **options,
    )

    history = training.adam_step(epochs=EPOCHS, val_every=VAL_EVERY, verbose=True)
    return history


def save_outputs(history: dict):
    """Save the training history (logs/) and plot it (figures/)."""
    Save.excel(history)
    Save.csv(history)

    folder_name = "figures"
    os.makedirs(folder_name, exist_ok=True)

    plots = Plotter(history=history, val_every=VAL_EVERY)
    plots.plot_everything(savepath=folder_name, scale=1000)
