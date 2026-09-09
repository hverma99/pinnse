import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch, torch.nn as nn, numpy as np, pandas as pd
from pinnse import Normalization, Save
from pinnse import SequenceDataModule
from pinnse import RecurrentANN
from pinnse import Training
from pinnse import Plotter

import cstr_model as cstr
from phys_res import Physics

"""
Recurrent discrete-time PINN model for a non-isothermal CSTR.

The model maps an initial condition and a coolant schedule to the state
trajectory they produce, predicting the whole trajectory in one forward pass,
with the dynamics enforced by an unrolled discrete-time residual applied at
every step.
"""

# ----------------------------- Run configuration -----------------------------
SCHEME = "euler"  # "euler" | "rk4" | "bwd_euler" | "trapezoid"
CELL = "gru"  # "rnn" | "gru" | "lstm"

hidden, layers = 64, 1
head_layers = [64]
epochs = 500
LR = 1e-3

B_D, B_C_P = 64, 64  # trajectories per batch
N_C_P = 1000  # collocation trajectories
PHYS_WEIGHT = 1.0


def load_sequences(filename: str, keys: list[str]):
    """
    Read data and restore the (N, SEQ_LEN, features) array.

    The trajectories are stored one row per (trajectory, step).
    """
    df = pd.read_excel(filename).sort_values(["traj", "step"])
    n_traj = df["traj"].nunique()
    seq_len = len(df) // n_traj

    if len(df) != n_traj * seq_len:
        raise ValueError(f"{filename}: rows are not a multiple of the trajectory count")
    if seq_len != cstr.SEQ_LEN:
        raise ValueError(
            f"{filename}: stored horizon is {seq_len} steps but cstr.SEQ_LEN is "
            f"{cstr.SEQ_LEN}; regenerate the data with data_gen.py"
        )

    return df[keys].to_numpy(dtype=np.float64).reshape(n_traj, seq_len, len(keys))


def normalize(A: np.ndarray, keys: list[str], metrics: dict):
    """Min-max normalize a sequence array column by column, preserving shape."""
    out = np.empty_like(A)
    for j, key in enumerate(keys):
        lo, hi = metrics[key]["min"], metrics[key]["max"]
        out[..., j] = (A[..., j] - lo) / (hi - lo)
    return out


def prepare():
    """
    Load the trajectories, returning the dimensional and normalized arrays
    alongside the normalization metrics.

    The metrics are computed from the sequences flattened across trajectories
    and time steps, so the package utility applies unchanged, and over the whole
    labeled dataset.
    """
    I_S_data = load_sequences(r"I_S_data.xlsx", cstr.I_S_keys)
    D_S_data = load_sequences(r"D_S_data.xlsx", cstr.D_S_keys)

    _, I_S_metrics = Normalization.min_max(
        pd.DataFrame(I_S_data.reshape(-1, len(cstr.I_S_keys)), columns=cstr.I_S_keys)
    )
    _, D_S_metrics = Normalization.min_max(
        pd.DataFrame(D_S_data.reshape(-1, len(cstr.D_S_keys)), columns=cstr.D_S_keys)
    )

    norm_I_S_data = normalize(I_S_data, cstr.I_S_keys, I_S_metrics)
    norm_D_S_data = normalize(D_S_data, cstr.D_S_keys, D_S_metrics)
    return I_S_data, D_S_data, norm_I_S_data, norm_D_S_data, I_S_metrics, D_S_metrics


def data_module(
    norm_I_S_data: np.ndarray,
    norm_D_S_data: np.ndarray,
    labeled_data_batch_size: int = B_D,
    physics_coll_data_size: int | None = None,
    physics_coll_batch_size: int | None = None,
):
    """
    Construct the sequence data module with this example's split settings, so
    that training and every evaluation script partition the trajectories
    identically.
    """
    return SequenceDataModule(
        I_S_data=norm_I_S_data,
        D_S_data=norm_D_S_data,
        labeled_data_batch_size=labeled_data_batch_size,
        physics_coll_data_size=physics_coll_data_size,
        physics_coll_batch_size=physics_coll_batch_size,
        n_segments=cstr.N_SEG,
        test_frac=0.1,
        val_frac=0.1,
        random_state=42,
    )


def build(device):
    """Assemble the loaders, model and residual shared with the evaluation scripts."""
    _, _, norm_I_S_data, norm_D_S_data, I_S_metrics, D_S_metrics = prepare()

    data = data_module(
        norm_I_S_data=norm_I_S_data,
        norm_D_S_data=norm_D_S_data,
        labeled_data_batch_size=B_D,
        physics_coll_data_size=N_C_P,
        physics_coll_batch_size=B_C_P,
    )

    train_loader, val_loader, test_loader = data.labeled_data_loader()
    phys_coll_loader = data.phys_colloc_loader()

    model = RecurrentANN(
        in_dim=len(cstr.I_S_keys),
        hidden_dim=hidden,
        out_dim=len(cstr.D_S_keys),
        n_layers=layers,
        cell=CELL,
        head_layers=head_layers,
        activation=nn.Tanh,
    ).to(device)

    physics = Physics(I_S_metrics, D_S_metrics, scheme=SCHEME)

    return (
        model,
        physics,
        (train_loader, val_loader, test_loader),
        phys_coll_loader,
        I_S_metrics,
        D_S_metrics,
    )


def main():
    torch.manual_seed(1234)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    folder_name = "logs"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    model, physics, loaders, phys_coll_loader, _, _ = build(device)
    train_loader, val_loader, test_loader = loaders

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    training = Training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        phys_coll_loader=phys_coll_loader,
        phys_residual=physics,
        ckpt_path="./logs/best_model.pth",
        phys_weight=PHYS_WEIGHT,
        bnd_weight=0.0,
        adapt_wts=False,
    )

    history = training.adam_step(epochs=epochs, val_every=50, verbose=True)
    return history


if __name__ == "__main__":
    history = main()

    Save.excel(history)
    Save.csv(history)

    folder_name = "figures"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    plots = Plotter(history=history, val_every=50)
    plots.plot_everything(savepath=folder_name, scale=100)
