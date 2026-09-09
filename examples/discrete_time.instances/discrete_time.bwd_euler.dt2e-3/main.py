import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch, torch.nn as nn, pandas as pd
from pinnse import Normalization, Save
from pinnse import DataModule
from pinnse import ANN
from pinnse import Training
from pinnse import Plotter
from phys_res import Physics

"""
Discrete-time (state-transition) PINN model for a non-isothermal CSTR.

The model maps the current state and manipulated variable to the state one
sampling interval later, and the dynamics are enforced by a discrete-time
residual rather than by automatic differentiation with respect to time, which
would not give the total derivative along the trajectory when the states are
themselves inputs.

SCHEME selects the integration scheme.
"""

# ----------------------------- Run configuration -----------------------------
SCHEME = "bwd_euler"  # "euler" | "rk4" | "bwd_euler" | "trapezoid"


def main():
    torch.manual_seed(1234)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    folder_name = "logs"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    I_S_data = pd.read_excel(r"I_S_data.xlsx")
    D_S_data = pd.read_excel(r"D_S_data.xlsx")

    dim_in, dim_ot = I_S_data.shape[1], D_S_data.shape[1]

    norm_I_S_data, I_S_metrics = Normalization.min_max(I_S_data)
    norm_D_S_data, D_S_metrics = Normalization.min_max(D_S_data)

    physics = Physics(I_S_metrics, D_S_metrics, scheme=SCHEME)

    N_C_P = 20000
    B_D, B_C_P = 500, 500

    data = DataModule(
        I_S_data=norm_I_S_data,
        D_S_data=norm_D_S_data,
        labeled_data_batch_size=B_D,
        physics_coll_data_size=N_C_P,
        physics_coll_batch_size=B_C_P,
        test_frac=0.1,
        val_frac=0.1,
    )

    train_loader, val_loader, test_loader = data.labeled_data_loader()
    phys_coll_loader = data.phys_colloc_loader()

    depth, width = 4, 64
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
        phys_residual=physics,
        ckpt_path="./logs/best_model.pth",
        phys_weight=1.0,
        bnd_weight=0.0,
        adapt_wts=False,
    )

    history = training.adam_step(epochs=20000, val_every=100, verbose=True)
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
