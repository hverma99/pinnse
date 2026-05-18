import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch, torch.nn as nn, pandas as pd

from phys_res import Physics, Boundary
from pinnse import DataModule
from pinnse import SANN
from pinnse import Training
from pinnse import Save
from pinnse import Plotter


def main():
    torch.manual_seed(1234)

    folder_name = "logs"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    I_S_data = pd.read_excel("I_S_data.xlsx")
    D_S_data = pd.read_excel("D_S_data.xlsx")

    dim_in, dim_ot = I_S_data.shape[1], D_S_data.shape[1]

    N_C_P, N_C_B = 100000, 5000
    B_D, B_C_P, B_C_B = 1000, 4000, 5000

    data = DataModule(
        I_S_data,
        D_S_data,
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
    bnd_coll_loader = data.bnd_colloc_loader(bnd_value=0)

    depth, width = 6, 64
    layer_size = [dim_in] + [width] * depth + [dim_ot]

    model = SANN(layer_size, activation=nn.Tanh).to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    species = ["A", "B", "C"]
    physics = Physics(
        species=species,
        delta_Hrxn={0: -20000, 1: -60000},
        Cp={"A": 90, "B": 90, "C": 180},
        Ua=4000.0,
        T_R=300,
        R=8.314,
        F_ch=200,
        C_ch=0.1,
        T_ch=800,
        V_ch=1,
        H_ch=-1e5,
        Cp_ch=200,
    )

    boundary = Boundary(species)

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
        phys_weight=1.0,
        bnd_weight=1.0,
        adapt_wts=False,
    )

    history_adam = training.adam_step(epochs=75000, val_every=100, verbose=True)
    history_lbfgs = training.lbfgs_step(mode="combined", N_LBFGS=150, verbose=True)
    return history_adam, history_lbfgs


if __name__ == "__main__":
    history_adam, history_lbfgs = main()

    Save.excel(history_adam, filename="adam_history.xlsx")
    Save.excel(history_lbfgs, filename="lbfgs_history.xlsx")

    Save.csv(history=history_adam, filename="csv_adam")
    Save.csv(history=history_lbfgs, filename="csv_lbfgs")

    folder_name = "figures"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    adam_plots = Plotter(history=history_adam, val_every=100)
    adam_plots.plot_everything(savepath=folder_name, scale=1000)

    lbfgs_plots = Plotter(history=history_lbfgs)
    lbfgs_plots.plot_all_train_losses(
        savepath=os.path.join(folder_name, "lbfgs_train.png")
    )
