import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch, torch.nn as nn, numpy as np, pandas as pd
from pinnse import Normalization, Save
from pinnse import DataModule
from pinnse import ANN
from pinnse import Training
from phys_res import Physics
from pinnse import Plotter


def main():
    torch.manual_seed(1234)
    species = ["O2", "CO2", "H2O", "C6H6", "C4H2O3"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    folder_name = "logs"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    I_S_data = pd.read_excel(r"I_S_data.xlsx")
    D_S_data = pd.read_excel(r"D_S_data.xlsx")

    dim_in, dim_ot = I_S_data.shape[1], D_S_data.shape[1]

    norm_I_S_data, I_S_metrics = Normalization.min_max(
        I_S_data, normalize_cols=["T_flash", "P_flash", "T_feed", "P_feed", "F"]
    )
    norm_D_S_data, D_S_metrics = Normalization.min_max(D_S_data, normalize_cols=["Q"])

    physics = Physics(I_S_metrics, D_S_metrics, species)

    N_C_P = 25000
    B_D, B_C = 500, 500

    data = DataModule(
        I_S_data=norm_I_S_data,
        D_S_data=norm_D_S_data,
        labeled_data_batch_size=B_D,
        physics_coll_data_size=N_C_P,
        physics_coll_batch_size=B_C,
        test_frac=0.1,
        val_frac=0.1,
        random_state=42,
    )

    train_loader, val_loader, test_loader = data.labeled_data_loader()
    phys_coll_loader = data.phys_colloc_loader()

    depth, width = 6, 64
    layer_size = [dim_in] + [width] * depth + [dim_ot]

    model = ANN(layer_size, activation=nn.Tanh).to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    trainer = Training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        phys_coll_loader=phys_coll_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        phys_residual=physics,
        ckpt_path="./logs/best_model.pth",
        phys_weight=1.0,
    )

    history = trainer.adam_step(epochs=50000, val_every=100, verbose=True)
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
