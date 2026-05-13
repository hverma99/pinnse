import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from data_gen import species, nu, key_species, k, E
from pfr_model import ERM
from phys_res import Physics
from utils import Normalization, Analyze


def build_profile_inputs(
    F_in: np.ndarray,
    P: float,
    T: float,
    V: float,
    N_points: int,
    species: list[str],
):
    """
    Build a profile input DataFrame along the reactor volume.
    """
    V_eval = np.linspace(0.0, V, N_points, dtype=np.float32)
    I_S = pd.DataFrame(
        {
            **{
                f"F_in_{sp}": np.full(N_points, F_in[i], dtype=np.float32)
                for i, sp in enumerate(species)
            },
            "P": np.full(N_points, P, dtype=np.float32),
            "T": np.full(N_points, T, dtype=np.float32),
            "V": V_eval,
        }
    )
    return I_S, V_eval


def truth_eval(
    F_in: np.ndarray,
    P: float,
    T: float,
    V_eval: np.ndarray,
    species: list[str],
    nu: np.ndarray,
    key_species: list[str],
    k: np.ndarray,
    E: np.ndarray,
):
    """
    Evaluate the first-principles ERM model along the reactor profile.
    """
    model = ERM(species, nu, key_species, k, E, F_in, P, T)
    sol = model.solve(V_end=float(V_eval[-1]), V_eval=V_eval)
    E_truth = sol.y.T.astype(np.float32)

    return pd.DataFrame(E_truth, columns=[f"E_R{j+1}" for j in range(nu.shape[0])])


def plot_extent_profile(
    V_eval: np.ndarray,
    D_S_truth: pd.DataFrame,
    D_S_pinn: pd.DataFrame,
    save_dir: str,
    fontsize: int = 19,
    fontfamily: str = "Arial",
):
    """
    Plot truth vs PINN extent-of-reaction profiles.
    """
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    reactions = ["E_R1", "E_R2", "E_R3"]
    label_reactions = [r"$E_{R1}$", r"$E_{R2}$", r"$E_{R3}$"]
    colors = ["red", "blue", "green"]
    color_map = {rxn: col for rxn, col in zip(reactions, colors)}

    for rxn, color in zip(reactions, colors):
        ax.plot(
            V_eval,
            D_S_truth[rxn],
            linestyle="--",
            linewidth=1.5,
            color=color,
        )
        ax.plot(
            V_eval,
            D_S_pinn[rxn],
            linestyle="-",
            linewidth=1.5,
            color=color,
        )

    legend1 = ax.legend(
        handles=[
            Line2D(
                [0], [0], color="black", linestyle="--", linewidth=1.0, label="Truth"
            ),
            Line2D([0], [0], color="black", linestyle="-", linewidth=1.0, label="PINN"),
        ],
        loc="center right",
        frameon=False,
        prop={"family": fontfamily, "size": fontsize - 2},
    )

    reaction_lines = [
        Line2D(
            [0], [0], color=color_map[rxn], linestyle="-", linewidth=1.5, label=label
        )
        for rxn, label in zip(reactions, label_reactions)
    ]
    legend2 = ax.legend(
        handles=reaction_lines,
        loc="upper left",
        frameon=False,
        prop={"family": fontfamily, "size": fontsize - 2},
    )

    ax.add_artist(legend1)
    ax.add_artist(legend2)

    ax.set_xlabel(
        r"Reactor Volume ($\mathrm{m}^3$)",
        fontsize=fontsize,
        fontfamily=fontfamily,
    )
    ax.set_ylabel(
        "Extent of Reaction (mol/s)",
        fontsize=fontsize,
        fontfamily=fontfamily,
    )
    ax.tick_params(
        axis="x", which="major", labelsize=fontsize, labelfontfamily=fontfamily
    )
    ax.tick_params(
        axis="y", which="major", labelsize=fontsize, labelfontfamily=fontfamily
    )

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "extent_profile.png"), bbox_inches="tight")
    plt.close()


def plot_mae_profile(
    V_eval: np.ndarray,
    D_S_truth: pd.DataFrame,
    D_S_pinn: pd.DataFrame,
    save_dir: str,
    fontsize: int = 19,
    fontfamily: str = "Arial",
):
    """
    Plot absolute error profile along reactor volume.
    """
    mae = np.abs(D_S_truth.to_numpy() - D_S_pinn.to_numpy())

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    reactions = ["E_R1", "E_R2", "E_R3"]
    label_reactions = [r"$E_{R1}$", r"$E_{R2}$", r"$E_{R3}$"]
    colors = ["firebrick", "darkcyan", "slateblue"]

    reaction_lines = []
    for j, (rxn, label, color) in enumerate(zip(reactions, label_reactions, colors)):
        ax.plot(V_eval, mae[:, j], color=color, linewidth=1.5)
        reaction_lines.append(
            Line2D([0], [0], color=color, linestyle="-", linewidth=1.5, label=label)
        )

    ax.legend(
        handles=reaction_lines,
        loc="upper left",
        frameon=False,
        prop={"family": fontfamily, "size": fontsize - 2},
    )

    ax.set_xlabel(
        r"Reactor Volume ($\mathrm{m}^3$)",
        fontsize=fontsize,
        fontfamily=fontfamily,
    )
    ax.set_ylabel(
        "Absolute Error (mol/s)",
        fontsize=fontsize,
        fontfamily=fontfamily,
    )
    ax.tick_params(
        axis="x", which="major", labelsize=fontsize, labelfontfamily=fontfamily
    )
    ax.tick_params(
        axis="y", which="major", labelsize=fontsize, labelfontfamily=fontfamily
    )

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "mae_profile.png"), bbox_inches="tight")
    plt.close()

    return mae


def plot_phys_residual(
    I_S: pd.DataFrame,
    model: nn.Module,
    physics: Physics,
    I_S_metrics: dict,
    device: torch.device,
    save_dir: str,
    fontsize: int = 19,
    fontfamily: str = "Arial",
):
    """
    Evaluate and plot physics residuals along reactor volume.
    """
    I_S_norm = Normalization.scale_centered_defined_metrics(I_S, I_S_metrics)
    x = torch.tensor(
        I_S_norm.to_numpy(), dtype=torch.float32, device=device
    ).requires_grad_(True)
    y = model(x)
    res = physics(x, y).detach().cpu().numpy()

    res_abs = np.abs(res)
    res_l2 = np.linalg.norm(res, axis=1)
    V_eval = I_S["V"].to_numpy()

    reactions = [r"$R_1$", r"$R_2$", r"$R_3$"]
    colors = ["firebrick", "darkcyan", "slateblue"]

    # per-reaction residual
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    reaction_lines = []
    for j, (label, color) in enumerate(zip(reactions, colors)):
        ax.plot(V_eval, res_abs[:, j], color=color, linewidth=1.5)
        reaction_lines.append(
            Line2D([0], [0], color=color, linestyle="-", linewidth=1.5, label=label)
        )

    ax.legend(
        handles=reaction_lines,
        loc="upper left",
        frameon=False,
        prop={"family": fontfamily, "size": fontsize - 2},
    )

    ax.set_xlabel(
        r"Reactor Volume ($\mathrm{m}^3$)",
        fontsize=fontsize,
        fontfamily=fontfamily,
    )
    ax.set_ylabel(
        r"Physics Residual ($|L_{P,j}|$)",
        fontsize=fontsize,
        fontfamily=fontfamily,
    )
    ax.tick_params(
        axis="x", which="major", labelsize=fontsize, labelfontfamily=fontfamily
    )
    ax.tick_params(
        axis="y", which="major", labelsize=fontsize, labelfontfamily=fontfamily
    )

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "phys_residual_per_reaction.png"),
        bbox_inches="tight",
    )
    plt.close()

    # overall residual
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    ax.semilogy(V_eval, res_l2, linestyle="-.", linewidth=1.5, color="black")

    legend1 = ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color="black",
                linestyle="-.",
                linewidth=1.5,
                label=r"$\|L_P\|_2$",
            )
        ],
        loc="upper center",
        frameon=False,
        prop={"family": fontfamily, "size": fontsize - 2},
    )
    ax.add_artist(legend1)

    ax.set_xlabel(
        r"Reactor Volume ($\mathrm{m}^3$)",
        fontsize=fontsize,
        fontfamily=fontfamily,
    )
    ax.set_ylabel(
        r"Physics Residual ($\|L_P\|_2$)",
        fontsize=fontsize,
        fontfamily=fontfamily,
    )
    ax.tick_params(
        axis="x", which="major", labelsize=fontsize, labelfontfamily=fontfamily
    )
    ax.tick_params(
        axis="y", which="major", labelsize=fontsize, labelfontfamily=fontfamily
    )

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "phys_residual_overall.png"),
        bbox_inches="tight",
    )
    plt.close()

    return res_abs, res_l2


def save_results(
    I_S: pd.DataFrame,
    D_S_truth: pd.DataFrame,
    D_S_pinn: pd.DataFrame,
    mae: np.ndarray,
    res_abs: np.ndarray,
    res_l2: np.ndarray,
    metrics_df: pd.DataFrame,
    save_dir: str,
):
    """
    Save profile results to Excel.
    """
    V_eval = I_S["V"].to_numpy()

    truth_sheet = pd.concat(
        [I_S.reset_index(drop=True), D_S_truth.reset_index(drop=True)], axis=1
    )
    pinn_sheet = pd.concat(
        [I_S.reset_index(drop=True), D_S_pinn.reset_index(drop=True)], axis=1
    )

    mae_sheet = pd.DataFrame({"V": V_eval})
    for j, col in enumerate(D_S_truth.columns):
        mae_sheet[f"MAE_{col}"] = mae[:, j]

    res_sheet = pd.DataFrame({"V": V_eval})
    for j in range(res_abs.shape[1]):
        res_sheet[f"Residual_R{j+1}"] = res_abs[:, j]

    res_overall_sheet = pd.DataFrame({"V": V_eval, "Residual_Overall": res_l2})

    with pd.ExcelWriter(os.path.join(save_dir, "compare.xlsx")) as writer:
        truth_sheet.to_excel(writer, sheet_name="Truth", index=False)
        pinn_sheet.to_excel(writer, sheet_name="PINN", index=False)
        mae_sheet.to_excel(writer, sheet_name="MAE", index=False)
        res_sheet.to_excel(writer, sheet_name="Physics_Residual", index=False)
        res_overall_sheet.to_excel(writer, sheet_name="Residual_Overall", index=False)
        metrics_df.to_excel(writer, sheet_name="Error_Metrics", index=False)


if __name__ == "__main__":
    torch.manual_seed(1234)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    I_S_data = pd.read_excel("I_S_data.xlsx")
    D_S_data = pd.read_excel("D_S_data.xlsx")

    _, _, I_S_metrics, D_S_metrics = Normalization.min_max_pfr(
        I_S_data=I_S_data,
        D_S_data=D_S_data,
        formulation="ERM",
        species=species,
        N_reaction=nu.shape[0],
    )

    model = Analyze.load_ann_pinn(
        dim_in=I_S_data.shape[1],
        dim_out=D_S_data.shape[1],
        depth=6,
        width=64,
        activation=nn.Tanh,
        ckpt_path="./logs/best_model.pth",
        device=device,
    )

    physics = Physics(I_S_metrics, D_S_metrics, species, nu, key_species, k, E)

    save_dir = "compare"
    os.makedirs(save_dir, exist_ok=True)

    # ------------------ Case-specific process specifications ------------------
    F_in = np.array([1000.0, 40.0, 25.0, 80.0, 20.0], dtype=np.float32)
    P = 1.75e5
    T = 425.0 + 273.15
    V = 250.0
    N_points = 100
    # -------------------------------------------------------------------------

    I_S_prof, V_eval = build_profile_inputs(F_in, P, T, V, N_points, species)

    D_S_truth = truth_eval(
        F_in=F_in,
        P=P,
        T=T,
        V_eval=V_eval,
        species=species,
        nu=nu,
        key_species=key_species,
        k=k,
        E=E,
    )

    D_S_pinn = Analyze.pinn_eval_pfr(
        model=model,
        I_S_model=I_S_prof,
        I_S_metrics=I_S_metrics,
        D_S_metrics=D_S_metrics,
        output_cols=list(D_S_data.columns),
        device=device,
    )

    groups = {col: [col] for col in D_S_data.columns}
    groups["Overall"] = list(D_S_data.columns)
    metrics_df = Analyze.error_metrics(D_S_truth, D_S_pinn, groups)

    mae = plot_mae_profile(V_eval, D_S_truth, D_S_pinn, save_dir)
    plot_extent_profile(V_eval, D_S_truth, D_S_pinn, save_dir)
    res_abs, res_l2 = plot_phys_residual(
        I_S_prof, model, physics, I_S_metrics, device, save_dir
    )

    save_results(
        I_S=I_S_prof,
        D_S_truth=D_S_truth,
        D_S_pinn=D_S_pinn,
        mae=mae,
        res_abs=res_abs,
        res_l2=res_l2,
        metrics_df=metrics_df,
        save_dir=save_dir,
    )
