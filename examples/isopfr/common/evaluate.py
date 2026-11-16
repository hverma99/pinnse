"""
Evaluation shared by the isothermal PFR cases (check.py).

The trained model is compared with the first-principles reactor model along
one reactor profile: the predicted and reference profiles, their absolute
error and the physics residual of the model are plotted, and saved with the
error metrics of Analyze.error_metrics to compare/compare.xlsx.

STYLES holds what differs between the formulations: the outputs plotted, their
labels, colors and axis titles, and the file and column names.
"""

import os
from dataclasses import dataclass

import torch
import torch.nn as nn

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from pinnse import Analyze, Normalization

from common.pfr_model import MODELS
from common.process import E, k, key_species, nu, output_columns, species
from common.train import CKPT_PATH, DEPTH, DEVICE, WIDTH

SAVE_DIR = "compare"

# ------------------ Reactor profile used for the comparison ------------------
PROFILE_F_IN = np.array([1000.0, 40.0, 25.0, 80.0, 20.0], dtype=np.float32)
PROFILE_P = 1.75e5
PROFILE_T = 425.0 + 273.15
PROFILE_V = 250.0
PROFILE_N_POINTS = 100
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class Style:
    """How check.py plots and names the outputs of one formulation."""

    columns: list[str]  # D_S columns plotted in the profile and error figures
    labels: list[str]
    colors: list[str]  # profile figure
    error_colors: list[str]  # absolute-error figure
    residual_labels: list[str]  # one per plotted column
    residual_colors: list[str]
    profile_ylabel: str
    profile_file: str
    error_ylabel: str
    residual_ylabel: str
    residual_file: str
    output_names: list[str]  # MAE_<name> columns of compare.xlsx
    residual_names: list[str]  # Residual_<name> columns of compare.xlsx


_SPECIES = ["CO2", "H2O", "C6H6", "C4H2O3"]  # O2, in large excess, is not plotted
_SPECIES_LABELS = [r"CO$_2$", r"H$_2$O", r"C$_6$H$_6$", r"C$_4$H$_2$O$_3$"]
_SPECIES_COLORS = ["lawngreen", "darkcyan", "slateblue", "mediumorchid"]
_REACTION_LABELS = [r"$R_1$", r"$R_2$", r"$R_3$"]
_REACTION_COLORS = ["firebrick", "darkcyan", "slateblue"]
_REACTION_NAMES = [f"R{j + 1}" for j in range(nu.shape[0])]

STYLES = {
    "EFM": Style(
        columns=[f"F_ot_{sp}" for sp in _SPECIES],
        labels=_SPECIES_LABELS,
        colors=_SPECIES_COLORS,
        error_colors=_SPECIES_COLORS,
        residual_labels=_SPECIES_LABELS,
        residual_colors=_SPECIES_COLORS,
        profile_ylabel="Effluent Molar Flowrate (mol/s)",
        profile_file="effluent_flow.png",
        error_ylabel="Absolute Error (mol/s)",
        residual_ylabel=r"Physics Residual ($|L_{P,i}|$)",
        residual_file="phys_residual_per_species.png",
        output_names=species,
        residual_names=species,
    ),
    "ERM": Style(
        columns=output_columns("ERM"),
        labels=[r"$E_{R1}$", r"$E_{R2}$", r"$E_{R3}$"],
        colors=["red", "blue", "green"],
        error_colors=_REACTION_COLORS,
        residual_labels=_REACTION_LABELS,
        residual_colors=_REACTION_COLORS,
        profile_ylabel="Extent of Reaction (mol/s)",
        profile_file="extent_profile.png",
        error_ylabel="Absolute Error (mol/s)",
        residual_ylabel=r"Physics Residual ($|L_{P,j}|$)",
        residual_file="phys_residual_per_reaction.png",
        output_names=output_columns("ERM"),
        residual_names=_REACTION_NAMES,
    ),
    "CM": Style(
        columns=output_columns("CM"),
        labels=[r"$X_{R1}$", r"$X_{R2}$", r"$X_{R3}$"],
        colors=_REACTION_COLORS,
        error_colors=_REACTION_COLORS,
        residual_labels=_REACTION_LABELS,
        residual_colors=_REACTION_COLORS,
        profile_ylabel="Reaction Conversion",
        profile_file="conversion_profile.png",
        error_ylabel="Absolute Error",
        residual_ylabel=r"Physics Residual ($|L_{P,j}|$)",
        residual_file="phys_residual_per_reaction.png",
        output_names=output_columns("CM"),
        residual_names=_REACTION_NAMES,
    ),
}


# ------------------------------- Evaluation ---------------------------------


def load_model(I_S_data: pd.DataFrame, D_S_data: pd.DataFrame) -> nn.Module:
    """Load the trained network of the case folder from its checkpoint."""
    return Analyze.load_ann_pinn(
        dim_in=I_S_data.shape[1],
        dim_out=D_S_data.shape[1],
        depth=DEPTH,
        width=WIDTH,
        activation=nn.Tanh,
        ckpt_path=CKPT_PATH,
        device=DEVICE,
    )


def build_profile_inputs():
    """
    Build a profile input DataFrame along the reactor volume.
    """
    V_eval = np.linspace(0.0, PROFILE_V, PROFILE_N_POINTS, dtype=np.float32)
    I_S = pd.DataFrame(
        {
            **{
                f"F_in_{sp}": np.full(
                    PROFILE_N_POINTS, PROFILE_F_IN[i], dtype=np.float32
                )
                for i, sp in enumerate(species)
            },
            "P": np.full(PROFILE_N_POINTS, PROFILE_P, dtype=np.float32),
            "T": np.full(PROFILE_N_POINTS, PROFILE_T, dtype=np.float32),
            "V": V_eval,
        }
    )
    return I_S, V_eval


def truth_eval(formulation: str, I_S: pd.DataFrame, V_eval: np.ndarray):
    """
    Evaluate the first-principles model of a formulation along the reactor
    profile.

    The EFM reference solves the model separately up to each profile point,
    from the (float32) profile inputs; the ERM and CM references integrate
    once over the profile and report the solution at V_eval.
    """
    Model = MODELS[formulation]
    columns = output_columns(formulation)

    if formulation != "EFM":
        model = Model(
            species, nu, key_species, k, E, PROFILE_F_IN, PROFILE_P, PROFILE_T
        )
        sol = model.solve(V_end=float(V_eval[-1]), V_eval=V_eval)
        return pd.DataFrame(sol.y.T.astype(np.float32), columns=columns)

    F_cols = [f"F_in_{sp}" for sp in species]
    F_in_all = I_S[F_cols].to_numpy(dtype=np.float32)
    P_all = I_S["P"].to_numpy(dtype=np.float32)
    T_all = I_S["T"].to_numpy(dtype=np.float32)
    V_all = I_S["V"].to_numpy(dtype=np.float32)

    Y = np.zeros((len(I_S), len(columns)), dtype=np.float32)

    for i in range(len(I_S)):
        F_in = F_in_all[i]
        P = float(P_all[i])
        T = float(T_all[i])
        V = float(V_all[i])

        model = Model(species, nu, key_species, k, E, F_in, P, T)
        Y[i, :] = model.solve(V).y[:, -1]

    return pd.DataFrame(Y, columns=columns)


def pinn_eval(model, I_S, I_S_metrics, D_S_metrics, output_cols):
    """Predictions of the trained model at I_S, in dimensional units."""
    return Analyze.pinn_eval_pfr(
        model=model,
        I_S_model=I_S,
        I_S_metrics=I_S_metrics,
        D_S_metrics=D_S_metrics,
        output_cols=list(output_cols),
        device=DEVICE,
    )


def metric_groups(output_cols) -> dict[str, list[str]]:
    """Error-metric groups: every output on its own, then all together."""
    groups = {col: [col] for col in output_cols}
    groups["Overall"] = list(output_cols)
    return groups


def phys_residual_eval(I_S, model, physics, I_S_metrics):
    """
    Evaluate the physics residual of the trained model at I_S.

    Returns the absolute residual per output and its l2 norm per point.
    """
    I_S_norm = Normalization.scale_centered_defined_metrics(I_S, I_S_metrics)
    x = torch.tensor(
        I_S_norm.to_numpy(), dtype=torch.float32, device=DEVICE
    ).requires_grad_(True)
    y = model(x)
    res = physics(x, y).detach().cpu().numpy()

    res_abs = np.abs(res)
    res_l2 = np.linalg.norm(res, axis=1)
    return res_abs, res_l2


# --------------------------------- Plots ------------------------------------


def _line_handles(labels, colors, linestyle="-", linewidth=1.5):
    return [
        Line2D(
            [0], [0], color=color, linestyle=linestyle, linewidth=linewidth, label=label
        )
        for label, color in zip(labels, colors)
    ]


def _finish(ax, ylabel, path, fontsize, fontfamily, ylim=None, yticks=None):
    """Axis titles and ticks shared by all figures; save and close."""
    ax.set_xlabel(
        r"Reactor Volume ($\mathrm{m}^3$)",
        fontsize=fontsize,
        fontfamily=fontfamily,
    )
    ax.set_ylabel(
        ylabel,
        fontsize=fontsize,
        fontfamily=fontfamily,
    )
    ax.tick_params(
        axis="x", which="major", labelsize=fontsize, labelfontfamily=fontfamily
    )
    ax.tick_params(
        axis="y", which="major", labelsize=fontsize, labelfontfamily=fontfamily
    )
    if ylim is not None:
        ax.set_ylim(*ylim)
    if yticks is not None:
        ax.set_yticks(yticks)

    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def plot_profile(
    V_eval: np.ndarray,
    D_S_truth: pd.DataFrame,
    D_S_pinn: pd.DataFrame,
    style: Style,
    save_dir: str = SAVE_DIR,
    model_label: str = "PINN",
    fontsize: int = 19,
    fontfamily: str = "Arial",
):
    """
    Plot truth vs model profiles of the outputs.
    """
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    for col, color in zip(style.columns, style.colors):
        ax.plot(
            V_eval,
            D_S_truth[col],
            linestyle="--",
            linewidth=1.5,
            color=color,
        )
        ax.plot(
            V_eval,
            D_S_pinn[col],
            linestyle="-",
            linewidth=1.5,
            color=color,
        )

    prop = {"family": fontfamily, "size": fontsize - 2}

    # Legend 1: line styles
    line_styles = [
        Line2D([0], [0], color="black", linestyle="--", linewidth=1.0, label="Truth"),
        Line2D(
            [0], [0], color="black", linestyle="-", linewidth=1.0, label=model_label
        ),
    ]
    legend1 = ax.legend(
        handles=line_styles, loc="center right", frameon=False, prop=prop
    )

    # Legend 2: output colors
    legend2 = ax.legend(
        handles=_line_handles(style.labels, style.colors),
        loc="upper left",
        frameon=False,
        prop=prop,
    )

    ax.add_artist(legend1)
    ax.add_artist(legend2)

    _finish(
        ax,
        style.profile_ylabel,
        os.path.join(save_dir, style.profile_file),
        fontsize,
        fontfamily,
    )


def _plot_columns(
    V_eval,
    values,
    indices,
    labels,
    colors,
    ylabel,
    path,
    fontsize,
    fontfamily,
):
    """One line per selected column of values along the reactor volume."""
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    for j, color in zip(indices, colors):
        ax.plot(V_eval, values[:, j], color=color, linewidth=1.5)

    ax.legend(
        handles=_line_handles(labels, colors),
        loc="upper left",
        frameon=False,
        prop={"family": fontfamily, "size": fontsize - 2},
    )
    _finish(ax, ylabel, path, fontsize, fontfamily)


def plot_mae_profile(
    V_eval: np.ndarray,
    D_S_truth: pd.DataFrame,
    D_S_pinn: pd.DataFrame,
    style: Style,
    save_dir: str = SAVE_DIR,
    fontsize: int = 19,
    fontfamily: str = "Arial",
):
    """
    Plot absolute error profile along reactor volume.
    """
    mae = np.abs(D_S_truth.to_numpy() - D_S_pinn.to_numpy())
    indices = [list(D_S_truth.columns).index(col) for col in style.columns]
    _plot_columns(
        V_eval,
        mae,
        indices,
        style.labels,
        style.error_colors,
        style.error_ylabel,
        os.path.join(save_dir, "mae_profile.png"),
        fontsize,
        fontfamily,
    )
    return mae


def plot_phys_residual(
    V_eval: np.ndarray,
    res_abs: np.ndarray,
    res_l2: np.ndarray,
    style: Style,
    output_cols,
    save_dir: str = SAVE_DIR,
    ylim=None,
    yticks=None,
    fontsize: int = 19,
    fontfamily: str = "Arial",
):
    """
    Plot physics residuals along reactor volume: per output, and overall.
    """
    # ---------------- per-output residual ----------------
    indices = [list(output_cols).index(col) for col in style.columns]
    _plot_columns(
        V_eval,
        res_abs,
        indices,
        style.residual_labels,
        style.residual_colors,
        style.residual_ylabel,
        os.path.join(save_dir, style.residual_file),
        fontsize,
        fontfamily,
    )

    # ---------------- overall residual ----------------
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    ax.semilogy(V_eval, res_l2, linestyle="-.", linewidth=1.5, color="black")

    legend1 = ax.legend(
        handles=_line_handles([r"$\|L_P\|_2$"], ["black"], linestyle="-."),
        loc="upper center",
        frameon=False,
        prop={"family": fontfamily, "size": fontsize - 2},
    )
    ax.add_artist(legend1)

    _finish(
        ax,
        r"Physics Residual ($\|L_P\|_2$)",
        os.path.join(save_dir, "phys_residual_overall.png"),
        fontsize,
        fontfamily,
        ylim=ylim,
        yticks=yticks,
    )


def save_results(
    I_S: pd.DataFrame,
    D_S_truth: pd.DataFrame,
    D_S_pinn: pd.DataFrame,
    mae: np.ndarray,
    res_abs: np.ndarray,
    res_l2: np.ndarray,
    metrics_df: pd.DataFrame,
    style: Style,
    save_dir: str = SAVE_DIR,
    extra_sheets: dict[str, pd.DataFrame] | None = None,
):
    """
    Save profile results to Excel; extra_sheets are appended after
    Error_Metrics.
    """
    V_eval = I_S["V"].to_numpy()

    truth_sheet = pd.concat(
        [I_S.reset_index(drop=True), D_S_truth.reset_index(drop=True)], axis=1
    )
    pinn_sheet = pd.concat(
        [I_S.reset_index(drop=True), D_S_pinn.reset_index(drop=True)], axis=1
    )

    mae_sheet = pd.DataFrame({"V": V_eval})
    res_sheet = pd.DataFrame({"V": V_eval})
    for j, name in enumerate(style.output_names):
        mae_sheet[f"MAE_{name}"] = mae[:, j]
    for j, name in enumerate(style.residual_names):
        res_sheet[f"Residual_{name}"] = res_abs[:, j]

    res_overall_sheet = pd.DataFrame({"V": V_eval, "Residual_Overall": res_l2})

    with pd.ExcelWriter(os.path.join(save_dir, "compare.xlsx")) as writer:
        truth_sheet.to_excel(writer, sheet_name="Truth", index=False)
        pinn_sheet.to_excel(writer, sheet_name="PINN", index=False)
        mae_sheet.to_excel(writer, sheet_name="MAE", index=False)
        res_sheet.to_excel(writer, sheet_name="Physics_Residual", index=False)
        res_overall_sheet.to_excel(writer, sheet_name="Residual_Overall", index=False)
        metrics_df.to_excel(writer, sheet_name="Error_Metrics", index=False)
        for name, sheet in (extra_sheets or {}).items():
            sheet.to_excel(writer, sheet_name=name, index=False)


# ------------------------------ check.py flow -------------------------------


def compare_profile(
    formulation: str,
    I_S_data: pd.DataFrame,
    D_S_data: pd.DataFrame,
    I_S_metrics: dict,
    D_S_metrics: dict,
    physics,
    residual_ylim=None,
    residual_yticks=None,
):
    """
    Compare the trained model of the case folder with the reactor model along
    the reactor profile; figures and compare.xlsx are written to compare/.

    Returns the error metrics of the profile.
    """
    style = STYLES[formulation]
    model = load_model(I_S_data, D_S_data)
    os.makedirs(SAVE_DIR, exist_ok=True)

    I_S_prof, V_eval = build_profile_inputs()
    D_S_truth = truth_eval(formulation, I_S_prof, V_eval)
    D_S_pinn = pinn_eval(model, I_S_prof, I_S_metrics, D_S_metrics, D_S_data.columns)

    metrics_df = Analyze.error_metrics(
        D_S_truth, D_S_pinn, metric_groups(D_S_data.columns)
    )

    mae = plot_mae_profile(V_eval, D_S_truth, D_S_pinn, style)
    plot_profile(V_eval, D_S_truth, D_S_pinn, style)
    res_abs, res_l2 = phys_residual_eval(I_S_prof, model, physics, I_S_metrics)
    plot_phys_residual(
        I_S_prof["V"].to_numpy(),
        res_abs,
        res_l2,
        style,
        D_S_data.columns,
        ylim=residual_ylim,
        yticks=residual_yticks,
    )

    save_results(
        I_S=I_S_prof,
        D_S_truth=D_S_truth,
        D_S_pinn=D_S_pinn,
        mae=mae,
        res_abs=res_abs,
        res_l2=res_l2,
        metrics_df=metrics_df,
        style=style,
    )
    return metrics_df
