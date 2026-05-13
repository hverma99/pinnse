import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from pinnse import Normalization, Analyze, Denormalization
from data_gen import D_S_samples


def build_grid_inputs(
    T_range: list[float],
    P_range: list[float],
    nT: int,
    nP: int,
    Z: list[float],
    F: float,
    T_feed: float,
    P_feed: float,
    species: list[str],
):
    """
    Build the full physical-space input grid for Aspen evaluation.

    Inputs
    ------
    T_range, P_range : list[float]
        Bounds for flash temperature and pressure.
    nT, nP : int
        Number of grid points in temperature and pressure.
    Z : list[float]
        Feed composition.
    F, T_feed, P_feed : float
        Fixed feed flowrate, temperature, and pressure.
    species : list[str]
        Component names.

    Returns
    -------
    I_S_full : pd.DataFrame
        Full input DataFrame required by D_S_samples().
    T_grid, P_grid : np.ndarray
        Temperature and pressure mesh grids.
    """
    T1 = np.linspace(T_range[0], T_range[1], nT, dtype=np.float32)
    P1 = np.linspace(P_range[0], P_range[1], nP, dtype=np.float32)
    T_grid, P_grid = np.meshgrid(T1, P1, indexing="xy")

    T_flat = T_grid.ravel()
    P_flat = P_grid.ravel()
    npts = T_flat.size

    I_S_full = pd.DataFrame(
        {
            "T_flash": T_flat,
            "P_flash": P_flat,
            "T_feed": np.full(npts, T_feed, dtype=np.float32),
            "P_feed": np.full(npts, P_feed, dtype=np.float32),
            "F": np.full(npts, F, dtype=np.float32),
            **{
                f"Z_{sp}": np.full(npts, Z[i], dtype=np.float32)
                for i, sp in enumerate(species)
            },
        }
    )

    return I_S_full, T_grid, P_grid


def plot_profiles(
    T_grid: np.ndarray,
    P_grid: np.ndarray,
    D_S_aspen: pd.DataFrame,
    D_S_pinn: pd.DataFrame,
    species: list[str],
    save_dir: str,
    P_target: float | None = None,
    T_target: float | None = None,
    fontfamily: str = "Arial",
    fontsize: int = 20,
):
    """
    Generate Aspen vs PINN profile plots and save profile Excel files.

    For fixed P_target:
        - combined Y_i + VF plot versus T_flash
        - X_i plot versus T_flash
        - Excel workbook: profiles_at_P_<value>.xlsx

    For fixed T_target:
        - combined Y_i + VF plot versus P_flash
        - X_i plot versus P_flash
        - Excel workbook: profiles_at_T_<value>.xlsx
    """
    os.makedirs(save_dir, exist_ok=True)

    nP, nT = T_grid.shape
    expected_pts = nP * nT

    if len(D_S_aspen) != expected_pts or len(D_S_pinn) != expected_pts:
        print(
            "Skipping profile plots because Aspen outputs do not cover the full grid "
            f"({len(D_S_aspen)} available vs {expected_pts} expected points)."
        )
        return

    T = T_grid[0, :]
    P = P_grid[:, 0]
    N_comp = len(species)

    VF_a = D_S_aspen["VF"].to_numpy().reshape(nP, nT)
    VF_p = D_S_pinn["VF"].to_numpy().reshape(nP, nT)

    Y_a = D_S_aspen[[f"Y_{sp}" for sp in species]].to_numpy().reshape(nP, nT, N_comp)
    Y_p = D_S_pinn[[f"Y_{sp}" for sp in species]].to_numpy().reshape(nP, nT, N_comp)

    X_a = D_S_aspen[[f"X_{sp}" for sp in species]].to_numpy().reshape(nP, nT, N_comp)
    X_p = D_S_pinn[[f"X_{sp}" for sp in species]].to_numpy().reshape(nP, nT, N_comp)

    colors = ["darkorange", "lawngreen", "darkcyan", "slateblue", "mediumorchid"]
    color_map = {comp: col for comp, col in zip(species, colors)}
    label_components = [
        r"O$_2$",
        r"CO$_2$",
        r"H$_2$O",
        r"C$_6$H$_6$",
        r"C$_4$H$_2$O$_3$",
    ]

    if P_target is not None:
        i = int(np.argmin(np.abs(P - P_target)))

        # ------------------ combined Y + VF plot versus T_flash ------------------
        fig, ax1 = plt.subplots(figsize=(8, 6), dpi=300)
        ax2 = ax1.twinx()

        # Y components on left axis
        for comp, color in zip(species, colors):
            k = species.index(comp)
            ax1.plot(
                T,
                Y_a[i, :, k],
                linestyle="--",
                linewidth=1.5,
                color=color,
            )
            ax1.plot(
                T,
                Y_p[i, :, k],
                linestyle="-",
                linewidth=1.5,
                color=color,
            )

        # VF on right axis
        right_axis_color = "navy"
        ax2.plot(
            T,
            VF_a[i, :],
            linestyle="--",
            linewidth=1.5,
            color=right_axis_color,
        )
        ax2.plot(
            T,
            VF_p[i, :],
            linestyle="-",
            linewidth=1.5,
            color=right_axis_color,
        )

        ax1.set_xlabel(
            "Flash Temperature (°C)", fontsize=fontsize, fontfamily=fontfamily
        )
        ax1.set_ylabel(
            "Vapor Component Molar Fraction ($Y_i$)",
            fontsize=fontsize,
            fontfamily=fontfamily,
        )
        ax1.tick_params(
            axis="x", which="major", labelsize=fontsize, labelfontfamily=fontfamily
        )
        ax1.tick_params(
            axis="y", which="major", labelsize=fontsize, labelfontfamily=fontfamily
        )

        ax2.set_ylabel(
            "Molar Vapor Fraction $(V/F)$",
            fontsize=fontsize,
            fontfamily=fontfamily,
        )
        ax2.tick_params(
            axis="y", which="major", labelsize=fontsize, labelfontfamily=fontfamily
        )
        ax2.yaxis.set_major_locator(MultipleLocator(0.1))
        ax2.spines["right"].set_color(right_axis_color)
        ax2.yaxis.label.set_color(right_axis_color)
        ax2.tick_params(axis="y", colors=right_axis_color)

        line_styles = [
            Line2D([0], [0], color="black", linestyle="--", linewidth=1, label="ASPEN"),
            Line2D([0], [0], color="black", linestyle="-", linewidth=1, label="PINN"),
        ]
        legend1 = ax1.legend(
            handles=line_styles,
            frameon=False,
            prop={"family": fontfamily, "size": fontsize - 2},
        )

        component_lines = [
            Line2D(
                [0], [0], color=color_map[comp], linestyle="-", linewidth=2, label=label
            )
            for comp, label in zip(species, label_components)
        ]
        legend2 = ax1.legend(
            handles=component_lines,
            loc="best",
            ncol=2,
            frameon=False,
            prop={"family": fontfamily, "size": fontsize - 2},
        )
        ax1.add_artist(legend1)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"Y_VF_vs_T_at_P_{P[i]:.4g}.png"))
        plt.close()

        # ------------------ X plot versus T_flash ------------------
        fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

        for k, sp in enumerate(species):
            ax.plot(T, X_a[i, :, k], "--", color=colors[k], linewidth=1.5)
            ax.plot(T, X_p[i, :, k], "-", color=colors[k], linewidth=1.5)

        ax.set_xlabel(
            "Flash Temperature (°C)", fontsize=fontsize, fontfamily=fontfamily
        )
        ax.set_ylabel(
            "Liquid Component Molar Fraction ($X_i$)",
            fontsize=fontsize,
            fontfamily=fontfamily,
        )
        ax.tick_params(
            axis="x", which="major", labelsize=fontsize, labelfontfamily=fontfamily
        )
        ax.tick_params(
            axis="y", which="major", labelsize=fontsize, labelfontfamily=fontfamily
        )

        # Legend 1: line styles
        line_styles = [
            Line2D(
                [0], [0], color="black", linestyle="--", linewidth=1.5, label="ASPEN"
            ),
            Line2D([0], [0], color="black", linestyle="-", linewidth=1.5, label="PINN"),
        ]
        legend1 = ax.legend(
            handles=line_styles,
            loc="upper left",
            frameon=False,
            prop={"family": fontfamily, "size": fontsize - 2},
        )

        # Legend 2: component colors
        label_components = [
            r"O$_2$",
            r"CO$_2$",
            r"H$_2$O",
            r"C$_6$H$_6$",
            r"C$_4$H$_2$O$_3$",
        ]
        component_lines = [
            Line2D(
                [0],
                [0],
                color=colors[k],
                linestyle="-",
                linewidth=2,
                label=label_components[k],
            )
            for k in range(len(species))
        ]
        legend2 = ax.legend(
            handles=component_lines,
            loc="best",
            ncol=2,
            frameon=False,
            prop={"family": fontfamily, "size": fontsize - 2},
        )

        ax.add_artist(legend1)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"X_vs_T_at_P_{P[i]:.4g}.png"))
        plt.close()

        # ------------------ Excel at fixed P ------------------
        excel_path = os.path.join(save_dir, f"profiles_at_P_{P[i]:.4g}.xlsx")
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            df_vf = pd.DataFrame(
                {
                    "T_flash": T,
                    "VF_aspen": VF_a[i, :],
                    "VF_pinn": VF_p[i, :],
                }
            )
            df_vf.to_excel(writer, sheet_name="VF", index=False)

            df_x = pd.DataFrame({"T_flash": T})
            for k, sp in enumerate(species):
                df_x[f"X_{sp}_aspen"] = X_a[i, :, k]
                df_x[f"X_{sp}_pinn"] = X_p[i, :, k]
            df_x.to_excel(writer, sheet_name="X", index=False)

            df_y = pd.DataFrame({"T_flash": T})
            for k, sp in enumerate(species):
                df_y[f"Y_{sp}_aspen"] = Y_a[i, :, k]
                df_y[f"Y_{sp}_pinn"] = Y_p[i, :, k]
            df_y.to_excel(writer, sheet_name="Y", index=False)

    if T_target is not None:
        j = int(np.argmin(np.abs(T - T_target)))

        # ------------------ combined Y + VF plot versus P_flash ------------------
        fig, ax1 = plt.subplots(figsize=(8, 6), dpi=300)
        ax2 = ax1.twinx()

        # Y components on left axis
        for comp, color in zip(species, colors):
            k = species.index(comp)
            ax1.plot(
                P,
                Y_a[:, j, k],
                linestyle="--",
                linewidth=1.5,
                color=color,
            )
            ax1.plot(
                P,
                Y_p[:, j, k],
                linestyle="-",
                linewidth=1.5,
                color=color,
            )

        # VF on right axis
        right_axis_color = "navy"
        ax2.plot(
            P,
            VF_a[:, j],
            linestyle="--",
            linewidth=1.5,
            color=right_axis_color,
        )
        ax2.plot(
            P,
            VF_p[:, j],
            linestyle="-",
            linewidth=1.5,
            color=right_axis_color,
        )

        ax1.set_xlabel("Flash Pressure (bar)", fontsize=fontsize, fontfamily=fontfamily)
        ax1.set_ylabel(
            "Vapor Component Molar Fraction ($Y_i$)",
            fontsize=fontsize,
            fontfamily=fontfamily,
        )
        ax1.tick_params(
            axis="x", which="major", labelsize=fontsize, labelfontfamily=fontfamily
        )
        ax1.tick_params(
            axis="y", which="major", labelsize=fontsize, labelfontfamily=fontfamily
        )

        ax2.set_ylabel(
            "Molar Vapor Fraction $(V/F)$",
            fontsize=fontsize,
            fontfamily=fontfamily,
        )
        ax2.tick_params(
            axis="y", which="major", labelsize=fontsize, labelfontfamily=fontfamily
        )
        ax2.yaxis.set_major_locator(MultipleLocator(0.1))
        ax2.spines["right"].set_color(right_axis_color)
        ax2.yaxis.label.set_color(right_axis_color)
        ax2.tick_params(axis="y", colors=right_axis_color)

        line_styles = [
            Line2D([0], [0], color="black", linestyle="--", linewidth=1, label="ASPEN"),
            Line2D([0], [0], color="black", linestyle="-", linewidth=1, label="PINN"),
        ]
        legend1 = ax1.legend(
            handles=line_styles,
            frameon=False,
            prop={"family": fontfamily, "size": fontsize - 2},
        )

        component_lines = [
            Line2D(
                [0], [0], color=color_map[comp], linestyle="-", linewidth=2, label=label
            )
            for comp, label in zip(species, label_components)
        ]
        legend2 = ax1.legend(
            handles=component_lines,
            loc="best",
            ncol=2,
            frameon=False,
            prop={"family": fontfamily, "size": fontsize - 2},
        )
        ax1.add_artist(legend1)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"Y_VF_vs_P_at_T_{T[j]:.4g}.png"))
        plt.close()

        # ------------------ X plot versus P_flash ------------------
        fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

        for k, sp in enumerate(species):
            ax.plot(P, X_a[:, j, k], "--", color=colors[k], linewidth=1.5)
            ax.plot(P, X_p[:, j, k], "-", color=colors[k], linewidth=1.5)

        ax.set_xlabel("Flash Pressure (bar)", fontsize=fontsize, fontfamily=fontfamily)
        ax.set_ylabel(
            "Liquid Component Molar Fraction ($X_i$)",
            fontsize=fontsize,
            fontfamily=fontfamily,
        )
        ax.tick_params(
            axis="x", which="major", labelsize=fontsize, labelfontfamily=fontfamily
        )
        ax.tick_params(
            axis="y", which="major", labelsize=fontsize, labelfontfamily=fontfamily
        )

        # Legend 1: line styles
        line_styles = [
            Line2D(
                [0], [0], color="black", linestyle="--", linewidth=1.5, label="ASPEN"
            ),
            Line2D([0], [0], color="black", linestyle="-", linewidth=1.5, label="PINN"),
        ]
        legend1 = ax.legend(
            handles=line_styles,
            loc="upper left",
            frameon=False,
            prop={"family": fontfamily, "size": fontsize - 2},
        )

        # Legend 2: component colors
        label_components = [
            r"O$_2$",
            r"CO$_2$",
            r"H$_2$O",
            r"C$_6$H$_6$",
            r"C$_4$H$_2$O$_3$",
        ]
        component_lines = [
            Line2D(
                [0],
                [0],
                color=colors[k],
                linestyle="-",
                linewidth=2,
                label=label_components[k],
            )
            for k in range(len(species))
        ]
        legend2 = ax.legend(
            handles=component_lines,
            loc="best",
            ncol=2,
            frameon=False,
            prop={"family": fontfamily, "size": fontsize - 2},
        )

        ax.add_artist(legend1)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"X_vs_P_at_T_{T[j]:.4g}.png"))
        plt.close()

        # ------------------ Excel at fixed T ------------------
        excel_path = os.path.join(save_dir, f"profiles_at_T_{T[j]:.4g}.xlsx")
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            df_vf = pd.DataFrame(
                {
                    "P_flash": P,
                    "VF_aspen": VF_a[:, j],
                    "VF_pinn": VF_p[:, j],
                }
            )
            df_vf.to_excel(writer, sheet_name="VF", index=False)

            df_x = pd.DataFrame({"P_flash": P})
            for k, sp in enumerate(species):
                df_x[f"X_{sp}_aspen"] = X_a[:, j, k]
                df_x[f"X_{sp}_pinn"] = X_p[:, j, k]
            df_x.to_excel(writer, sheet_name="X", index=False)

            df_y = pd.DataFrame({"P_flash": P})
            for k, sp in enumerate(species):
                df_y[f"Y_{sp}_aspen"] = Y_a[:, j, k]
                df_y[f"Y_{sp}_pinn"] = Y_p[:, j, k]
            df_y.to_excel(writer, sheet_name="Y", index=False)


if __name__ == "__main__":
    torch.manual_seed(1234)

    species = ["O2", "CO2", "H2O", "C6H6", "C4H2O3"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    I_S_data = pd.read_excel("I_S_data.xlsx")
    D_S_data = pd.read_excel("D_S_data.xlsx")

    _, I_S_metrics = Normalization.min_max(
        I_S_data, normalize_cols=["T_flash", "P_flash"]
    )
    _, D_S_metrics = Normalization.min_max(D_S_data, normalize_cols=[])

    input_cols = list(I_S_data.columns)
    output_cols = list(D_S_data.columns)

    dim_in, dim_out = I_S_data.shape[1], D_S_data.shape[1]

    depth, width = 6, 64

    model = Analyze.load_ann_pinn(
        dim_in=dim_in,
        dim_out=dim_out,
        depth=depth,
        width=width,
        activation=nn.Tanh,
        ckpt_path="./logs/best_model.pth",
        device=device,
    )

    # -------- Case-specific process specifications --------
    T_range, nT = [60.0, 90.0], 61  # deg C
    P_range, nP = [1.0, 2.0], 21  # bar
    Z = [0.2, 0.3, 0.1, 0.1, 0.3]  # molar fractions

    # Fixed values needed by D_S_samples(I_S)
    # Note: T_feed \in [300, 400] C; P_feed \in [1, 2] bar; and F \in [500, 7500]
    F = 3500.0  # kmol/h
    T_feed = 400.0  # deg C
    P_feed = 1.0  # bar
    P_target = 1.5  # bar
    T_target = 75.0  # deg C
    # ------------------------------------------------------

    # Full physical-space input grid
    I_S_grid_full, T_grid, P_grid = build_grid_inputs(
        T_range=T_range,
        P_range=P_range,
        nT=nT,
        nP=nP,
        Z=Z,
        F=F,
        T_feed=T_feed,
        P_feed=P_feed,
        species=species,
    )

    # Aspen evaluation using existing data_gen method
    D_S_aspen_full = D_S_samples(I_S_grid_full)
    # Keep only the surrogate outputs and align inputs using returned case_id
    D_S_aspen = D_S_aspen_full[["case_id", *output_cols]].reset_index(drop=True)
    I_S_model = (
        I_S_grid_full.loc[D_S_aspen["case_id"], input_cols]
        .reset_index(drop=True)
        .copy()
    )

    # PINN evaluation on the Aspen cases
    D_S_pinn = Analyze.pinn_eval(
        model=model,
        I_S_model=I_S_model,
        I_S_metrics=I_S_metrics,
        D_S_metrics=D_S_metrics,
        output_cols=output_cols,
        device=device,
    )

    # Error Metrics
    groups = {
        "VF": ["VF"],
        "Y": [f"Y_{sp}" for sp in species],
        "X": [f"X_{sp}" for sp in species],
    }
    metrics_df = Analyze.error_metrics(
        df_true=D_S_aspen[output_cols], df_pred=D_S_pinn, groups=groups
    )

    # Aspen ground truth sheet
    aspen_sheet = pd.concat(
        [
            I_S_model.reset_index(drop=True),
            D_S_aspen[output_cols].reset_index(drop=True),
        ],
        axis=1,
    )

    # PINN predictions sheet
    pinn_sheet = pd.concat(
        [
            I_S_model.reset_index(drop=True),
            D_S_pinn.reset_index(drop=True),
        ],
        axis=1,
    )

    # Save Aspen ground truth, PINN predictions, and error metrics into .xlsx
    save_dir = "compare"
    os.makedirs(save_dir, exist_ok=True)
    with pd.ExcelWriter(os.path.join(save_dir, "comparison.xlsx")) as writer:
        aspen_sheet.to_excel(writer, sheet_name="Aspen", index=False)
        pinn_sheet.to_excel(writer, sheet_name="PINN", index=False)
        metrics_df.to_excel(writer, sheet_name="Errors", index=False)

    # Plot and save profile comparisons across PINN and Aspen
    plot_profiles(
        T_grid=T_grid,
        P_grid=P_grid,
        D_S_aspen=D_S_aspen[output_cols],
        D_S_pinn=D_S_pinn,
        species=species,
        save_dir=save_dir,
        P_target=1.5,
        T_target=75.0,
    )
