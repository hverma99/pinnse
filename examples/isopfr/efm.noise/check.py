import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.model_selection import train_test_split

from data_gen import species, nu, key_species, k, E
from pfr_model import EFM
from phys_res import Physics
from pinnse import Normalization, Analyze, Noise
from main import NOISE_LEVEL, NOISE_MODE, NOISE_SEED

"""
Evaluation for the noisy-data EFM study.

Two evaluations are reported. First, the trained model is compared against the
first-principle solution along a reactor profile, which is noise free by
construction. Second, aggregate predictive errors are computed on the held-out
test partition against the CLEAN labels rather than the noisy labels the model
was trained on, which is what measures whether the model recovered the
underlying process or fitted the measurement noise.

The physics residual is evaluated in both cases, including for the data-only
baseline, where it quantifies the extent to which an unconstrained model
violates the governing material balances.
"""


def test_split_indices(n_samples: int):
    """
    Reproduce the held-out test partition used by DataModule.
    """
    idx = np.arange(n_samples)
    idx_tv, idx_test = train_test_split(idx, test_size=0.1, random_state=42)
    _, _ = train_test_split(idx_tv, test_size=0.1, random_state=42)
    return idx_test


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
    I_S: pd.DataFrame,
    species: list[str],
    nu: np.ndarray,
    key_species: list[str],
    k: np.ndarray,
    E: np.ndarray,
):
    """
    Evaluate the first-principles EFM model along the reactor profile.
    """
    F_cols = [f"F_in_{sp}" for sp in species]
    F_in_all = I_S[F_cols].to_numpy(dtype=np.float32)
    P_all = I_S["P"].to_numpy(dtype=np.float32)
    T_all = I_S["T"].to_numpy(dtype=np.float32)
    V_all = I_S["V"].to_numpy(dtype=np.float32)

    Y = np.zeros((len(I_S), len(species)), dtype=np.float32)

    for i in range(len(I_S)):
        F_in = F_in_all[i]
        P = float(P_all[i])
        T = float(T_all[i])
        V = float(V_all[i])

        model = EFM(species, nu, key_species, k, E, F_in, P, T)
        Y[i, :] = model.solve(V)

    return pd.DataFrame(Y, columns=[f"F_ot_{sp}" for sp in species])


def phys_residual_eval(I_S, model, physics, I_S_metrics, device):
    """Evaluate the physics residual of a trained model at given inputs."""
    I_S_norm = Normalization.scale_centered_defined_metrics(I_S, I_S_metrics)
    x = torch.tensor(
        I_S_norm.to_numpy(), dtype=torch.float32, device=device
    ).requires_grad_(True)
    y = model(x)
    res = physics(x, y).detach().cpu().numpy()
    return np.abs(res), np.linalg.norm(res, axis=1)


def plot_profile(
    V_eval, D_S_truth, D_S_pinn, save_dir, fontsize=19, fontfamily="Arial"
):
    """Plot first-principle vs model effluent flow profiles."""
    components = ["CO2", "H2O", "C6H6", "C4H2O3"]
    labels = [r"CO$_2$", r"H$_2$O", r"C$_6$H$_6$", r"C$_4$H$_2$O$_3$"]
    colors = ["lawngreen", "darkcyan", "slateblue", "mediumorchid"]

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    for comp, color in zip(components, colors):
        ax.plot(V_eval, D_S_truth[f"F_ot_{comp}"], "--", lw=1.5, color=color)
        ax.plot(V_eval, D_S_pinn[f"F_ot_{comp}"], "-", lw=1.5, color=color)

    style_legend = ax.legend(
        handles=[
            Line2D([0], [0], color="black", ls="--", lw=1.0, label="Truth"),
            Line2D([0], [0], color="black", ls="-", lw=1.0, label="Model"),
        ],
        loc="center right",
        frameon=False,
        prop={"family": fontfamily, "size": fontsize - 2},
    )
    ax.legend(
        handles=[
            Line2D([0], [0], color=c, ls="-", lw=1.5, label=l)
            for c, l in zip(colors, labels)
        ],
        loc="upper left",
        frameon=False,
        prop={"family": fontfamily, "size": fontsize - 2},
    )
    ax.add_artist(style_legend)

    ax.set_xlabel(
        r"Reactor Volume ($\mathrm{m}^3$)", fontsize=fontsize, fontfamily=fontfamily
    )
    ax.set_ylabel(
        "Effluent Molar Flowrate (mol/s)", fontsize=fontsize, fontfamily=fontfamily
    )
    for axis in ("x", "y"):
        ax.tick_params(
            axis=axis, which="major", labelsize=fontsize, labelfontfamily=fontfamily
        )

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "effluent_flow.png"), bbox_inches="tight")
    plt.close()


def save_results(
    I_S: pd.DataFrame,
    D_S_truth: pd.DataFrame,
    D_S_pinn: pd.DataFrame,
    mae: np.ndarray,
    res_abs: np.ndarray,
    res_l2: np.ndarray,
    metrics_df: pd.DataFrame,
    test_metrics_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    species: list[str],
    save_dir: str,
):
    """
    Save profile results to Excel.

    Mirrors the EFM case, with two additional sheets: the aggregate errors on
    the held-out test set against the clean labels, and a one-row summary of
    this instance. The summary rows of the individual instances concatenate
    directly into the comparison across noise levels.
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
    for j, sp in enumerate(species):
        mae_sheet[f"MAE_{sp}"] = mae[:, j]
        res_sheet[f"Residual_{sp}"] = res_abs[:, j]

    res_overall_sheet = pd.DataFrame({"V": V_eval, "Residual_Overall": res_l2})

    with pd.ExcelWriter(os.path.join(save_dir, "compare.xlsx")) as writer:
        truth_sheet.to_excel(writer, sheet_name="Truth", index=False)
        pinn_sheet.to_excel(writer, sheet_name="PINN", index=False)
        mae_sheet.to_excel(writer, sheet_name="MAE", index=False)
        res_sheet.to_excel(writer, sheet_name="Physics_Residual", index=False)
        res_overall_sheet.to_excel(writer, sheet_name="Residual_Overall", index=False)
        metrics_df.to_excel(writer, sheet_name="Error_Metrics", index=False)
        test_metrics_df.to_excel(writer, sheet_name="Test_Error_Metrics", index=False)
        summary_df.to_excel(writer, sheet_name="Run_Summary", index=False)

    summary_df.to_csv(os.path.join(save_dir, "summary.csv"), index=False)


if __name__ == "__main__":
    torch.manual_seed(1234)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    I_S_data = pd.read_excel("I_S_data.xlsx")
    D_S_data = pd.read_excel("D_S_data.xlsx")

    # Regenerate the same noisy dataset used for training, so that the
    # normalization metrics below match those the model was trained with.
    D_S_noisy, noise_metrics = Noise.gaussian(
        data=D_S_data,
        noise_level=NOISE_LEVEL,
        mode=NOISE_MODE,
        clip_min=0.0,
        random_state=NOISE_SEED,
    )

    _, _, I_S_metrics, D_S_metrics = Normalization.min_max_pfr(
        I_S_data=I_S_data, D_S_data=D_S_noisy, formulation="EFM", species=species
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

    # ---------- 1. Aggregate errors on the held-out test set vs CLEAN labels ----------
    idx_test = test_split_indices(len(I_S_data))
    I_S_test = I_S_data.iloc[idx_test].reset_index(drop=True)
    D_S_test_clean = D_S_data.iloc[idx_test].reset_index(drop=True)

    D_S_test_pred = Analyze.pinn_eval_pfr(
        model=model,
        I_S_model=I_S_test,
        I_S_metrics=I_S_metrics,
        D_S_metrics=D_S_metrics,
        output_cols=list(D_S_data.columns),
        device=device,
    )

    save_dir = "compare"
    os.makedirs(save_dir, exist_ok=True)

    groups = {f"F_ot_{sp}": [f"F_ot_{sp}"] for sp in species}
    groups["Overall"] = list(D_S_data.columns)
    test_metrics_df = Analyze.error_metrics(D_S_test_clean, D_S_test_pred, groups)

    res_abs_test, res_l2_test = phys_residual_eval(
        I_S_test, model, physics, I_S_metrics, device
    )

    print(f"\nHeld-out test set ({len(idx_test)} samples), against clean labels:")
    print(test_metrics_df.to_string(index=False))
    print(f"  mean physics residual : {res_l2_test.mean():.4e}")

    # ---------- 2. Profile comparison against the first-principle solution ----------
    F_in = np.array([1000.0, 40.0, 25.0, 80.0, 20.0], dtype=np.float32)
    P, T, V, N_points = 1.75e5, 425.0 + 273.15, 250.0, 100

    I_S_prof, V_eval = build_profile_inputs(F_in, P, T, V, N_points, species)
    D_S_truth = truth_eval(I_S_prof, species, nu, key_species, k, E)

    D_S_pinn = Analyze.pinn_eval_pfr(
        model=model,
        I_S_model=I_S_prof,
        I_S_metrics=I_S_metrics,
        D_S_metrics=D_S_metrics,
        output_cols=list(D_S_data.columns),
        device=device,
    )

    prof_metrics_df = Analyze.error_metrics(D_S_truth, D_S_pinn, groups)
    res_abs_prof, res_l2_prof = phys_residual_eval(
        I_S_prof, model, physics, I_S_metrics, device
    )
    plot_profile(V_eval, D_S_truth, D_S_pinn, save_dir)

    print(f"\nReactor profile, against the first-principle solution:")
    print(prof_metrics_df.to_string(index=False))
    print(f"  mean physics residual : {res_l2_prof.mean():.4e}")

    mae = np.abs(D_S_truth.to_numpy() - D_S_pinn.to_numpy())

    summary_df = pd.DataFrame(
        [
            {
                # Instance folder name, so that the summary rows of the
                # individual runs remain distinguishable once concatenated.
                "run": os.path.basename(os.getcwd()),
                "noise_mode": NOISE_MODE,
                "noise_level": NOISE_LEVEL,
                "noise_seed": NOISE_SEED,
                "test_MAE_overall": float(
                    test_metrics_df.set_index("Output")
                    .loc[["Overall"], "MAE"]
                    .to_numpy(dtype=np.float64)
                    .item()
                ),
                "test_phys_residual": float(res_l2_test.mean()),
                "profile_MAE_overall": float(
                    prof_metrics_df.set_index("Output")
                    .loc[["Overall"], "MAE"]
                    .to_numpy(dtype=np.float64)
                    .item()
                ),
                "profile_phys_residual": float(res_l2_prof.mean()),
            }
        ]
    )

    save_results(
        I_S=I_S_prof,
        D_S_truth=D_S_truth,
        D_S_pinn=D_S_pinn,
        mae=mae,
        res_abs=res_abs_prof,
        res_l2=res_l2_prof,
        metrics_df=prof_metrics_df,
        test_metrics_df=test_metrics_df,
        summary_df=summary_df,
        species=species,
        save_dir=save_dir,
    )
