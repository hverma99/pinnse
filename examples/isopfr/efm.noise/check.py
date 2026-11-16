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

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
from pathlib import Path

import torch  # noqa: F401  # load torch before NumPy (see pyproject.toml)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from pinnse import Analyze

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # examples/isopfr

from common.evaluate import (
    SAVE_DIR,
    STYLES,
    build_profile_inputs,
    load_model,
    metric_groups,
    phys_residual_eval,
    pinn_eval,
    plot_profile,
    save_results,
    truth_eval,
)
from common.process import E, k, key_species, nu, species
from common.train import load_data, normalize
from main import NOISE_LEVEL, NOISE_MODE, NOISE_SEED, add_noise
from phys_res import Physics


def test_split_indices(n_samples: int):
    """
    Reproduce the held-out test partition used by DataModule.
    """
    idx = np.arange(n_samples)
    idx_tv, idx_test = train_test_split(idx, test_size=0.1, random_state=42)
    _, _ = train_test_split(idx_tv, test_size=0.1, random_state=42)
    return idx_test


def overall_mae(metrics_df: pd.DataFrame) -> float:
    return float(
        metrics_df.set_index("Output")
        .loc[["Overall"], "MAE"]
        .to_numpy(dtype=np.float64)
        .item()
    )


if __name__ == "__main__":
    I_S_data, D_S_data = load_data()

    # Regenerate the same noisy dataset used for training, so that the
    # normalization metrics below match those the model was trained with.
    D_S_noisy = add_noise(D_S_data)

    _, _, I_S_metrics, D_S_metrics = normalize("EFM", I_S_data, D_S_noisy)

    model = load_model(I_S_data, D_S_data)
    physics = Physics(I_S_metrics, D_S_metrics, species, nu, key_species, k, E)
    groups = metric_groups(D_S_data.columns)

    # ---------- 1. Aggregate errors on the held-out test set vs CLEAN labels ----------
    idx_test = test_split_indices(len(I_S_data))
    I_S_test = I_S_data.iloc[idx_test].reset_index(drop=True)
    D_S_test_clean = D_S_data.iloc[idx_test].reset_index(drop=True)

    D_S_test_pred = pinn_eval(
        model, I_S_test, I_S_metrics, D_S_metrics, D_S_data.columns
    )

    os.makedirs(SAVE_DIR, exist_ok=True)

    test_metrics_df = Analyze.error_metrics(D_S_test_clean, D_S_test_pred, groups)
    res_abs_test, res_l2_test = phys_residual_eval(
        I_S_test, model, physics, I_S_metrics
    )

    print(f"\nHeld-out test set ({len(idx_test)} samples), against clean labels:")
    print(test_metrics_df.to_string(index=False))
    print(f"  mean physics residual : {res_l2_test.mean():.4e}")

    # ---------- 2. Profile comparison against the first-principle solution ----------
    I_S_prof, V_eval = build_profile_inputs()
    D_S_truth = truth_eval("EFM", I_S_prof, V_eval)
    D_S_pinn = pinn_eval(model, I_S_prof, I_S_metrics, D_S_metrics, D_S_data.columns)

    prof_metrics_df = Analyze.error_metrics(D_S_truth, D_S_pinn, groups)
    res_abs_prof, res_l2_prof = phys_residual_eval(
        I_S_prof, model, physics, I_S_metrics
    )
    plot_profile(V_eval, D_S_truth, D_S_pinn, STYLES["EFM"], model_label="Model")

    print("\nReactor profile, against the first-principle solution:")
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
                "test_MAE_overall": overall_mae(test_metrics_df),
                "test_phys_residual": float(res_l2_test.mean()),
                "profile_MAE_overall": overall_mae(prof_metrics_df),
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
        style=STYLES["EFM"],
        extra_sheets={"Test_Error_Metrics": test_metrics_df, "Run_Summary": summary_df},
    )
    summary_df.to_csv(os.path.join(SAVE_DIR, "summary.csv"), index=False)
