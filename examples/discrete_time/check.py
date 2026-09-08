import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from sklearn.model_selection import train_test_split

import cstr_model as cstr
from main import SCHEME
from phys_res import Physics
from pinnse import Normalization, Denormalization, Analyze

"""
Evaluation for the discrete-time CSTR model.

Reports one-step accuracy on the held-out test partition, then a multi-step
rollout in which the model is applied recursively to its own predictions while
the coolant temperature follows a prescribed schedule, compared against the
governing ODEs.
"""

# Rollout specification.
# The horizon spans one residence time (V/q = 1 min) and the coolant steps mid-way.
N_STEPS = 2000
CA_0, T_0 = 0.8, 340.0
TC_BEFORE, TC_AFTER = 300.0, 302.0  # step change in the manipulated variable
STEP_AT = N_STEPS // 2


def test_split_indices(
    n_samples: int, test_frac: float = 0.1, val_frac: float = 0.1, seed: int = 42
):
    """Reproduce the held-out test partition used by DataModule."""
    idx = np.arange(n_samples)
    idx_tv, idx_test = train_test_split(idx, test_size=test_frac, random_state=seed)
    _, _ = train_test_split(idx_tv, test_size=val_frac, random_state=seed)
    return idx_test


def tc_schedule(step: int):
    """Piecewise-constant coolant temperature applied over the rollout."""
    return TC_BEFORE if step < STEP_AT else TC_AFTER


def model_rollout(model, I_S_metrics, D_S_metrics, device):
    """Apply the model recursively to its own predictions over the horizon."""
    CA, T = CA_0, T_0
    traj = [(0.0, CA, T, tc_schedule(0))]

    for step in range(N_STEPS):
        TC = tc_schedule(step)
        row = pd.DataFrame([{"CA_k": CA, "T_k": T, "TC_k": TC}])
        row_norm = Normalization.min_max_defined_metrics(row, I_S_metrics)
        x = torch.tensor(row_norm.to_numpy(), dtype=torch.float32, device=device)

        with torch.no_grad():
            y = model(x)

        CA = float(Denormalization.min_max_col(y[0, 0].item(), "CA_k1", D_S_metrics))
        T = float(Denormalization.min_max_col(y[0, 1].item(), "T_k1", D_S_metrics))
        traj.append(((step + 1) * cstr.deltaT, CA, T, TC))

    return pd.DataFrame(traj, columns=["t", "CA", "T", "TC"])


def truth_rollout():
    """Integrate the governing ODEs under the same zero-order-hold schedule."""

    def rhs(t, y, TC):
        dCA, dT = cstr.derivatives(y[0], y[1], TC)
        return [dCA, dT]

    CA, T = CA_0, T_0
    traj = [(0.0, CA, T, tc_schedule(0))]

    for step in range(N_STEPS):
        TC = tc_schedule(step)
        sol = solve_ivp(
            rhs,
            t_span=(0.0, cstr.deltaT),
            y0=[CA, T],
            args=(TC,),
            method="RK45",
            rtol=1e-10,
            atol=1e-12,
        )
        CA, T = sol.y[0, -1], sol.y[1, -1]
        traj.append(((step + 1) * cstr.deltaT, CA, T, TC))

    return pd.DataFrame(traj, columns=["t", "CA", "T", "TC"])


def plot_rollout(truth, pred, save_dir, fontsize=19, fontfamily="Arial"):
    """Plot the model and reference trajectories over the rollout horizon."""
    fig, axes = plt.subplots(2, 1, figsize=(8, 9), dpi=300, sharex=True)

    for ax, col, label, color in zip(
        axes,
        ["CA", "T"],
        ["Concentration of A (mol/L)", "Reactor Temperature (K)"],
        ["slateblue", "indianred"],
    ):
        ax.plot(truth["t"], truth[col], "--", lw=1.8, color="black", label="Truth")
        ax.plot(pred["t"], pred[col], "-", lw=1.8, color=color, label="PINN rollout")
        ax.axvline(STEP_AT * cstr.deltaT, color="gray", ls=":", lw=1.2)
        ax.set_ylabel(label, fontsize=fontsize, fontfamily=fontfamily)
        ax.legend(frameon=False, prop={"family": fontfamily, "size": fontsize - 3})
        for axis in ("x", "y"):
            ax.tick_params(
                axis=axis, which="major", labelsize=fontsize, labelfontfamily=fontfamily
            )

    axes[-1].set_xlabel("Time (min)", fontsize=fontsize, fontfamily=fontfamily)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "rollout.png"), bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    torch.manual_seed(1234)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_dir = "compare"
    os.makedirs(save_dir, exist_ok=True)

    I_S_data = pd.read_excel("I_S_data.xlsx")
    D_S_data = pd.read_excel("D_S_data.xlsx")

    # Recomputed exactly as in main.py, so evaluation uses the same scaling
    _, I_S_metrics = Normalization.min_max(I_S_data)
    _, D_S_metrics = Normalization.min_max(D_S_data)

    model = Analyze.load_ann_pinn(
        dim_in=I_S_data.shape[1],
        dim_out=D_S_data.shape[1],
        depth=4,
        width=64,
        activation=nn.Tanh,
        ckpt_path="./logs/best_model.pth",
        device=device,
    )
    physics = Physics(I_S_metrics, D_S_metrics, scheme=SCHEME)

    # ---------------- 1. One-step accuracy on the held-out test set ----------------
    idx_test = test_split_indices(len(I_S_data), test_frac=0.1, val_frac=0.1, seed=42)
    I_S_test = I_S_data.iloc[idx_test].reset_index(drop=True)
    D_S_test = D_S_data.iloc[idx_test].reset_index(drop=True)

    I_S_test_norm = Normalization.min_max_defined_metrics(I_S_test, I_S_metrics)
    x_test = torch.tensor(
        I_S_test_norm.to_numpy(), dtype=torch.float32, device=device
    ).requires_grad_(True)
    y_test = model(x_test)

    pred = pd.DataFrame(
        {
            "CA_k1": Denormalization.min_max_col(
                y_test[:, 0].detach().cpu().numpy(), "CA_k1", D_S_metrics
            ),
            "T_k1": Denormalization.min_max_col(
                y_test[:, 1].detach().cpu().numpy(), "T_k1", D_S_metrics
            ),
        }
    )
    groups = {"CA_k1": ["CA_k1"], "T_k1": ["T_k1"], "Overall": ["CA_k1", "T_k1"]}
    step_metrics = Analyze.error_metrics(D_S_test, pred, groups)

    res = physics(x_test, y_test).detach().cpu().numpy()
    res_l2 = np.linalg.norm(res, axis=1)

    print(f"One-step accuracy on held-out test set ({len(idx_test)} samples):")
    print(step_metrics.to_string(index=False))
    print(f"  mean physics residual : {res_l2.mean():.4e}")

    # ---------------- 2. Multi-step rollout ----------------
    truth = truth_rollout()
    pred_traj = model_rollout(model, I_S_metrics, D_S_metrics, device)
    plot_rollout(truth, pred_traj, save_dir)

    err_CA = np.abs(truth["CA"].to_numpy() - pred_traj["CA"].to_numpy())
    err_T = np.abs(truth["T"].to_numpy() - pred_traj["T"].to_numpy())

    print(f"\nMulti-step rollout ({N_STEPS} steps, {N_STEPS * cstr.deltaT:.3g} min):")
    print(f"  CA : mean |err| {err_CA.mean():.4e}, final |err| {err_CA[-1]:.4e} mol/L")
    print(f"  T  : mean |err| {err_T.mean():.4e}, final |err| {err_T[-1]:.4e} K")

    # Warn if the rollout leaves the domain the model was trained on
    for name, series, key in (
        ("CA", pred_traj["CA"], "CA_k"),
        ("T", pred_traj["T"], "T_k"),
    ):
        lo, hi = cstr.bounds[key]
        if series.min() < lo or series.max() > hi:
            print(
                f"  NOTE: {name} rollout leaves the training domain "
                f"[{lo}, {hi}] (reached [{series.min():.4g}, {series.max():.4g}]); "
                f"predictions there are extrapolation."
            )

    # ---------------- 3. Persist ----------------
    with pd.ExcelWriter(os.path.join(save_dir, "compare.xlsx")) as writer:
        step_metrics.to_excel(writer, sheet_name="One_Step_Metrics", index=False)
        truth.to_excel(writer, sheet_name="Rollout_Truth", index=False)
        pred_traj.to_excel(writer, sheet_name="Rollout_Model", index=False)
        pd.DataFrame(
            {"t": truth["t"], "abs_err_CA": err_CA, "abs_err_T": err_T}
        ).to_excel(writer, sheet_name="Rollout_Error", index=False)

    print(f"\nSaved results to {save_dir}")
