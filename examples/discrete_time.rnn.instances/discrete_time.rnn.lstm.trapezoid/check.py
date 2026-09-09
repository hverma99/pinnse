import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch, torch.nn as nn, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from pinnse import ANN, Training, Normalization, Denormalization, Analyze

import cstr_model as cstr
import main as M
from phys_res import Physics

"""
Evaluation and cross-model comparison for the recurrent CSTR model.

Four models predict the same held-out trajectories, over the same horizon,
from the same initial condition and coolant schedule, and are scored against
the same reference:

    bound           the best any function of the instantaneous input can do,
                    obtained analytically by predicting each group of identical
                    inputs with the mean of its own labels. No training.
    per-step        a feedforward network on the recurrent model's inputs,
                    applied independently at each step.
    autoregressive  a feedforward model of a single transition, which is given
                    the current state, applied repeatedly to its own predictions.
    recurrent       the model of main.py, emitting the trajectory in one pass.
"""

# ----------------------------- Run configuration -----------------------------
N_SHOW = 3  # example trajectories drawn in the comparison figure
PS_DEPTH, PS_WIDTH = 4, 64  # per-step feedforward baseline
AR_DEPTH, AR_WIDTH = 4, 75  # autoregressive baseline

AR_I_S_keys = ["CA_k", "T_k", "TC_k"]
AR_D_S_keys = cstr.D_S_keys


# ============================== shared helpers ===============================
def load_recurrent(device):
    """Rebuild the architecture used in training and load the best checkpoint."""
    model, physics, _, _, I_S_metrics, D_S_metrics = M.build(device)
    ckpt = torch.load("./logs/best_model.pth", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, physics


def predict_sequences(model, norm_I_S_test, device, batch=64):
    """Predict trajectories in batches, returning the normalized output."""
    out = []
    with torch.no_grad():
        for i in range(0, len(norm_I_S_test), batch):
            x = torch.tensor(
                norm_I_S_test[i : i + batch], dtype=torch.float32, device=device
            )
            out.append(model(x).cpu().numpy())
    return np.concatenate(out, axis=0)


def denorm_outputs(norm_D_S_data, D_S_metrics, keys=None):
    """Denormalize a normalized output array to dimensional variables."""
    keys = keys if keys is not None else cstr.D_S_keys
    out = np.empty_like(norm_D_S_data)
    for j, key in enumerate(keys):
        out[..., j] = Denormalization.min_max_col(
            norm_D_S_data[..., j], key, D_S_metrics
        )
    return out


def fit(model, loaders, coll_loader, physics, ckpt_path, device):
    """Train a baseline under the settings of `main.py` and load its best state."""
    train_loader, val_loader, test_loader = loaders
    Training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=torch.optim.Adam(model.parameters(), lr=M.LR),
        loss_fn=nn.MSELoss(),
        device=device,
        phys_coll_loader=coll_loader,
        phys_residual=physics,
        ckpt_path=ckpt_path,
        phys_weight=M.PHYS_WEIGHT,
        bnd_weight=0.0,
        adapt_wts=False,
    ).adam_step(epochs=M.epochs, val_every=50, verbose=True)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


# ======================= 1. bound: any memoryless model ======================
def bound_predictions(I_S_data, D_S_data, idx):
    """
    The best prediction that is a function of the instantaneous input.

    Steps are grouped by identical input vectors and each group is predicted by
    the mean of its own labels, which no function of the input alone can beat.
    Evaluated in sample on the given partition, so the bound is conservative.
    """
    I_S_p, D_S_p = I_S_data[idx], D_S_data[idx]
    flat_x = I_S_p.reshape(-1, I_S_p.shape[-1])
    flat_y = D_S_p.reshape(-1, D_S_p.shape[-1])

    # Rounding guards against float noise in the repeated context features
    _, group = np.unique(np.round(flat_x, 9), axis=0, return_inverse=True)

    pred = np.empty_like(flat_y)
    for g in range(group.max() + 1):
        m = group == g
        pred[m] = flat_y[m].mean(axis=0)

    return pred.reshape(D_S_p.shape), group.max() + 1


# ===================== 2. per-step feedforward baseline =====================
class PerStepANN(nn.Module):
    """
    Wrap a feedforward network so that it maps sequences to sequences by acting
    independently at each time step, making it a drop-in replacement for the
    recurrent model so every other component of the run is held fixed.
    """

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x):
        if x.dim() != 3:
            raise ValueError(
                f"PerStepANN expects (batch, seq_len, in_dim); got {tuple(x.shape)}."
            )
        b, t, d = x.shape
        return self.net(x.reshape(b * t, d)).reshape(b, t, -1)


def perstep_predictions(data, norm_I_S_data, idx_test, metrics, device):
    """Train the per-step feedforward baseline and predict the test trajectories."""
    I_S_metrics, D_S_metrics = metrics
    layers = [len(cstr.I_S_keys)] + [PS_WIDTH] * PS_DEPTH + [len(cstr.D_S_keys)]
    model = PerStepANN(ANN(layers, activation=nn.Tanh)).to(device)

    print(
        f"\nTraining the per-step feedforward baseline "
        f"({sum(p.numel() for p in model.parameters()):,} parameters)"
    )
    model = fit(
        model,
        data.labeled_data_loader(),
        data.phys_colloc_loader(),
        Physics(I_S_metrics, D_S_metrics, scheme=M.SCHEME),
        "./logs/perstep_ann.pth",
        device,
    )

    norm_pred = predict_sequences(model, norm_I_S_data[idx_test], device)
    return denorm_outputs(norm_pred, D_S_metrics)


# =================== 3. autoregressive one-step baseline ====================
def extract_transitions(I_S_data, D_S_data, idx):
    """
    Recover the individual state transitions contained in a set of trajectories.

    The current state at step k is the previous label, with the initial
    condition supplying step 0, so a trajectory of length T yields T transitions.
    """
    I_S_p, D_S_p = I_S_data[idx], D_S_data[idx]
    u_curr = np.concatenate([I_S_p[:, :1, 0:2], D_S_p[:, :-1, :]], axis=1)
    inputs = np.concatenate([u_curr, I_S_p[:, :, 2:3]], axis=2).reshape(-1, 3)
    return inputs, D_S_p.reshape(-1, 2)


class OneStepPhysics:
    """
    Single-transition discrete-time residual, the counterpart of the unrolled
    residual in `phys_res.py`. The current state is an input here, so the
    residual relates one input to one output and does not chain predictions.
    """

    def __init__(self, I_metrics, D_metrics, dt=cstr.deltaT, scheme=M.SCHEME):
        self.I_metrics, self.D_metrics = I_metrics, D_metrics
        self.dt, self.scheme = dt, scheme
        self.ot_ranges = [
            D_metrics[k]["max"] - D_metrics[k]["min"] for k in AR_D_S_keys
        ]

    def _f(self, CA, T, TC):
        rate = cstr.k0 * torch.exp(-cstr.E_over_R / T) * CA
        dCA = (cstr.q / cstr.V) * (cstr.CA_f - CA) - rate
        dT = (
            (cstr.q / cstr.V) * (cstr.T_f - T)
            + ((-cstr.dH) / (cstr.rho * cstr.Cp)) * rate
            + (cstr.UA / (cstr.V * cstr.rho * cstr.Cp)) * (TC - T)
        )
        return dCA, dT

    def _increment(self, CA_k, T_k, CA_k1, T_k1, TC):
        if self.scheme == "euler":
            return self._f(CA_k, T_k, TC)
        if self.scheme == "bwd_euler":
            return self._f(CA_k1, T_k1, TC)
        if self.scheme == "trapezoid":
            a0, b0 = self._f(CA_k, T_k, TC)
            a1, b1 = self._f(CA_k1, T_k1, TC)
            return 0.5 * (a0 + a1), 0.5 * (b0 + b1)
        h = self.dt
        a1, b1 = self._f(CA_k, T_k, TC)
        a2, b2 = self._f(CA_k + 0.5 * h * a1, T_k + 0.5 * h * b1, TC)
        a3, b3 = self._f(CA_k + 0.5 * h * a2, T_k + 0.5 * h * b2, TC)
        a4, b4 = self._f(CA_k + h * a3, T_k + h * b3, TC)
        return (a1 + 2 * a2 + 2 * a3 + a4) / 6.0, (b1 + 2 * b2 + 2 * b3 + b4) / 6.0

    def __call__(self, x, y):
        CA_k = Denormalization.min_max_col(x[:, 0:1], "CA_k", self.I_metrics)
        T_k = Denormalization.min_max_col(x[:, 1:2], "T_k", self.I_metrics)
        TC = Denormalization.min_max_col(x[:, 2:3], "TC_k", self.I_metrics)
        CA_1 = Denormalization.min_max_col(y[:, 0:1], "CA_k1", self.D_metrics)
        T_1 = Denormalization.min_max_col(y[:, 1:2], "T_k1", self.D_metrics)

        phi_CA, phi_T = self._increment(CA_k, T_k, CA_1, T_1, TC)
        res = torch.cat(
            [CA_1 - (CA_k + self.dt * phi_CA), T_1 - (T_k + self.dt * phi_T)], dim=1
        )
        rngs = torch.tensor(self.ot_ranges, device=res.device, dtype=res.dtype)
        return res / (rngs.view(1, -1) + 1e-8)


def rollout(model, I_S_test, I_metrics, D_metrics, device):
    """
    Apply the one-step model recursively over the horizon, vectorized across
    trajectories. Only the coolant schedule is read from the inputs; every state
    after the first is the model's own prediction.
    """
    CA, T = I_S_test[:, 0, 0].copy(), I_S_test[:, 0, 1].copy()
    traj = np.empty((I_S_test.shape[0], cstr.SEQ_LEN, 2))

    with torch.no_grad():
        for k in range(cstr.SEQ_LEN):
            raw = np.stack([CA, T, I_S_test[:, k, 2]], axis=1)
            xn = M.normalize(raw, AR_I_S_keys, I_metrics)
            y = model(torch.tensor(xn, dtype=torch.float32, device=device))
            yn = y.cpu().numpy()
            CA = Denormalization.min_max_col(yn[:, 0], "CA_k1", D_metrics)
            T = Denormalization.min_max_col(yn[:, 1], "T_k1", D_metrics)
            traj[:, k, 0], traj[:, k, 1] = CA, T

    return traj


def autoregressive_predictions(I_S_data, D_S_data, splits, device):
    """
    Train a single-transition model on the transitions of the same training
    trajectories, then roll it out over the test horizon.

    The transition batch holds as many transitions as a batch of trajectories
    does for the recurrent model, so both see the same number of updates.
    """
    idx_train, idx_val, idx_test = splits
    tr_in, tr_ot = extract_transitions(I_S_data, D_S_data, idx_train)
    va_in, va_ot = extract_transitions(I_S_data, D_S_data, idx_val)
    te_in, te_ot = extract_transitions(I_S_data, D_S_data, idx_test)

    _, I_metrics = Normalization.min_max(pd.DataFrame(tr_in, columns=AR_I_S_keys))
    _, D_metrics = Normalization.min_max(pd.DataFrame(tr_ot, columns=AR_D_S_keys))

    batch = M.B_D * cstr.SEQ_LEN

    def loader(a, b, shuffle):
        ds = torch.utils.data.TensorDataset(
            torch.tensor(M.normalize(a, AR_I_S_keys, I_metrics), dtype=torch.float32),
            torch.tensor(M.normalize(b, AR_D_S_keys, D_metrics), dtype=torch.float32),
        )
        return torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=shuffle)

    # Collocation points over the range of states actually visited
    rng = np.random.default_rng(7)
    coll = np.stack(
        [
            rng.uniform(
                I_metrics[k]["min"], I_metrics[k]["max"], M.N_C_P * cstr.SEQ_LEN
            )
            for k in AR_I_S_keys
        ],
        axis=1,
    )
    coll_ds = torch.utils.data.TensorDataset(
        torch.tensor(M.normalize(coll, AR_I_S_keys, I_metrics), dtype=torch.float32)
    )
    coll_loader = torch.utils.data.DataLoader(coll_ds, batch_size=batch, shuffle=True)

    layers = [3] + [AR_WIDTH] * AR_DEPTH + [2]
    model = ANN(layers, activation=nn.Tanh).to(device)

    print(
        f"\nTraining the autoregressive one-step baseline "
        f"({sum(p.numel() for p in model.parameters()):,} parameters, "
        f"{len(tr_in):,} transitions)"
    )
    model = fit(
        model,
        (
            loader(tr_in, tr_ot, True),
            loader(va_in, va_ot, False),
            loader(te_in, te_ot, False),
        ),
        coll_loader,
        OneStepPhysics(I_metrics, D_metrics),
        "./logs/onestep_ann.pth",
        device,
    )

    # One-step accuracy before any compounding, which separates representational
    # error from error accumulated along the rollout
    with torch.no_grad():
        xt = torch.tensor(
            M.normalize(te_in, AR_I_S_keys, I_metrics),
            dtype=torch.float32,
            device=device,
        )
        step_pred = denorm_outputs(model(xt).cpu().numpy(), D_metrics, keys=AR_D_S_keys)
    step_mae = np.abs(te_ot - step_pred).mean(axis=0)
    print(
        f"  one-step accuracy, no compounding : "
        f"CA {step_mae[0]:.4e} mol/L, T {step_mae[1]:.4e} K"
    )

    return rollout(model, I_S_data[idx_test], I_metrics, D_metrics, device), step_mae


# ================================= figures ==================================
def plot_trajectories(t, truth, pred, tc, save_dir, fontsize=19, fontfamily="Arial"):
    """Draw truth against the recurrent prediction for a few test trajectories."""
    n = truth.shape[0]
    fig, axes = plt.subplots(3, n, figsize=(5.2 * n, 11), dpi=300, sharex=True)
    axes = np.atleast_2d(axes)
    panels = (
        (0, "Concentration of A (mol/L)", "slateblue"),
        (1, "Reactor Temperature (K)", "indianred"),
    )

    for j in range(n):
        for row, (col, label, color) in enumerate(panels):
            ax = axes[row, j]
            ax.plot(t, truth[j, :, col], "--", lw=1.8, color="black", label="Truth")
            ax.plot(t, pred[j, :, col], "-", lw=1.8, color=color, label="PINN")
            if j == 0:
                ax.set_ylabel(label, fontsize=fontsize, fontfamily=fontfamily)
            if row == 0:
                ax.legend(
                    frameon=False, prop={"family": fontfamily, "size": fontsize - 4}
                )

        ax = axes[2, j]
        ax.step(t, tc[j], where="post", lw=1.8, color="darkslategray")
        if j == 0:
            ax.set_ylabel(
                "Coolant Temperature (K)", fontsize=fontsize, fontfamily=fontfamily
            )
        ax.set_xlabel("Time (min)", fontsize=fontsize, fontfamily=fontfamily)

    for ax in axes.ravel():
        for axis in ("x", "y"):
            ax.tick_params(
                axis=axis,
                which="major",
                labelsize=fontsize - 3,
                labelfontfamily=fontfamily,
            )

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "trajectories.png"), bbox_inches="tight")
    plt.close()


def plot_error_vs_step(t, errors, save_dir, fontsize=19, fontfamily="Arial"):
    """Mean absolute error against position along the horizon, every model."""
    colors = {
        "bound": "darkgrey",
        "per-step": "seagreen",
        "autoregressive": "darkorange",
        "recurrent": "slateblue",
    }
    fig, axes = plt.subplots(2, 1, figsize=(8, 9), dpi=300, sharex=True)
    panels = (
        (0, "Absolute error in concentration (mol/L)"),
        (1, "Absolute error in temperature (K)"),
    )

    for ax, (col, label) in zip(axes, panels):
        for name, err in errors.items():
            ax.plot(
                t,
                err[:, :, col].mean(axis=0),
                "-",
                lw=1.8,
                color=colors.get(name, "black"),
                label=name,
            )
        ax.set_yscale("log")
        ax.set_ylabel(label, fontsize=fontsize - 2, fontfamily=fontfamily)
        ax.legend(frameon=False, prop={"family": fontfamily, "size": fontsize - 5})
        for axis in ("x", "y"):
            ax.tick_params(
                axis=axis,
                which="major",
                labelsize=fontsize - 3,
                labelfontfamily=fontfamily,
            )

    axes[-1].set_xlabel("Time (min)", fontsize=fontsize, fontfamily=fontfamily)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "error_vs_step.png"), bbox_inches="tight")
    plt.close()


# =================================== main ===================================
if __name__ == "__main__":
    torch.manual_seed(1234)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("logs", exist_ok=True)
    save_dir = "compare"
    os.makedirs(save_dir, exist_ok=True)

    I_S_data, D_S_data, norm_I_S_data, norm_D_S_data, I_S_metrics, D_S_metrics = (
        M.prepare()
    )
    data = M.data_module(
        norm_I_S_data=norm_I_S_data,
        norm_D_S_data=norm_D_S_data,
        labeled_data_batch_size=M.B_D,
        physics_coll_data_size=M.N_C_P,
        physics_coll_batch_size=M.B_C_P,
    )
    splits = (data.idx_train, data.idx_val, data.idx_test)
    idx_test = data.idx_test

    truth = D_S_data[idx_test]
    t = cstr.deltaT * np.arange(1, cstr.SEQ_LEN + 1)

    # ---------------- 1. The four models ----------------
    recurrent, physics = load_recurrent(device)
    preds = {
        "recurrent": denorm_outputs(
            predict_sequences(recurrent, norm_I_S_data[idx_test], device), D_S_metrics
        )
    }

    preds["bound"], n_groups = bound_predictions(I_S_data, D_S_data, idx_test)
    print(
        f"Lower bound for any function of the instantaneous input: "
        f"{n_groups:,} distinct input vectors over "
        f"{len(idx_test) * cstr.SEQ_LEN:,} predicted states"
    )

    preds["per-step"] = perstep_predictions(
        data, norm_I_S_data, idx_test, (I_S_metrics, D_S_metrics), device
    )
    preds["autoregressive"], step_mae = autoregressive_predictions(
        I_S_data, D_S_data, splits, device
    )

    errors = {name: np.abs(truth - p) for name, p in preds.items()}

    # ---------------- 2. Accuracy of the recurrent model ----------------
    groups = {"CA_k1": ["CA_k1"], "T_k1": ["T_k1"], "Overall": cstr.D_S_keys}
    traj_metrics = Analyze.error_metrics(
        pd.DataFrame(truth.reshape(-1, 2), columns=cstr.D_S_keys),
        pd.DataFrame(preds["recurrent"].reshape(-1, 2), columns=cstr.D_S_keys),
        groups,
    )
    print(
        f"\nRecurrent model on {len(idx_test)} held-out test trajectories "
        f"({len(idx_test) * cstr.SEQ_LEN:,} predicted states):"
    )
    print(traj_metrics.to_string(index=False))

    x_t = torch.tensor(norm_I_S_data[idx_test], dtype=torch.float32, device=device)
    with torch.no_grad():
        res = physics(x_t, recurrent(x_t)).cpu().numpy()
    res_l2 = np.linalg.norm(res, axis=2)
    print(
        f"  unrolled physics residual ({physics.scheme}): "
        f"mean {res_l2.mean():.4e}, max {res_l2.max():.4e}"
    )

    # ---------------- 3. Cross-model comparison ----------------
    quarter = cstr.SEQ_LEN // 4
    rows = []
    for name in ("bound", "per-step", "autoregressive", "recurrent"):
        err = errors[name]
        first_q = err[:, :quarter, :].mean(axis=(0, 1))
        last_q = err[:, -quarter:, :].mean(axis=(0, 1))
        rows.append(
            {
                "model": name,
                "mae_CA": err[:, :, 0].mean(),
                "mae_T": err[:, :, 1].mean(),
                "mae_T_first_quarter": first_q[1],
                "mae_T_last_quarter": last_q[1],
                "mae_T_final_step": err[:, -1, 1].mean(),
                "growth_T": last_q[1] / first_q[1],
            }
        )
    comparison = pd.DataFrame(rows)

    print(
        f"\nAll models over the {cstr.SEQ_LEN * cstr.deltaT:g} min horizon, "
        f"{len(idx_test)} test trajectories:"
    )
    print(comparison.to_string(index=False))

    ref = comparison.set_index("model")["mae_T"]
    print()
    for other in ("bound", "per-step", "autoregressive"):
        factor = ref[other] / ref["recurrent"]
        verdict = "better" if factor >= 1 else "worse"
        print(
            f"  recurrent vs {other:<15}: "
            f"{max(factor, 1 / factor):.1f}x {verdict} in temperature"
        )

    # ---------------- 4. Figures and persistence ----------------
    plot_trajectories(
        t,
        truth[:N_SHOW],
        preds["recurrent"][:N_SHOW],
        I_S_data[idx_test][:N_SHOW, :, 2],
        save_dir,
    )
    plot_error_vs_step(t, errors, save_dir)

    step_df = pd.DataFrame({"t": t, "mean_residual": res_l2.mean(axis=0)})
    for name, err in errors.items():
        step_df[f"mae_CA_{name}"] = err[:, :, 0].mean(axis=0)
        step_df[f"mae_T_{name}"] = err[:, :, 1].mean(axis=0)

    summary = pd.DataFrame(
        [
            {
                "run": os.path.basename(os.getcwd()),
                "cell": M.CELL,
                "scheme": M.SCHEME,
                "hidden": M.hidden,
                "n_layers": M.layers,
                "epochs": M.epochs,
                "phys_weight": M.PHYS_WEIGHT,
                "seq_len": cstr.SEQ_LEN,
                "dt": cstr.deltaT,
                "n_test_traj": len(idx_test),
                "mae_CA": errors["recurrent"][:, :, 0].mean(),
                "mae_T": errors["recurrent"][:, :, 1].mean(),
                "onestep_mae_CA": step_mae[0],
                "onestep_mae_T": step_mae[1],
                "mean_residual": res_l2.mean(),
            }
        ]
    )

    with pd.ExcelWriter(os.path.join(save_dir, "compare.xlsx")) as writer:
        traj_metrics.to_excel(writer, sheet_name="Trajectory_Metrics", index=False)
        comparison.to_excel(writer, sheet_name="Model_Comparison", index=False)
        step_df.to_excel(writer, sheet_name="Error_Vs_Step", index=False)
        summary.to_excel(writer, sheet_name="Run_Summary", index=False)
    summary.to_csv(os.path.join(save_dir, "summary.csv"), index=False)
    comparison.to_csv(os.path.join(save_dir, "model_comparison.csv"), index=False)

    print(f"\nSaved results to {save_dir}")
