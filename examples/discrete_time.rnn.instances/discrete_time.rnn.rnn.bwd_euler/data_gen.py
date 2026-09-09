import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.stats import qmc

import cstr_model as cstr

"""
Generate labeled trajectories for the recurrent state-transition CSTR model.

Inputs
------
- Process specifications, sampling interval deltaT and horizon SEQ_LEN
- Operating envelope for the initial state and the coolant schedule

Outputs
-------
- I_S_data : (N, SEQ_LEN, 3) inputs per step, [CA_0, T_0, TC_k]
- D_S_data : (N, SEQ_LEN, 2) outputs per step, [CA_k1, T_k1]

The initial state is carried as a constant context feature because the
recurrent hidden state starts at zero. Runaway trajectories are screened
against T_CEILING and the retained fraction reported. See README.md.
"""

N_TRAJ = 1200  # trajectories requested


def rk4_increment(CA, T, TC, h):
    """Classical RK fourth-order increment."""
    a1, b1 = cstr.derivatives(CA, T, TC)
    a2, b2 = cstr.derivatives(CA + 0.5 * h * a1, T + 0.5 * h * b1, TC)
    a3, b3 = cstr.derivatives(CA + 0.5 * h * a2, T + 0.5 * h * b2, TC)
    a4, b4 = cstr.derivatives(CA + h * a3, T + h * b3, TC)
    return (a1 + 2 * a2 + 2 * a3 + a4) / 6.0, (b1 + 2 * b2 + 2 * b3 + b4) / 6.0


def sample_designs(n_traj: int, seed: int):
    """Latin hypercube design over the initial state and the coolant levels."""
    design = qmc.LatinHypercube(d=2 + cstr.N_SEG, seed=seed).random(n_traj)  # type: ignore

    lo_CA, hi_CA = cstr.bounds["CA_0"]
    lo_T, hi_T = cstr.bounds["T_0"]
    lo_TC, hi_TC = cstr.bounds["TC_k"]

    CA_0 = lo_CA + design[:, 0] * (hi_CA - lo_CA)
    T_0 = lo_T + design[:, 1] * (hi_T - lo_T)
    levels = lo_TC + design[:, 2:] * (hi_TC - lo_TC)
    return CA_0, T_0, levels


def coolant_schedule(levels: np.ndarray):
    """Expand segment levels into a per-step, piecewise-constant schedule."""
    per_seg = int(np.ceil(cstr.SEQ_LEN / cstr.N_SEG))
    return np.repeat(levels, per_seg)[: cstr.SEQ_LEN]


def integrate_trajectory(CA_0: float, T_0: float, tc_seq: np.ndarray):
    """
    Integrate one trajectory under a zero-order hold on the coolant temperature.

    Each constant-coolant segment is integrated in a single call, with the
    solution sampled on the step grid.
    """
    per_seg = int(np.ceil(cstr.SEQ_LEN / cstr.N_SEG))
    CA, T = CA_0, T_0
    states = []

    step = 0
    while step < cstr.SEQ_LEN:
        n = min(per_seg, cstr.SEQ_LEN - step)
        TC = tc_seq[step]
        t_eval = cstr.deltaT * np.arange(1, n + 1)
        sol = solve_ivp(
            lambda t, y: list(cstr.derivatives(y[0], y[1], TC)),
            t_span=(0.0, t_eval[-1]),
            y0=[CA, T],
            method="Radau",
            rtol=1e-11,
            atol=1e-13,
            t_eval=t_eval,
        )
        if not sol.success:
            return None
        states.append(sol.y.T)
        CA, T = sol.y[0, -1], sol.y[1, -1]
        step += n

    return np.concatenate(states, axis=0)  # (SEQ_LEN, 2), states u_1 .. u_T


def main():
    CA_0, T_0, levels = sample_designs(N_TRAJ, seed=42)

    I_S_data, D_S_data = [], []
    discarded = 0
    for i in range(N_TRAJ):
        tc_seq = coolant_schedule(levels[i])
        traj = integrate_trajectory(CA_0[i], T_0[i], tc_seq)

        if traj is None or not np.isfinite(traj).all():
            discarded += 1
            continue
        if traj[:, 1].max() > cstr.T_CEILING:
            discarded += 1
            continue

        ctx = np.repeat([[CA_0[i], T_0[i]]], cstr.SEQ_LEN, axis=0)  # (SEQ_LEN, 2)
        I_S_data.append(np.concatenate([ctx, tc_seq.reshape(-1, 1)], axis=1))
        D_S_data.append(traj)

    I_S_data = np.asarray(I_S_data, dtype=np.float64)
    D_S_data = np.asarray(D_S_data, dtype=np.float64)

    n_traj = len(I_S_data)
    index = pd.DataFrame(
        {
            "traj": np.repeat(np.arange(n_traj), cstr.SEQ_LEN),
            "step": np.tile(np.arange(cstr.SEQ_LEN), n_traj),
        }
    )
    I_S_flat = pd.DataFrame(
        I_S_data.reshape(-1, len(cstr.I_S_keys)), columns=cstr.I_S_keys
    )
    D_S_flat = pd.DataFrame(
        D_S_data.reshape(-1, len(cstr.D_S_keys)), columns=cstr.D_S_keys
    )
    pd.concat([index, I_S_flat], axis=1).to_excel("I_S_data.xlsx", index=False)
    pd.concat([index, D_S_flat], axis=1).to_excel("D_S_data.xlsx", index=False)

    retained = 100 * len(I_S_data) / N_TRAJ
    horizon = cstr.SEQ_LEN * cstr.deltaT
    tau = cstr.V / cstr.q
    CA, T = D_S_data[:, :, 0], D_S_data[:, :, 1]

    print(
        f"Generated {len(I_S_data)}/{N_TRAJ} trajectories "
        f"({retained:.1f}% retained, {discarded} discarded as runaway)"
    )
    print(
        f"  horizon      : {cstr.SEQ_LEN} steps x {cstr.deltaT} min = "
        f"{horizon:g} min ({horizon / tau:g} residence times)"
    )
    print(f"  CA range     : [{CA.min():.4f}, {CA.max():.4f}] mol/L")
    print(f"  T  range     : [{T.min():.2f}, {T.max():.2f}] K")

    spans = np.ptp(D_S_data[:, :, 1], axis=1)
    print(
        f"  within-trajectory T span : median {np.median(spans):.1f} K, "
        f"max {spans.max():.1f} K"
    )

    u_curr = np.concatenate([I_S_data[:, :1, 0:2], D_S_data[:, :-1, :]], axis=1)
    TC = I_S_data[:, :, 2]
    _, dT = cstr.derivatives(u_curr[:, :, 0], u_curr[:, :, 1], TC)
    _, phi = rk4_increment(u_curr[:, :, 0], u_curr[:, :, 1], TC, cstr.deltaT)
    err_eu = np.abs(D_S_data[:, :, 1] - (u_curr[:, :, 1] + cstr.deltaT * dT))
    err_rk = np.abs(D_S_data[:, :, 1] - (u_curr[:, :, 1] + cstr.deltaT * phi))

    print(f"\nOne-step truncation error in T over {err_eu.size:,} transitions:")
    print(f"  forward Euler : mean {err_eu.mean():.3e}  max {err_eu.max():.3e} K")
    print(f"  RK4           : mean {err_rk.mean():.3e}  max {err_rk.max():.3e} K")
    print(f"\nSaved I_S_data.xlsx and D_S_data.xlsx " f"({len(I_S_flat):,} rows each)")


if __name__ == "__main__":
    main()
