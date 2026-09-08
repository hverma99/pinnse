import numpy as np, pandas as pd
from scipy.stats import qmc
from scipy.integrate import solve_ivp
from cstr_model import BOUNDS, DT, derivatives

"""
Data generation for the discrete-time (state-transition) CSTR formulation.

The independent-input variables are the current reactor state and the current
manipulated variable, I_S = {CA_k, T_k, TC_k}, and the dependent-output
variables are the state one sampling interval later, D_S = {CA_k1, T_k1}. The
admissible operating domain is sampled using Latin hypercube sampling, and each
sample is advanced over one interval DT with an accurate adaptive solver to
produce the labeled dataset.
"""

N_SAMPLES = 20000
RANDOM_STATE = 42


def rhs(t, y, TC):
    dCA_dt, dT_dt = derivatives(y[0], y[1], TC)
    return [dCA_dt, dT_dt]


def advance(CA, T, TC):
    """Advance one state by a single sampling interval with RK45."""
    sol = solve_ivp(
        rhs,
        t_span=(0.0, DT),
        y0=[CA, T],
        args=(TC,),
        method="RK45",
        rtol=1e-10,
        atol=1e-12,
        dense_output=False,
    )
    return sol.y[0, -1], sol.y[1, -1]


def main():
    cols = list(BOUNDS.keys())
    lb = np.array([BOUNDS[c][0] for c in cols])
    ub = np.array([BOUNDS[c][1] for c in cols])

    sampler = qmc.LatinHypercube(d=len(cols), rng=np.random.default_rng(RANDOM_STATE))
    X = qmc.scale(sampler.random(N_SAMPLES), lb, ub)

    Y = np.empty((N_SAMPLES, 2))
    for n in range(N_SAMPLES):
        Y[n, 0], Y[n, 1] = advance(X[n, 0], X[n, 1], X[n, 2])

    I_S_data = pd.DataFrame(X, columns=cols)
    D_S_data = pd.DataFrame(Y, columns=["CA_k1", "T_k1"])

    I_S_data.to_excel("I_S_data.xlsx", index=False)
    D_S_data.to_excel("D_S_data.xlsx", index=False)

    # Consistency of the forward-Euler residual with the reference solution.
    # The physics residual enforces a forward-Euler update, so the sampling
    # interval must be small enough that the one-step Euler error is far below
    # the accuracy targeted from the PINN model.
    dCA_dt, dT_dt = derivatives(X[:, 0], X[:, 1], X[:, 2])
    CA_euler = X[:, 0] + DT * dCA_dt
    T_euler = X[:, 1] + DT * dT_dt
    print(f"Generated {N_SAMPLES} state-transition pairs at DT = {DT} min")
    print(
        f"Max |forward-Euler - RK45|: "
        f"CA {np.abs(CA_euler - Y[:, 0]).max():.3e} mol/L, "
        f"T {np.abs(T_euler - Y[:, 1]).max():.3e} K"
    )
    print(
        f"Max state change over DT:   "
        f"CA {np.abs(Y[:, 0] - X[:, 0]).max():.3e} mol/L, "
        f"T {np.abs(Y[:, 1] - X[:, 1]).max():.3e} K"
    )


if __name__ == "__main__":
    main()
