import numpy as np, pandas as pd
from scipy.stats import qmc
from scipy.integrate import solve_ivp
from cstr_model import bounds, deltaT, derivatives

"""
Generate labeled state-transition pairs for the discrete-time CSTR model.

Inputs
------
- Process specifications and the sampling interval deltaT
- Admissible operating bounds for the current state and manipulated variable

Outputs
-------
- I_S : sampled current states and manipulated variable [CA_k, T_k, TC_k]
- D_S : SciPy-integrated next states [CA_k1, T_k1]
"""

N_D = 20000


def rhs(t, y, TC):
    dCA_dt, dT_dt = derivatives(y[0], y[1], TC)
    return [dCA_dt, dT_dt]


def advance(CA, T, TC):
    """Advance one state by a single sampling interval with RK45."""
    sol = solve_ivp(
        rhs,
        t_span=(0.0, deltaT),
        y0=[CA, T],
        args=(TC,),
        method="RK45",
        rtol=1e-10,
        atol=1e-12,
        dense_output=False,
    )
    return sol.y[0, -1], sol.y[1, -1]


def main():
    cols = list(bounds.keys())
    lb = np.array([bounds[c][0] for c in cols])
    ub = np.array([bounds[c][1] for c in cols])

    sampler = qmc.LatinHypercube(d=len(cols), rng=np.random.default_rng(42))
    X = qmc.scale(sampler.random(N_D), lb, ub)

    Y = np.empty((N_D, 2))
    for n in range(N_D):
        Y[n, 0], Y[n, 1] = advance(X[n, 0], X[n, 1], X[n, 2])

    I_S_data = pd.DataFrame(X, columns=cols)
    D_S_data = pd.DataFrame(Y, columns=["CA_k1", "T_k1"])

    I_S_data.to_excel("I_S_data.xlsx", index=False)
    D_S_data.to_excel("D_S_data.xlsx", index=False)


if __name__ == "__main__":
    main()
