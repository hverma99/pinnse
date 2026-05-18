import numpy as np
import pandas as pd
from pfr_model import CM
from scipy.stats import qmc
from tqdm import tqdm

"""
Generate labeled input-output (I_S/D_S) datasets for the isothermal PFR process (CM).

Inputs
------
- Process specifications
- Admissible operating bounds for I_S
- Labeled dataset size used for sample generation

Outputs
-------
- I_S : sampled input dataset
- D_S : SciPy generated dataset corresponding to I_S
"""

# Process and reaction system specifications
species = ["O2", "CO2", "H2O", "C6H6", "C4H2O3"]
N_species = len(species)
nu = np.array(
    [
        [-4.5, 2.0, 2.0, -1.0, 1.0],
        [-7.5, 6.0, 3.0, -1.0, 0.0],
        [-3.0, 4.0, 1.0, 0.0, -1.0],
    ]
)
key_species = ["C6H6", "C6H6", "C4H2O3"]
k = np.array([7.7e6, 6.31e7, 2.33e4])  # 1/s
E = np.array([105.2, 124.89, 89.66]) * 1000  # J/mol

# Admissible operating ranges
F_bds = [(625, 5200), (0, 1100), (0, 800), (40, 275), (0, 60)]  # kmol/h
F_bds = [(lower / 3.6, upper / 3.6) for lower, upper in F_bds]  # mol/s
P_bds = (1 * 1e5, 2 * 1e5)  # Pa, factor of 1e5 to convert bar to Pa
T_bds = (350 + 273.15, 450 + 273.15)  # K
V_bds = (0, 250)  # m3

N_D = 25000  # Labeled dataset size


def I_S_samples(seed=42):
    """
    Generate input samples for the PFR process (EFM).

    Inputs
    ------
    seed : int, optional, default=42
           Random seed used to initialize the NumPy random number generator
     for reproducible sampling..

    Returns
    -------
    I_S : pd.DataFrame
           DataFrame containing the generated input samples with columns:
           - F_in_i  : inlet molar flowrates for each of components
           - P       : PFR pressure
           - T       : PFR temperature
           - V       : PFR volume

    Notes
    -----
    - Samples are drawn over the bounds in `F_bds`, `P_bds`, `T_bds`, and `V_bds`.
    - Oversampling is used to collect enough feasible samples.
    - The feed O2-to-C6H6 molar ratio is constrained to be at least 10.
    - The returned DataFrame represents the sampled I_S used for data generation.
    """
    bounds = F_bds + [P_bds] + [T_bds] + [V_bds]

    low_bds = np.array([b[0] for b in bounds], dtype=float)
    up_bds = np.array([b[1] for b in bounds], dtype=float)
    sampler = qmc.LatinHypercube(d=len(bounds), rng=np.random.default_rng(seed))

    spec_index = {s: i for i, s in enumerate(species)}
    i_O2, i_C6H6 = spec_index["O2"], spec_index["C6H6"]

    oversample_factor = 3
    N_collected = 0
    collected = []

    pbar = tqdm(total=N_D, desc="Accepted Samples")
    while N_collected < N_D:
        remaining = N_D - N_collected
        batch_size = max(1024, remaining * oversample_factor)

        unit_pts = sampler.random(n=batch_size)
        X_batch = qmc.scale(unit_pts, l_bounds=low_bds, u_bounds=up_bds)
        X_batch = X_batch.astype(np.float32, copy=False)

        # Enforce feed O2 to feed C6H6 molar ratio constraint
        mask = (X_batch[:, i_O2] / X_batch[:, i_C6H6]) >= 10
        valid = X_batch[mask]

        if valid.size == 0:
            continue

        collected.append(valid)
        N_collected += valid.shape[0]
        pbar.update(min(valid.shape[0], remaining))

    pbar.close()
    X = np.vstack(collected)[:N_D]
    cols = [f"F_in_{sp}" for sp in species] + ["P", "T", "V"]
    I_S = pd.DataFrame(X, columns=cols)
    return I_S


def D_S_samples(I_S: pd.DataFrame):
    """
    Generate D_S for the PFR process by running SciPy numerical solver
    for each input sample in I_S.

    Inputs
    ------
    I_S : pd.DataFrame
        DataFrame containing the sampled input conditions for the PFR process.

    Returns
    -------
    D_S : pd.DataFrame
        DataFrame containing the simulated output samples with columns:
        - X_j   : conversion for reaction j

    Notes
    -----
    - For each sampled input condition, the conversion-based formulation (CM)
      of the isothermal PFR model is solved using the specified kinetic and
      process parameters.
    - The model solution provides the reaction conversions at the
      specified reactor volume.
    - The collected reaction conversions form the labeled output dataset `D_S`.
    """

    F_in_cols = [f"F_in_{sp}" for sp in species]
    F_in_all = I_S[F_in_cols].to_numpy(dtype=np.float32)
    P_all = I_S["P"].to_numpy(dtype=np.float32)
    T_all = I_S["T"].to_numpy(dtype=np.float32)
    V_all = I_S["V"].to_numpy(dtype=np.float32)
    Y = np.zeros((N_D, nu.shape[0]), dtype=np.float32)

    for i in tqdm(range(N_D)):
        F_in, P, T, V = F_in_all[i], P_all[i], T_all[i], V_all[i]
        model = CM(species, nu, key_species, k, E, F_in, P, T)
        sol = model.solve(V)
        Conv = sol.y[:, -1]
        Y[i, :] = Conv.astype(np.float32, copy=False)
    D_S = pd.DataFrame(Y, columns=[f"X_R{j+1}" for j in range(nu.shape[0])])
    return D_S


if __name__ == "__main__":
    """
    Generate and save input and output datasets
    """
    I_S = I_S_samples()
    D_S = D_S_samples(I_S)

    I_S_data = I_S[[*(f"F_in_{sp}" for sp in species), "P", "T", "V"]]
    D_S_data = D_S[[*(f"X_R{j+1}" for j in range(nu.shape[0]))]]

    I_S_data.to_excel("I_S_data.xlsx", index=False)
    D_S_data.to_excel("D_S_data.xlsx", index=False)
