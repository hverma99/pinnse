"""
Generate labeled input-output (I_S/D_S) datasets for the isothermal PFR process.

Inputs
------
- Process specifications (process.py)
- Admissible operating bounds for I_S
- Labeled dataset size used for sample generation

Outputs
-------
- I_S : sampled input dataset
- D_S : SciPy generated dataset corresponding to I_S, in the formulation of
  the case (effluent flowrates, reaction extents or reaction conversions)

Each case folder's data_gen.py calls write_datasets() with its formulation.
"""

import numpy as np
import pandas as pd
from scipy.stats import qmc
from tqdm import tqdm

from common.pfr_model import MODELS
from common.process import (
    N_D,
    E,
    F_bds,
    P_bds,
    T_bds,
    V_bds,
    k,
    key_species,
    nu,
    output_columns,
    species,
)


def I_S_samples(seed=42):
    """
    Generate input samples for the PFR process.

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


def D_S_samples(I_S: pd.DataFrame, formulation: str):
    """
    Generate D_S for the PFR process by running SciPy numerical solver
    for each input sample in I_S.

    Inputs
    ------
    I_S : pd.DataFrame
        DataFrame containing the sampled input conditions for the PFR process.
    formulation : str
        "EFM", "ERM" or "CM": the reactor model solved and the outputs recorded.

    Returns
    -------
    D_S : pd.DataFrame
        DataFrame containing the simulated output samples, i.e. the states of
        the formulation at the reactor outlet, with columns:
        - EFM : F_ot_i, effluent molar flowrates for each of components
        - ERM : E_Rj, extent of reaction for reaction j
        - CM  : X_Rj, conversion for reaction j
    """
    Model = MODELS[formulation]
    columns = output_columns(formulation)

    F_in_cols = [f"F_in_{sp}" for sp in species]
    F_in_all = I_S[F_in_cols].to_numpy(dtype=np.float32)
    P_all = I_S["P"].to_numpy(dtype=np.float32)
    T_all = I_S["T"].to_numpy(dtype=np.float32)
    V_all = I_S["V"].to_numpy(dtype=np.float32)
    Y = np.zeros((len(I_S), len(columns)), dtype=np.float32)

    for i in tqdm(range(len(I_S))):
        F_in, P, T, V = F_in_all[i], P_all[i], T_all[i], V_all[i]
        model = Model(species, nu, key_species, k, E, F_in, P, T)
        Y[i, :] = model.solve(V).y[:, -1]
    D_S = pd.DataFrame(Y, columns=columns)
    return D_S


def write_datasets(formulation: str):
    """
    Generate and save the input and output datasets of a formulation to
    I_S_data.xlsx and D_S_data.xlsx in the current folder.
    """
    I_S = I_S_samples()
    D_S = D_S_samples(I_S, formulation)

    I_S.to_excel("I_S_data.xlsx", index=False)
    D_S.to_excel("D_S_data.xlsx", index=False)
