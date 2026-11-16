"""
Process specification of the isothermal PFR cases: the reaction network, its
Arrhenius kinetics, the admissible operating ranges sampled for the labeled
datasets, and the output columns of each formulation.
"""

import numpy as np

# Process and reaction system specifications
species = ["O2", "CO2", "H2O", "C6H6", "C4H2O3"]
nu = np.array(  # stoichiometric coefficients, one row per reaction
    [
        [-4.5, 2.0, 2.0, -1.0, 1.0],
        [-7.5, 6.0, 3.0, -1.0, 0.0],
        [-3.0, 4.0, 1.0, 0.0, -1.0],
    ]
)
key_species = ["C6H6", "C6H6", "C4H2O3"]  # key reactant of each reaction
k = np.array([7.7e6, 6.31e7, 2.33e4])  # pre-exponential factors, 1/s
E = np.array([105.2, 124.89, 89.66]) * 1000  # activation energies, J/mol

# Admissible operating ranges
F_bds = [(625, 5200), (0, 1100), (0, 800), (40, 275), (0, 60)]  # kmol/h
F_bds = [(lower / 3.6, upper / 3.6) for lower, upper in F_bds]  # mol/s
P_bds = (1 * 1e5, 2 * 1e5)  # Pa, factor of 1e5 to convert bar to Pa
T_bds = (350 + 273.15, 450 + 273.15)  # K
V_bds = (0, 250)  # m3

N_D = 25000  # Labeled dataset size

FORMULATIONS = ("EFM", "ERM", "CM")


def output_columns(formulation: str) -> list[str]:
    """
    D_S columns of a formulation: the effluent flowrates F_ot_i (EFM), the
    reaction extents E_Rj (ERM) or the reaction conversions X_Rj (CM).
    """
    if formulation == "EFM":
        return [f"F_ot_{sp}" for sp in species]
    if formulation in ("ERM", "CM"):
        prefix = "E_R" if formulation == "ERM" else "X_R"
        return [f"{prefix}{j + 1}" for j in range(nu.shape[0])]
    raise ValueError(f"formulation must be one of {FORMULATIONS}; got {formulation!r}")
