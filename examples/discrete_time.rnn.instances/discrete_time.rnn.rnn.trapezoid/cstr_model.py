import numpy as np

"""
Non-isothermal CSTR process model, trajectory formulation.

Same reactor as the `discrete_time` example (Seborg et al.), differing only in
the sampling interval and the sampling domain, both set for trajectory rather
than single-transition prediction.

    dCA/dt = (q/V)(CA_f - CA) - k0 exp(-E/(R T)) CA
    dT/dt  = (q/V)(T_f - T) + ((-dH)/(rho Cp)) k0 exp(-E/(R T)) CA
             + (UA/(V rho Cp))(TC - T)
"""

# Process specifications and kinetic parameters
q = 100.0  # Volumetric flowrate, L/min
V = 100.0  # Reactor volume, L
rho = 1000.0  # Density, g/L
Cp = 0.239  # Heat capacity, J/(g K)
dH = -5.0e4  # Heat of reaction, J/mol
E_over_R = 8750.0  # Activation energy over gas constant, K
k0 = 7.2e10  # Pre-exponential factor, 1/min
UA = 5.0e4  # Heat-transfer coefficient times area, J/(min K)
CA_f = 1.0  # Feed concentration of A, mol/L
T_f = 350.0  # Feed temperature, K

# Sampling interval.
deltaT = 0.005

# Trajectory length and horizon
SEQ_LEN = 200  # steps; SEQ_LEN * deltaT = 1.0 min = one residence time
N_SEG = 4  # piecewise-constant segments of the coolant schedule

# Operating bounds for trajectory sampling. This reactor admits multiple
# steady states (at TC = 300 K: 324.5 K stable, 350.0 K unstable, 369.7 K
# stable), and trajectories crossing the unstable branch run away to about
# 580 K, so the envelope stays on the low-temperature branch.
bounds = {
    "CA_0": (0.20, 0.60),  # initial concentration of A, mol/L
    "T_0": (330.0, 345.0),  # initial reactor temperature, K
    "TC_k": (295.0, 306.0),  # coolant temperature, K
}

T_CEILING = 400.0  # trajectories exceeding this are discarded as runaway

I_S_keys = ["CA_0", "T_0", "TC_k"]  # input features per time step
D_S_keys = ["CA_k1", "T_k1"]  # output features per time step


def derivatives(CA, T, TC):
    """
    Inputs
    ------
    CA, T, TC : np.ndarray
        Concentration of A (mol/L), reactor and coolant temperature (K).

    Returns
    -------
    dCA_dt, dT_dt : np.ndarray
        Rates of change, in mol/(L min) and K/min.
    """
    rate = k0 * np.exp(-E_over_R / T) * CA
    dCA_dt = (q / V) * (CA_f - CA) - rate
    dT_dt = (
        (q / V) * (T_f - T)
        + ((-dH) / (rho * Cp)) * rate
        + (UA / (V * rho * Cp)) * (TC - T)
    )
    return dCA_dt, dT_dt
