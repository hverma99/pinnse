import numpy as np

"""
Non-isothermal CSTR process model.

Benchmark CSTR with a single irreversible exothermic reaction A -> B and a
cooling jacket, adapted from Seborg et al., Process Dynamics and Control.
The states are the concentration of A and the reactor temperature; the coolant
temperature is the manipulated variable.

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

# Admissible operating bounds
bounds = {
    "CA_k": (0.1, 1.0),  # mol/L
    "T_k": (320.0, 380.0),  # K
    "TC_k": (280.0, 320.0),  # K
}

# Sampling interval
deltaT = 0.0005


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
