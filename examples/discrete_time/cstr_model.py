import numpy as np

"""
Non-isothermal CSTR process model.

Benchmark continuous stirred-tank reactor with a single irreversible
exothermic reaction A -> B and a cooling jacket, adapted from Seborg et al.,
Process Dynamics and Control. The reactor states are the concentration of A
and the reactor temperature; the coolant temperature is the manipulated
variable.

    dCA/dt = (q/V)(CA_f - CA) - k0 exp(-E/(R T)) CA
    dT/dt  = (q/V)(T_f - T)
             + ((-dH)/(rho Cp)) k0 exp(-E/(R T)) CA
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

# Admissible operating domain of the state-transition formulation
BOUNDS = {
    "CA_k": (0.1, 1.0),  # mol/L
    "T_k": (320.0, 380.0),  # K
    "TC_k": (280.0, 320.0),  # K
}

# Sampling interval of the discrete-time formulation, min. The interval is
# chosen so that the one-step forward-Euler truncation error stays small
# relative to the state change over the interval; `data_gen.py` reports this
# comparison, since the accuracy of the discrete-time residual bounds the
# physics loss that the PINN model can attain.
DT = 0.0005


def derivatives(CA, T, TC):
    """
    Evaluate the CSTR right-hand side f(x, u) in dimensional variables.

    Inputs
    ------
    CA : np.ndarray
        Concentration of A, mol/L.
    T : np.ndarray
        Reactor temperature, K.
    TC : np.ndarray
        Coolant temperature, K.

    Returns
    -------
    dCA_dt : np.ndarray
        Rate of change of concentration, mol/(L min).
    dT_dt : np.ndarray
        Rate of change of reactor temperature, K/min.
    """
    rate = k0 * np.exp(-E_over_R / T) * CA
    dCA_dt = (q / V) * (CA_f - CA) - rate
    dT_dt = (
        (q / V) * (T_f - T)
        + ((-dH) / (rho * Cp)) * rate
        + (UA / (V * rho * Cp)) * (TC - T)
    )
    return dCA_dt, dT_dt
