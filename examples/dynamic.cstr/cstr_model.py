import numpy as np

"""
Non-isothermal CSTR process model, trajectory formulation.

Non-isothermal CSTR (Seborg et al.), with the sampling interval and the
sampling domain set for trajectory rather than single-transition prediction.

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

# Operating bounds for trajectory sampling.
bounds = {
    "CA_0": (0.20, 0.60),  # initial concentration of A, mol/L
    "T_0": (330.0, 345.0),  # initial reactor temperature, K
    "TC_k": (295.0, 306.0),  # coolant temperature, K
}

T_CEILING = 400.0  # trajectories exceeding this are discarded as runaway

I_S_keys = ["CA_0", "T_0", "TC_k"]  # input features per time step
D_S_keys = ["CA_k1", "T_k1"]  # output features per time step


def derivatives(CA, T, TC, exp=np.exp):
    """
    Inputs
    ------
    CA, T, TC : np.ndarray or torch.Tensor
        Concentration of A (mol/L), reactor and coolant temperature (K).
    exp : callable, optional, default=np.exp
        Exponential matching the array type, e.g. torch.exp for tensors, so the
        same kinetics serve data generation and the physics residuals.

    Returns
    -------
    dCA_dt, dT_dt : np.ndarray or torch.Tensor
        Rates of change, in mol/(L min) and K/min.
    """
    rate = k0 * exp(-E_over_R / T) * CA
    dCA_dt = (q / V) * (CA_f - CA) - rate
    dT_dt = (
        (q / V) * (T_f - T)
        + ((-dH) / (rho * Cp)) * rate
        + (UA / (V * rho * Cp)) * (TC - T)
    )
    return dCA_dt, dT_dt


def rk4_increment(CA, T, TC, h, exp=np.exp):
    """
    Classical fourth-order Runge-Kutta increment phi over one interval h, with
    the coolant temperature held constant, such that u_k1 = u_k + h * phi.
    """
    a1, b1 = derivatives(CA, T, TC, exp)
    a2, b2 = derivatives(CA + 0.5 * h * a1, T + 0.5 * h * b1, TC, exp)
    a3, b3 = derivatives(CA + 0.5 * h * a2, T + 0.5 * h * b2, TC, exp)
    a4, b4 = derivatives(CA + h * a3, T + h * b3, TC, exp)
    return (a1 + 2 * a2 + 2 * a3 + a4) / 6.0, (b1 + 2 * b2 + 2 * b3 + b4) / 6.0
