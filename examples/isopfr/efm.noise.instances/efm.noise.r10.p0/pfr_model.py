import numpy as np
from scipy.integrate import solve_ivp


class EFM:
    """
    Isothermal PFR model in effluent-flow formulation (EFM).

    This module defines a first-principles plug-flow reactor model in which the
    species molar flowrates are the state variables. The model evaluates reaction
    rates using Arrhenius kinetics and solves the species material balances along
    the reactor volume coordinate.
    """

    def __init__(
        self,
        species: list[str],
        nu: np.ndarray,
        key_species: list[str],
        k: np.ndarray,
        E: np.ndarray,
        F_in: np.ndarray,
        P: float,
        T: float,
        R: float = 8.314,
    ):
        """
        Inputs
        ------
        species : list[str]
            Species names in the reactor model.
        nu : np.ndarray
            Stoichiometric coefficient matrix of shape (n_reactions, n_species).
        key_species : list[str]
            Key reactant species used in the rate expressions for each reaction.
        k : np.ndarray
            Pre-exponential factors for each reaction (in 1/s).
        E : np.ndarray
            Activation energies for each reaction (in J/mol).
        F_in : np.ndarray
            Inlet species molar flowrates (in mol/s).
        P : float
            Reactor pressure (in Pa).
        T : float
            Reactor temperature (in K).
        R : float, optional, default=8.314
            Gas constant (in J/mol/K).

        """
        self.species = species
        self.nu = nu
        self.key_species = key_species
        self.k = k
        self.E = E
        self.F_in = F_in
        self.P = P
        self.T = T
        self.R = R

        self.nr, self.ns = self.nu.shape
        self.spec_index = {s: i for i, s in enumerate(self.species)}

    def mat_balance(self, V, F):
        """
        Evaluate the species material-balance equations.

        Inputs
        ------
        V : float
            Reactor volume coordinate (in m3).
        F : np.ndarray
            Species molar flowrates at volume V (in mol/s).

        Returns
        -------
        dFdV : np.ndarray
            Derivative of species molar flowrates with respect to reactor volume (in mol/s/m3).
        """
        r_dim = np.zeros(self.nr)
        for j in range(self.nr):
            key = self.key_species[j]
            i_key = self.spec_index[key]
            r_dim[j] = (
                self.k[j]
                * np.exp(-self.E[j] / (self.R * self.T))
                * ((self.P / (self.R * self.T)) * F[i_key] / sum(F))
                ** np.abs(self.nu[j, i_key])
            )

        dFdV = self.nu.T @ r_dim
        return dFdV

    def solve(self, V, rtol=1e-12, atol=1e-12):
        """
        Solve the PFR model from inlet to reactor volume V.

        Inputs
        ------
        V : float
            Final reactor volume.
        rtol : float, optional, default=1e-12
            Relative tolerance for the ODE solver.
        atol : float, optional, default=1e-12
            Absolute tolerance for the ODE solver.

        Returns
        -------
        F : np.ndarray
            Species molar flowrates at the reactor outlet (in mol/s).
        """
        sol = solve_ivp(
            self.mat_balance,
            (0.0, V),
            self.F_in,
            rtol=rtol,
            atol=atol,
            dense_output=True,
        )
        F = sol.y[:, -1]
        return F
