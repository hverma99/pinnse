import numpy as np
from scipy.integrate import solve_ivp


class CM:
    """
    Isothermal PFR model in conversion-based formulation (CM).

    This module defines a first-principles plug-flow reactor model in which the
    reaction conversions are the state variables. The model evaluates reaction
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
        self.F_key_in = np.array(
            [F_in[self.spec_index[key_species[j]]] for j in range(self.nr)]
        )

    def mat_balance(self, V, X_j):
        """
        Evaluate the species material-balance equations.

        Inputs
        ------
        V : float
            Reactor volume coordinate (in m3).
        X_j : np.ndarray
            Reaction conversion at volume V.

        Returns
        -------
        dXdV : np.ndarray
            Derivative of reaction conversions with respect to reactor volume (in 1/m3).
        """
        F = self.F_in + self.nu.T @ (self.F_key_in * X_j)
        F_tot = F.sum()

        dXdV = np.zeros(self.nr)
        for j in range(self.nr):
            key = self.key_species[j]
            i_key = self.spec_index[key]
            nu_key = self.nu[j, i_key]
            y_key = F[i_key] / F_tot
            C_key = (self.P / (self.R * self.T)) * y_key

            r_star = (
                self.k[j]
                * np.exp(-self.E[j] / (self.R * self.T))
                * (C_key ** abs(nu_key))
            )
            dXdV[j] = r_star / ((-nu_key) * self.F_key_in[j])

        return dXdV

    def solve(self, V_end, V_eval=None, rtol=1e-12, atol=1e-12):
        """
        Solve the PFR model from inlet to reactor volume V.

        Inputs
        ------
        V_end : float
            Final reactor volume.
        V_eval: np.ndarry
            Evaluation reactor volume coordinates.
        rtol : float, optional, default=1e-12
            Relative tolerance for the ODE solver.
        atol : float, optional, default=1e-12
            Absolute tolerance for the ODE solver.

        Returns
        -------
        X_j : np.ndarray
            reaction conversions at the reactor outlet.
        """
        X_j0 = np.zeros(self.nr)
        sol = solve_ivp(
            self.mat_balance,
            (0.0, V_end),
            X_j0,
            t_eval=V_eval,
            rtol=rtol,
            atol=atol,
            dense_output=True,
        )
        X_j = sol
        return X_j
