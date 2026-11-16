"""
First-principles isothermal PFR models, one per formulation of the species
material balances along the reactor volume coordinate:

- EFM, effluent-flow formulation: the species molar flowrates are the states.
- ERM, extent of reaction-based formulation: the reaction extents are the states.
- CM, conversion-based formulation: the reaction conversions are the states.

All three evaluate the reaction rates with Arrhenius kinetics. They generate
the labeled datasets (data_gen.py) and the reference profiles of check.py.
"""

import numpy as np
from scipy.integrate import solve_ivp


class IsothermalPFR:
    """
    Shared set-up and solver of the isothermal PFR models; the subclasses
    define the states, their inlet values and their balances.
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

    def inlet_state(self) -> np.ndarray:
        """States at the reactor inlet (V = 0)."""
        raise NotImplementedError

    def mat_balance(self, V, state):
        """Derivative of the states with respect to the reactor volume."""
        raise NotImplementedError

    def solve(self, V_end, V_eval=None, rtol=1e-12, atol=1e-12):
        """
        Solve the PFR model from inlet to reactor volume V_end.

        Inputs
        ------
        V_end : float
            Final reactor volume.
        V_eval: np.ndarray, optional
            Evaluation reactor volume coordinates.
        rtol : float, optional, default=1e-12
            Relative tolerance for the ODE solver.
        atol : float, optional, default=1e-12
            Absolute tolerance for the ODE solver.

        Returns
        -------
        sol : OdeResult
            SciPy solution; sol.y[:, -1] holds the states at the reactor outlet.
        """
        return solve_ivp(
            self.mat_balance,
            (0.0, V_end),
            self.inlet_state(),
            t_eval=V_eval,
            rtol=rtol,
            atol=atol,
            dense_output=True,
        )


class EFM(IsothermalPFR):
    """Isothermal PFR model in effluent-flow formulation (EFM)."""

    def inlet_state(self):
        return self.F_in

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


class ERM(IsothermalPFR):
    """Isothermal PFR model in extent of reaction-based formulation (ERM)."""

    def inlet_state(self):
        return np.zeros(self.nr)

    def mat_balance(self, V, E_j):
        """
        Evaluate the material-balance equations in terms of reaction extents.

        Inputs
        ------
        V : float
            Reactor volume coordinate (in m3).
        E_j : np.ndarray
            Reaction extents at volume V (in mol/s).

        Returns
        -------
        dEdV : np.ndarray
            Derivative of extent of reactions with respect to reactor volume (in mol/s/m3).
        """
        F = self.F_in + self.nu.T @ E_j
        F_tot = F.sum()

        r_dim = np.zeros(self.nr)
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
            r_dim[j] = r_star / (-nu_key)

        dEdV = r_dim
        return dEdV


class CM(IsothermalPFR):
    """Isothermal PFR model in conversion-based formulation (CM)."""

    def __init__(self, species, nu, key_species, k, E, F_in, P, T, R=8.314):
        super().__init__(species, nu, key_species, k, E, F_in, P, T, R)
        self.F_key_in = np.array(
            [F_in[self.spec_index[key_species[j]]] for j in range(self.nr)]
        )

    def inlet_state(self):
        return np.zeros(self.nr)

    def mat_balance(self, V, X_j):
        """
        Evaluate the material-balance equations in terms of reaction conversions.

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


MODELS = {"EFM": EFM, "ERM": ERM, "CM": CM}
