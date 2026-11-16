"""
Physics and boundary residuals of the three isothermal PFR formulations.

The physics residuals evaluate the material balances in dimensional space at
the collocation points and return them scaled by the chain-rule factor of the
normalization; the boundary residuals enforce the inlet condition at V = 0.

Each case's phys_res.py exposes the residuals it trains with as `Physics` and
`Boundary`, either one of these or a case-specific variant (efm.inverse).
"""

import torch

import numpy as np

from pinnse import Denormalization


class PFRPhysics:
    """
    Set-up and helpers shared by the physics residuals; the subclasses
    implement physics_residual() for their formulation.
    """

    def __init__(
        self,
        I_S_metrics: dict,
        D_S_metrics: dict,
        species: list[str],
        nu: np.ndarray,
        key_species: list[str],
        k: np.ndarray,
        E: np.ndarray,
        R: float = 8.314,
    ):
        self.I_S_metrics = I_S_metrics
        self.D_S_metrics = D_S_metrics
        self.species = species
        self.nu = nu
        self.key_species = key_species
        self.k = k
        self.E = E
        self.R = R

        self.N_comp = len(species)
        self.N_rxn = nu.shape[0]
        self.spec_index = {s: i for i, s in enumerate(self.species)}
        self.key_idx = [self.spec_index[sp] for sp in self.key_species]
        self.orders = np.array(
            [abs(self.nu[j, self.key_idx[j]]) for j in range(self.N_rxn)],
            dtype=np.float32,
        )

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.physics_residual(x, y)

    def physics_residual(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def activation_energies(self, device, dtype) -> torch.Tensor:
        """Activation energies of the reactions (in J/mol), shape (nr,)."""
        return torch.tensor(self.E, device=device, dtype=dtype)

    def constants(self, device, dtype):
        """Stoichiometry, kinetic parameters, orders and key-species indices."""
        nu_t = torch.tensor(self.nu, device=device, dtype=dtype)  # (nr, ns)
        k_t = torch.tensor(self.k, device=device, dtype=dtype)  # (nr,)
        E_t = self.activation_energies(device, dtype)  # (nr,)
        ord_t = torch.tensor(self.orders, device=device, dtype=dtype)  # (nr,)
        key_idx_t = torch.tensor(self.key_idx, device=device, dtype=torch.long)  # (nr,)
        return nu_t, k_t, E_t, ord_t, key_idx_t

    def split_PTV(self, x: torch.Tensor):
        """Normalized P, T and V columns of x = [F_in_i..., P, T, V]."""
        P = x[:, self.N_comp : self.N_comp + 1]  # (batch, 1)
        T = x[:, self.N_comp + 1 : self.N_comp + 2]  # (batch, 1)
        V = x[:, self.N_comp + 2 : self.N_comp + 3]  # (batch, 1)
        return P, T, V

    @staticmethod
    def d_dV(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Derivative of each output column with respect to V (normalized)."""
        dy_dV = torch.zeros_like(y)  # (batch, n_out)
        for j in range(y.shape[1]):
            y_j = y[:, j : j + 1]
            grad_y_j = torch.autograd.grad(
                outputs=y_j,
                inputs=x,
                grad_outputs=torch.ones_like(y_j),
                create_graph=True,
                retain_graph=True,
                allow_unused=False,
            )[0][:, -1:]  # derivative wrt V
            dy_dV[:, j : j + 1] = grad_y_j
        return dy_dV

    def denormalize_PTV(self, P, T, V):
        """Dimensional P and T, and the range of V."""
        P_dim, _ = Denormalization.min_max_pfr(
            P,
            self.I_S_metrics,
            keys=["P"],
        )  # (batch, 1), (1,)

        T_dim, _ = Denormalization.min_max_pfr(
            T,
            self.I_S_metrics,
            keys=["T"],
        )  # (batch, 1), (1,)

        _, V_rng = Denormalization.min_max_pfr(
            V,
            self.I_S_metrics,
            keys=["V"],
        )  # (batch, 1), (1,)
        return P_dim, T_dim, V_rng

    def denormalize_F_in(self, F_in):
        """Dimensional inlet flowrates."""
        F_in_dim, _ = Denormalization.min_max_pfr(
            F_in,
            self.I_S_metrics,
            keys=[f"F_in_{sp}" for sp in self.species],
        )  # (batch, ns), (ns,)
        return F_in_dim

    def rates(self, F_ot_dim, P_dim, T_dim, k_t, E_t, ord_t, key_idx_t):
        """Reaction rates (Arrhenius kinetics) in dimensional variables."""
        F_tot = F_ot_dim.sum(dim=1, keepdim=True)  # (batch, 1)
        F_key = F_ot_dim[:, key_idx_t]  # (batch, nr)

        P_over_RT = P_dim / (self.R * T_dim)  # (batch, 1)
        pref = k_t.view(1, -1) * torch.exp(-E_t.view(1, -1) / (self.R * T_dim))
        rate = pref * (P_over_RT * F_key / F_tot).pow(ord_t.view(1, -1))  # (batch, nr)
        return rate

    @staticmethod
    def scaled(res: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Dimensional residual scaled back by the chain-rule factor."""
        return res / (scale.view(1, -1) + 1e-8)


class EFMPhysics(PFRPhysics):
    def physics_residual(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute PFR-EFM physics residuals in dimensional space and return
        scaled nondimensional residuals.

        Inputs
        ------
        x : torch.Tensor
            Normalized input tensor with columns
            [F_in_i, P, T, V]

        y : torch.Tensor
            Normalized output tensor with columns
            [F_ot_i]

        Returns
        -------
        torch.Tensor
            Residual tensor of shape (batch, N_comp).
        """
        nu_t, k_t, E_t, ord_t, key_idx_t = self.constants(x.device, x.dtype)

        P, T, V = self.split_PTV(x)
        F_ot = y  # (batch, N_comp)

        # Compute dF_ot/dV in normalized coordinates
        dF_dV = self.d_dV(x, F_ot)  # (batch, N_comp)

        # Denormalize F_ot, P, T, and obtain ranges
        F_ot_dim, F_ot_rng = Denormalization.min_max_pfr(
            F_ot,
            self.D_S_metrics,
            keys=[f"F_ot_{sp}" for sp in self.species],
        )  # (batch, N_comp), (N_comp,)
        P_dim, T_dim, V_rng = self.denormalize_PTV(P, T, V)

        # Chain rule: dF_dim/dV_dim = (F_rng / V_rng) * dF/dV
        scale = F_ot_rng / V_rng  # (N_comp,)
        dF_dV_dim = dF_dV * scale.view(1, -1)  # (batch, N_comp)

        rate = self.rates(F_ot_dim, P_dim, T_dim, k_t, E_t, ord_t, key_idx_t)

        # Material balance: dF/dV = nu^T r
        dF_dV_phys = rate @ nu_t  # (batch, N_comp)

        # Residual in dimensional form, then scaled back
        return self.scaled(dF_dV_dim - dF_dV_phys, scale)


class ERMPhysics(PFRPhysics):
    def physics_residual(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute PFR-ERM physics residuals in dimensional space and return
        scaled nondimensional residuals.

        Inputs
        ------
        x : torch.Tensor
            Normalized input tensor with columns
            [F_in_i, P, T, V]

        y : torch.Tensor
            Normalized output tensor with columns
            [E_j]

        Returns
        -------
        torch.Tensor
            Residual tensor of shape (batch, N_rxn).
        """
        nu_t, k_t, E_t, ord_t, key_idx_t = self.constants(x.device, x.dtype)

        F_in = x[:, : self.N_comp]  # (batch, ns)
        P, T, V = self.split_PTV(x)
        E_R = y  # (batch, nr)

        # Compute dE_R/dV in normalized coordinates
        dE_dV = self.d_dV(x, E_R)  # (batch, nr)

        # Denormalize F_in, P, T, E_R, and obtain ranges
        F_in_dim = self.denormalize_F_in(F_in)
        P_dim, T_dim, V_rng = self.denormalize_PTV(P, T, V)
        E_R_dim, E_R_rng = Denormalization.min_max_pfr(
            E_R, self.D_S_metrics, keys=[f"E_R{j + 1}" for j in range(self.N_rxn)]
        )  # (batch, nr), (nr,)

        # Chain rule: dE_dim/dV_dim = (E_rng / V_rng) * dE/dV
        scale = E_R_rng / V_rng  # (nr,)
        dE_dV_dim = dE_dV * scale.view(1, -1)  # (batch, nr)

        # Reaction-rate expression in dimensional variables
        F_ot_dim = F_in_dim + E_R_dim @ nu_t  # (batch, ns)
        rate = self.rates(F_ot_dim, P_dim, T_dim, k_t, E_t, ord_t, key_idx_t)

        # Material balance: dE/dV = r
        dE_dV_phys = rate

        # Residual in dimensional form, then scaled back
        return self.scaled(dE_dV_dim - dE_dV_phys, scale)


class CMPhysics(PFRPhysics):
    def physics_residual(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute PFR-CM physics residuals in dimensional space and return
        scaled nondimensional residuals.

        Inputs
        ------
        x : torch.Tensor
            Normalized input tensor with columns
            [F_in_i, P, T, V]

        y : torch.Tensor
            Normalized output tensor with columns
            [X_j]

        Returns
        -------
        torch.Tensor
            Residual tensor of shape (batch, N_rxn).
        """
        nu_t, k_t, E_t, ord_t, key_idx_t = self.constants(x.device, x.dtype)

        F_in = x[:, : self.N_comp]  # (batch, ns)
        P, T, V = self.split_PTV(x)
        X_R = y  # (batch, nr)

        # Compute dX_R/dV in normalized coordinates
        dX_dV = self.d_dV(x, X_R)  # (batch, nr)

        # Denormalize F_in, P, T, X_R, and obtain ranges
        F_in_dim = self.denormalize_F_in(F_in)
        P_dim, T_dim, V_rng = self.denormalize_PTV(P, T, V)
        X_R_dim, X_R_rng = Denormalization.min_max_pfr(
            X_R, self.D_S_metrics, keys=[f"X_R{j + 1}" for j in range(self.N_rxn)]
        )  # (batch, nr), (nr,)

        # Chain rule: dX_dim/dV_dim = (X_rng / V_rng) * dX/dV
        scale = X_R_rng / V_rng  # (nr,)
        dX_dV_dim = dX_dV * scale.view(1, -1)  # (batch, nr)

        # Reconstruct effluent flowrates from conversions
        F_key_in_dim = F_in_dim[:, key_idx_t]  # (batch, nr)
        F_ot_dim = F_in_dim + (F_key_in_dim * X_R_dim) @ nu_t  # (batch, ns)
        rate = self.rates(F_ot_dim, P_dim, T_dim, k_t, E_t, ord_t, key_idx_t)

        # CM physics: dX_j/dV = r_j / [(-nu_key,j) * F_key,in,j]
        nu_key = nu_t[torch.arange(self.N_rxn, device=x.device), key_idx_t].view(1, -1)
        dX_dV_phys = rate / (((-nu_key) * F_key_in_dim) + 1e-12)  # (batch, nr)

        # Residual
        return self.scaled(dX_dV_dim - dX_dV_phys, scale)


class EFMBoundary:
    def __init__(self, species: list[str]):
        self.species = species
        self.N_comp = len(species)

    def boundary_residual(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute inlet boundary residuals in normalized space.

        Inputs
        ------
        x : torch.Tensor
            Normalized input tensor with columns
            [F_in_i, P, T, V]

        y : torch.Tensor
            Normalized output tensor with columns
            [F_ot_i]

        Returns
        -------
        torch.Tensor
            Boundary residual tensor of shape (batch, N_comp), enforcing
            F_ot = F_in at the inlet boundary.
        """
        F_in = x[:, : self.N_comp]  # (batch, N_comp)
        F_ot = y  # (batch, N_comp)

        res = F_in - F_ot
        return res

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.boundary_residual(x, y)


class ReactionBoundary:
    """
    Inlet boundary residual of the reaction-based formulations: the outputs
    (named <prefix>1, <prefix>2, ...) vanish at the inlet boundary.
    """

    prefix = ""

    def __init__(self, D_S_metrics: dict, N_rxn: int):
        self.D_S_metrics = D_S_metrics
        self.N_rxn = N_rxn

    def boundary_residual(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute inlet boundary residuals in normalized space.

        Inputs
        ------
        x : torch.Tensor
            Normalized input tensor with columns
            [F_in_i, P, T, V]

        y : torch.Tensor
            Normalized output tensor with one column per reaction

        Returns
        -------
        torch.Tensor
            Boundary residual tensor of shape (batch, N_rxn), enforcing
            y = 0 at the inlet boundary.
        """
        y_dim, y_rng = Denormalization.min_max_pfr(
            y,
            self.D_S_metrics,
            keys=[f"{self.prefix}{j + 1}" for j in range(self.N_rxn)],
        )

        # Dimensional BC: y = 0 at V = 0
        res_dim = y_dim
        res = res_dim / (y_rng.view(1, -1) + 1e-8)
        return res

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.boundary_residual(x, y)


class ERMBoundary(ReactionBoundary):
    """Enforces E_j = 0 at the inlet boundary."""

    prefix = "E_R"


class CMBoundary(ReactionBoundary):
    """Enforces X_j = 0 at the inlet boundary."""

    prefix = "X_R"
