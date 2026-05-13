import torch
import numpy as np
from utils import Denormalization


class Physics:
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
        device, dtype = x.device, x.dtype

        nu_t = torch.tensor(self.nu, device=device, dtype=dtype)  # (nr, ns)
        k_t = torch.tensor(self.k, device=device, dtype=dtype)  # (nr,)
        E_t = torch.tensor(self.E, device=device, dtype=dtype)  # (nr,)
        ord_t = torch.tensor(self.orders, device=device, dtype=dtype)  # (nr,)
        key_idx_t = torch.tensor(self.key_idx, device=device, dtype=torch.long)  # (nr,)

        # Extract normalized inputs and outputs
        # x = [F_in_i..., P, T, V]
        F_in = x[:, : self.N_comp]  # (batch, ns)
        P = x[:, self.N_comp : self.N_comp + 1]  # (batch, 1)
        T = x[:, self.N_comp + 1 : self.N_comp + 2]  # (batch, 1)
        V = x[:, self.N_comp + 2 : self.N_comp + 3]  # (batch, 1)
        E_R = y  # (batch, nr)

        # Compute dE_R/dV in normalized coordinates
        dE_dV = torch.zeros_like(E_R)  # (batch, nr)
        for j in range(self.N_rxn):
            E_j = E_R[:, j : j + 1]
            grad_E_j = torch.autograd.grad(
                outputs=E_j,
                inputs=x,
                grad_outputs=torch.ones_like(E_j),
                create_graph=True,
                retain_graph=True,
                allow_unused=False,
            )[0][
                :, -1:
            ]  # derivative wrt V
            dE_dV[:, j : j + 1] = grad_E_j

        # Denormalize F_in, P, T, and obtain ranges
        F_in_dim, _ = Denormalization.min_max_pfr(
            F_in,
            self.I_S_metrics,
            keys=[f"F_in_{sp}" for sp in self.species],
        )  # (batch, ns), (ns,)

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

        E_R_dim, E_R_rng = Denormalization.min_max_pfr(
            E_R, self.D_S_metrics, keys=[f"E_R{j+1}" for j in range(self.N_rxn)]
        )  # (batch, nr), (nr,)

        # Chain rule: dE_dim/dV_dim = (E_rng / V_rng) * dE/dV
        scale = E_R_rng / V_rng  # (nr,)
        dE_dV_dim = dE_dV * scale.view(1, -1)  # (batch, nr)

        # Reaction-rate expression in dimensional variables
        F_ot_dim = F_in_dim + E_R_dim @ nu_t  # (batch, ns)
        F_tot = F_ot_dim.sum(dim=1, keepdim=True)  # (batch, 1)
        F_key = F_ot_dim[:, key_idx_t]  # (batch, nr)

        P_over_RT = P_dim / (self.R * T_dim)  # (batch, 1)
        pref = k_t.view(1, -1) * torch.exp(-E_t.view(1, -1) / (self.R * T_dim))
        rate = pref * (P_over_RT * F_key / F_tot).pow(ord_t.view(1, -1))  # (batch, nr)

        # Material balance: dE/dV = r
        dE_dV_phys = rate

        # Residual in dimensional form, then scaled back
        res = dE_dV_dim - dE_dV_phys
        res_scaled = res / (scale.view(1, -1) + 1e-8)

        return res_scaled

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.physics_residual(x, y)


class Boundary:
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
            Normalized output tensor with columns
            [E_j]

        Returns
        -------
        torch.Tensor
            Boundary residual tensor of shape (batch, N_comp), enforcing
            E_j = 0 at the inlet boundary.
        """
        E_dim, E_rng = Denormalization.min_max_pfr(
            y, self.D_S_metrics, keys=[f"E_R{j+1}" for j in range(self.N_rxn)]
        )

        # Dimensional BC: E_R = 0 at V = 0
        res_dim = E_dim
        res = res_dim / (E_rng.view(1, -1) + 1e-8)
        return res

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.boundary_residual(x, y)
