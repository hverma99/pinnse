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
        theta,
        R: float = 8.314,
    ):
        self.I_S_metrics = I_S_metrics
        self.D_S_metrics = D_S_metrics
        self.species = species
        self.nu = nu
        self.key_species = key_species
        self.k = k
        self.theta = theta
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
        device, dtype = x.device, x.dtype

        nu_t = torch.tensor(self.nu, device=device, dtype=dtype)  # (nr, ns)
        k_t = torch.tensor(self.k, device=device, dtype=dtype)  # (nr,)

        # inverse parameter estimation - trainable parameter
        E_t = self.theta.to(device=device, dtype=dtype) * 1e3  # (nr,)

        ord_t = torch.tensor(self.orders, device=device, dtype=dtype)  # (nr,)
        key_idx_t = torch.tensor(self.key_idx, device=device, dtype=torch.long)  # (nr,)

        # Extract normalized inputs and outputs
        # x = [F_in_i..., P, T, V]
        P = x[:, self.N_comp : self.N_comp + 1]  # (batch, 1)
        T = x[:, self.N_comp + 1 : self.N_comp + 2]  # (batch, 1)
        V = x[:, self.N_comp + 2 : self.N_comp + 3]  # (batch, 1)
        F_ot = y  # (batch, N_comp)

        # Compute dF_ot/dV in normalized coordinates
        dF_dV = torch.zeros_like(F_ot)  # (batch, N_comp)
        for i in range(self.N_comp):
            F_i = F_ot[:, i : i + 1]
            grad_F_i = torch.autograd.grad(
                outputs=F_i,
                inputs=x,
                grad_outputs=torch.ones_like(F_i),
                create_graph=True,
                retain_graph=True,
                allow_unused=False,
            )[0][
                :, -1:
            ]  # derivative wrt V
            dF_dV[:, i : i + 1] = grad_F_i

        # Denormalize F_ot, P, T, and obtain ranges
        F_ot_dim, F_ot_rng = Denormalization.min_max_pfr(
            F_ot,
            self.D_S_metrics,
            keys=[f"F_ot_{sp}" for sp in self.species],
        )  # (batch, N_comp), (N_comp,)

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

        # Chain rule: dF_dim/dV_dim = (F_rng / V_rng) * dF/dV
        scale = F_ot_rng / V_rng  # (N_comp,)
        dF_dV_dim = dF_dV * scale.view(1, -1)  # (batch, N_comp)

        # Reaction-rate expression in dimensional variables
        F_tot = F_ot_dim.sum(dim=1, keepdim=True)  # (batch, 1)
        F_key = F_ot_dim[:, key_idx_t]  # (batch, nr)

        P_over_RT = P_dim / (self.R * T_dim)  # (batch, 1)
        pref = k_t.view(1, -1) * torch.exp(-E_t.view(1, -1) / (self.R * T_dim))
        rate = pref * (P_over_RT * F_key / F_tot).pow(ord_t.view(1, -1))  # (batch, nr)

        # Material balance: dF/dV = nu^T r
        dF_dV_phys = rate @ nu_t  # (batch, N_comp)

        # Residual in dimensional form, then scaled back
        res = dF_dV_dim - dF_dV_phys
        res_scaled = res / (scale.view(1, -1) + 1e-8)

        return res_scaled

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.physics_residual(x, y)


class Boundary:
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
