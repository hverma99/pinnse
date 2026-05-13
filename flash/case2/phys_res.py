import torch
from pinnse import Denormalization


class Physics:
    def __init__(self, I_S_metrics: dict, D_S_metrics: dict, species: list[str]):
        self.I_S_metrics = I_S_metrics
        self.D_S_metrics = D_S_metrics
        self.species = species
        self.N_comp = len(species)

    def physics_residual(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute flash-equilibrium residuals in physical space.

        Inputs
        ------
        x : torch.Tensor
            Normalized input tensor with columns [T_flash, P_flash, Z_i]

        y : torch.Tensor
            Normalized output tensor with columns [VF, Y_i, X_i]

        Returns
        -------
        torch.Tensor
            Residual tensor containing:
            - component material-balance residuals
            - liquid mole-fraction summation residual
            - vapor mole-fraction summation residual
        """
        N_comp = self.N_comp
        z_i = torch.cat(
            [
                Denormalization.min_max_col(
                    x[:, 2 + i : 3 + i], f"Z_{sp}", self.I_S_metrics
                )
                for i, sp in enumerate(self.species)
            ],
            dim=1,
        )

        vf = Denormalization.min_max_col(y[:, 0:1], "VF", self.D_S_metrics)

        y_i = torch.cat(
            [
                Denormalization.min_max_col(
                    y[:, 1 + i : 2 + i], f"Y_{sp}", self.D_S_metrics
                )
                for i, sp in enumerate(self.species)
            ],
            dim=1,
        )

        x_i = torch.cat(
            [
                Denormalization.min_max_col(
                    y[:, 1 + N_comp + i : 2 + N_comp + i], f"X_{sp}", self.D_S_metrics
                )
                for i, sp in enumerate(self.species)
            ],
            dim=1,
        )

        r_mb = (1 - vf) * x_i + vf * y_i - z_i
        r_xnm = 1.0 - x_i.sum(dim=1, keepdim=True)
        r_ynm = 1.0 - y_i.sum(dim=1, keepdim=True)

        return torch.cat([r_mb, r_xnm, r_ynm], dim=1)

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.physics_residual(x, y)
