import torch
import cstr_model as cstr
from pinnse import Denormalization

SCHEMES = ("euler", "rk4", "bwd_euler", "trapezoid")


class Physics:
    """
    Discrete-time state-transition residual,

        r = u_k1 - (u_k + deltaT * phi),

    where phi is the increment of the scheme named by `scheme`:

        euler      phi = f(u_k)                     explicit, 1st order
        rk4        classical four-stage increment   explicit, 4th order
        bwd_euler  phi = f(u_k1)                    implicit, 1st order
        trapezoid  phi = (f(u_k) + f(u_k1)) / 2     implicit, 2nd order

    The implicit schemes need no iterative solve, since the next state is the
    network output and is already available when the residual is formed.
    """

    def __init__(
        self,
        I_S_metrics: dict,
        D_S_metrics: dict,
        dt: float = cstr.deltaT,
        scheme: str = "euler",
    ):
        self.I_S_metrics = I_S_metrics
        self.D_S_metrics = D_S_metrics
        self.dt = dt

        scheme = scheme.lower()
        if scheme not in SCHEMES:
            raise ValueError(
                f"Unknown discretization scheme '{scheme}'; expected one of {', '.join(SCHEMES)}."
            )
        self.scheme = scheme

        self.in_keys = ["CA_k", "T_k", "TC_k"]
        self.ot_keys = ["CA_k1", "T_k1"]

        # Output ranges
        self.ot_ranges = [
            D_S_metrics[key]["max"] - D_S_metrics[key]["min"] for key in self.ot_keys
        ]

    def _f(self, CA: torch.Tensor, T: torch.Tensor, TC: torch.Tensor):
        rate = cstr.k0 * torch.exp(-cstr.E_over_R / T) * CA
        dCA_dt = (cstr.q / cstr.V) * (cstr.CA_f - CA) - rate
        dT_dt = (
            (cstr.q / cstr.V) * (cstr.T_f - T)
            + ((-cstr.dH) / (cstr.rho * cstr.Cp)) * rate
            + (cstr.UA / (cstr.V * cstr.rho * cstr.Cp)) * (TC - T)
        )
        return dCA_dt, dT_dt

    def _increment(
        self,
        CA_k: torch.Tensor,
        T_k: torch.Tensor,
        CA_k1: torch.Tensor,
        T_k1: torch.Tensor,
        TC_k: torch.Tensor,
    ):
        """
        Increment phi of the selected scheme, such that the update over one
        interval is u_k + dt * phi. The implicit schemes evaluate the
        right-hand side at the predicted next state, which is the network
        output. The manipulated variable is held constant across the interval.
        """
        if self.scheme == "euler":
            return self._f(CA_k, T_k, TC_k)

        if self.scheme == "bwd_euler":
            return self._f(CA_k1, T_k1, TC_k)

        if self.scheme == "trapezoid":
            a_k, b_k = self._f(CA_k, T_k, TC_k)
            a_k1, b_k1 = self._f(CA_k1, T_k1, TC_k)
            return 0.5 * (a_k + a_k1), 0.5 * (b_k + b_k1)

        h = self.dt
        k1_CA, k1_T = self._f(CA_k, T_k, TC_k)
        k2_CA, k2_T = self._f(CA_k + 0.5 * h * k1_CA, T_k + 0.5 * h * k1_T, TC_k)
        k3_CA, k3_T = self._f(CA_k + 0.5 * h * k2_CA, T_k + 0.5 * h * k2_T, TC_k)
        k4_CA, k4_T = self._f(CA_k + h * k3_CA, T_k + h * k3_T, TC_k)

        phi_CA = (k1_CA + 2.0 * k2_CA + 2.0 * k3_CA + k4_CA) / 6.0
        phi_T = (k1_T + 2.0 * k2_T + 2.0 * k3_T + k4_T) / 6.0
        return phi_CA, phi_T

    def physics_residual(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the discrete-time state-transition residual.

        Inputs
        ------
        x : torch.Tensor
            Normalized input tensor with columns [CA_k, T_k, TC_k].
        y : torch.Tensor
            Normalized output tensor with columns [CA_k1, T_k1].

        Returns
        -------
        torch.Tensor
            Nondimensionalized residual tensor of shape (batch, 2).
        """
        # Denormalize inputs and outputs to dimensional variables
        CA_k = Denormalization.min_max_col(x[:, 0:1], "CA_k", self.I_S_metrics)
        T_k = Denormalization.min_max_col(x[:, 1:2], "T_k", self.I_S_metrics)
        TC_k = Denormalization.min_max_col(x[:, 2:3], "TC_k", self.I_S_metrics)

        CA_k1 = Denormalization.min_max_col(y[:, 0:1], "CA_k1", self.D_S_metrics)
        T_k1 = Denormalization.min_max_col(y[:, 1:2], "T_k1", self.D_S_metrics)

        # Physics-based update of the current state over one sampling interval
        phi_CA, phi_T = self._increment(CA_k, T_k, CA_k1, T_k1, TC_k)

        # Discrete-time residual: predicted next state minus the physics update
        res_CA = CA_k1 - (CA_k + self.dt * phi_CA)
        res_T = T_k1 - (T_k + self.dt * phi_T)

        res = torch.cat([res_CA, res_T], dim=1)
        rngs = torch.tensor(self.ot_ranges, device=res.device, dtype=res.dtype)
        return res / (rngs.view(1, -1) + 1e-8)

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.physics_residual(x, y)
