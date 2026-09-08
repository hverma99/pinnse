import torch
import cstr_model as cstr
from pinnse import Denormalization

"""
Discrete-time physics residual for the state-transition CSTR formulation.

In a state-transition formulation the current states are themselves network
inputs, so differentiating the network output with respect to an explicit time
input does not represent the total derivative along the system trajectory.
The governing dynamics are therefore enforced in discrete-time form: the
residual is the mismatch between the predicted next state and a physics-based
update of the current state,

    r = u_k1_pred - (u_k + DT f(u_k, TC_k))

which for this example uses forward Euler. Higher-order schemes, for instance
a Runge-Kutta update, are obtained by replacing the update term only. Note
that this residual requires no automatic differentiation with respect to the
inputs; it uses the same `(x, y) -> residual` interface as the continuous-time
residuals of the other examples, so no change to the pinnse package is needed.
"""


class Physics:
    def __init__(
        self,
        I_S_metrics: dict,
        D_S_metrics: dict,
        dt: float = cstr.DT,
    ):
        self.I_S_metrics = I_S_metrics
        self.D_S_metrics = D_S_metrics
        self.dt = dt

        self.in_keys = ["CA_k", "T_k", "TC_k"]
        self.ot_keys = ["CA_k1", "T_k1"]

        # Output ranges, used to nondimensionalize the residuals
        self.ot_ranges = [
            D_S_metrics[key]["max"] - D_S_metrics[key]["min"] for key in self.ot_keys
        ]

    def physics_residual(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the forward-Euler state-transition residual.

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

        # Right-hand side of the governing ODEs, evaluated at the current state
        rate = cstr.k0 * torch.exp(-cstr.E_over_R / T_k) * CA_k
        dCA_dt = (cstr.q / cstr.V) * (cstr.CA_f - CA_k) - rate
        dT_dt = (
            (cstr.q / cstr.V) * (cstr.T_f - T_k)
            + ((-cstr.dH) / (cstr.rho * cstr.Cp)) * rate
            + (cstr.UA / (cstr.V * cstr.rho * cstr.Cp)) * (TC_k - T_k)
        )

        # Discrete-time residual: predicted next state minus the Euler update
        res_CA = CA_k1 - (CA_k + self.dt * dCA_dt)
        res_T = T_k1 - (T_k + self.dt * dT_dt)

        res = torch.cat([res_CA, res_T], dim=1)
        rngs = torch.tensor(self.ot_ranges, device=res.device, dtype=res.dtype)
        return res / (rngs.view(1, -1) + 1e-8)

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.physics_residual(x, y)
