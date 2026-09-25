import torch
import cstr_model as cstr
from pinnse import Denormalization

SCHEMES = ("euler", "rk4", "bwd_euler", "trapezoid")


class Physics:
    """
    Unrolled discrete-time residual, applied along each predicted trajectory,

        r_k = u_k1 - (u_k + deltaT * phi),   k = 0 .. T-1

    where u_k is the network's own prediction from the previous step and the
    initial condition supplies step 0. The residual therefore chains
    predictions: a trajectory satisfies it only if successive predictions are
    mutually consistent, which a single-transition residual cannot express.
    """

    def __init__(
        self,
        I_S_metrics: dict,
        D_S_metrics: dict,
        dt: float = cstr.deltaT,
        scheme: str = "rk4",
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

        self.in_keys = cstr.I_S_keys
        self.ot_keys = cstr.D_S_keys

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
        interval is u_k + dt * phi.
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
        Compute the unrolled discrete-time residual along each trajectory.

        Inputs
        ------
        x : torch.Tensor
            Normalized input sequence of shape (batch, seq_len, 3), with
            features [CA_0, T_0, TC_k]. The first two are constant along the
            sequence and carry the initial condition.
        y : torch.Tensor
            Normalized output sequence of shape (batch, seq_len, 2), with
            features [CA_k1, T_k1], holding the predicted states u_1 .. u_T.

        Returns
        -------
        torch.Tensor
            Nondimensionalized residual tensor of shape (batch, seq_len, 2).
        """
        if x.dim() != 3 or y.dim() != 3:
            raise ValueError(
                "Physics expects sequence-shaped tensors (batch, seq_len, features); "
                f"received x {tuple(x.shape)} and y {tuple(y.shape)}."
            )

        # Initial condition, taken from the constant context features at step 0
        CA_init = Denormalization.min_max_col(x[:, :1, 0:1], "CA_0", self.I_S_metrics)
        T_init = Denormalization.min_max_col(x[:, :1, 1:2], "T_0", self.I_S_metrics)

        # Coolant temperature at every step
        TC_k = Denormalization.min_max_col(x[:, :, 2:3], "TC_k", self.I_S_metrics)

        # Predicted next states u_1 .. u_T
        CA_next = Denormalization.min_max_col(y[:, :, 0:1], "CA_k1", self.D_S_metrics)
        T_next = Denormalization.min_max_col(y[:, :, 1:2], "T_k1", self.D_S_metrics)

        # Current states u_0 .. u_{T-1}: the initial condition, then the
        # network's own predictions shifted by one step
        CA_curr = torch.cat([CA_init, CA_next[:, :-1, :]], dim=1)
        T_curr = torch.cat([T_init, T_next[:, :-1, :]], dim=1)

        # Physics-based update of the current state over one sampling interval
        phi_CA, phi_T = self._increment(CA_curr, T_curr, CA_next, T_next, TC_k)

        res_CA = CA_next - (CA_curr + self.dt * phi_CA)
        res_T = T_next - (T_curr + self.dt * phi_T)

        res = torch.cat([res_CA, res_T], dim=2)
        rngs = torch.tensor(self.ot_ranges, device=res.device, dtype=res.dtype)
        return res / (rngs.view(1, 1, -1) + 1e-8)

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.physics_residual(x, y)
