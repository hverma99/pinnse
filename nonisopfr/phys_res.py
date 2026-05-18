import torch
from pfr_model import Reaction


class Physics:
    def __init__(
        self,
        species: list[str],
        delta_Hrxn: dict[int, float],
        Cp: dict[str, float],
        Ua: float,
        T_R: float,
        R: float,
        F_ch: float,
        C_ch: float,
        T_ch: float,
        V_ch: float,
        H_ch: float,
        Cp_ch: float,
    ) -> None:
        self.species = species
        self.Ua = Ua
        self.R = R

        self.F_ch = F_ch
        self.C_ch = C_ch
        self.T_ch = T_ch
        self.V_ch = V_ch
        self.H_ch = H_ch
        self.Cp_ch = Cp_ch

        self.delta_Hrxn = {k: v / self.H_ch for k, v in delta_Hrxn.items()}
        self.T_R = T_R / T_ch
        self.Cp = {k: v / self.Cp_ch for k, v in Cp.items()}

        r1 = Reaction(
            species=species,
            stoich={"A": -1, "B": 1},
            k0=10,
            E=4000 * R,
            T_R=self.T_R,
            T_ch=T_ch,
            C_ch=C_ch,
        )

        r2 = Reaction(
            species=species,
            stoich={"A": -2, "C": 3},
            k0=0.5,
            E=9000 * R,
            T_R=self.T_R,
            T_ch=T_ch,
            C_ch=C_ch,
        )

        self.reactions = [r1, r2]

        self.N_species = len(species)

    def reaction_rates(
        self,
        C: torch.Tensor,
        T: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        batch, _ = C.shape

        eps = 1e-8
        T_safe = torch.clamp(T, min=eps)
        C_safe = torch.clamp(C, min=0.0)

        delta_H = torch.tensor(
            [self.delta_Hrxn[i] for i in range(len(self.reactions))],
            dtype=C.dtype,
            device=C.device,
        ).view(1, -1)

        r_net_list: list[torch.Tensor] = []
        heat_gen = torch.zeros((batch, 1), dtype=C.dtype, device=C.device)

        for rxn_idx, rxn in enumerate(self.reactions):

            exponent = (rxn.E / (rxn.R * self.T_ch)) * (1.0 / rxn.T_R - 1.0 / T_safe)

            exponent = torch.clamp(exponent, min=-60.0, max=60.0)

            k = rxn.k0 * torch.exp(exponent)
            k = k.view(-1, 1)

            rA = k

            for sp, v in rxn.stoich.items():
                if v < 0:
                    idx = self.species.index(sp)
                    rA = rA * (C_safe[:, idx : idx + 1] ** (-v)) * (self.C_ch ** (-v))

            rA = -rA

            rates: list[torch.Tensor] = []

            for sp in self.species:
                v_ij = rxn.stoich.get(sp, 0.0)
                v_Aj = rxn.stoich["A"]
                rates.append((v_ij / v_Aj) * rA)

            r_net_list.append(torch.cat(rates, dim=1))

            dH = delta_H[0, rxn_idx]
            heat_gen = heat_gen + (-rA) * (-dH)

        r_net = torch.stack(r_net_list, dim=0).sum(dim=0)

        heat_gen = heat_gen * (
            self.V_ch * self.H_ch / (self.F_ch * self.T_ch * self.Cp_ch)
        )

        return r_net, heat_gen

    def conc_residual(
        self,
        C: torch.Tensor,
        C0: torch.Tensor,
        F: torch.Tensor,
        T0: torch.Tensor,
        T: torch.Tensor,
    ) -> torch.Tensor:

        eps = 1e-8

        C_tot0 = C0.sum(dim=1, keepdim=True)
        F_tot = torch.clamp(F.sum(dim=1, keepdim=True), min=eps)
        T_safe = torch.clamp(T, min=eps)

        ratio_F = F / F_tot
        ratio_T = T0 / T_safe

        C_ideal = C_tot0 * ratio_F * ratio_T

        return C - C_ideal

    def EB_RHS(
        self,
        T: torch.Tensor,
        Ta: torch.Tensor,
        heat_gen: torch.Tensor,
        F: torch.Tensor,
    ) -> torch.Tensor:

        eps = 1e-8

        Cp_vec = torch.tensor(
            [self.Cp[sp] for sp in self.species],
            dtype=F.dtype,
            device=F.device,
        ).view(1, -1)

        Cp_tot = (F * Cp_vec).sum(dim=1, keepdim=True)
        Cp_tot = torch.clamp(Cp_tot, min=eps)

        Ua_term = (self.Ua * self.V_ch / (self.F_ch * self.Cp_ch)) * (Ta - T)

        return (Ua_term + heat_gen) / Cp_tot

    def derivative(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute dy/dx using autograd.
        """

        return torch.autograd.grad(
            y,
            x,
            grad_outputs=torch.ones_like(y),
            create_graph=True,
            retain_graph=True,
        )[0]

    def physics_residual(
        self,
        x: torch.Tensor,
        y_pred: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute non-isothermal PFR physics residuals.

        Inputs
        ------
        x : torch.Tensor
            Input tensor with columns:

            [F0_A, C0_A, T0, Ta0, V]

        y_pred : torch.Tensor
            Output tensor with columns:

            [F_A, F_B, F_C, C_A, C_B, C_C, T, Ta]

        Returns
        -------
        torch.Tensor
            Physics residual tensor with columns:

            [MB_A, MB_B, MB_C, C_A, C_B, C_C, EB, CEB]
        """

        N = self.N_species

        C0_A = x[:, 1:2]
        T0 = x[:, 2:3]

        C0 = torch.cat(
            [
                C0_A,
                torch.zeros_like(C0_A),
                torch.zeros_like(C0_A),
            ],
            dim=1,
        )

        F = y_pred[:, :N]
        C = y_pred[:, N : 2 * N]
        T = y_pred[:, 2 * N : 2 * N + 1]
        Ta = y_pred[:, 2 * N + 1 : 2 * N + 2]

        dF_dV_list: list[torch.Tensor] = []

        for i in range(N):
            grad_Fi = self.derivative(x, F[:, i])
            dF_dV_i = grad_Fi[:, 4:5]
            dF_dV_list.append(dF_dV_i)

        dF_dV = torch.cat(dF_dV_list, dim=1)

        grad_T = self.derivative(x, T)
        dT_dV = grad_T[:, 4:5]

        grad_Ta = self.derivative(x, Ta)
        dTa_dV = grad_Ta[:, 4:5]

        r_net, heat_gen = self.reaction_rates(C, T)

        res_mb = dF_dV - (self.V_ch / self.F_ch) * r_net

        res_c = self.conc_residual(
            C=C,
            C0=C0,
            F=F,
            T0=T0,
            T=T,
        )

        res_eb = dT_dV - self.EB_RHS(
            T=T,
            Ta=Ta,
            heat_gen=heat_gen,
            F=F,
        )

        res_ceb = dTa_dV

        return torch.cat(
            [
                res_mb,
                res_c,
                res_eb,
                res_ceb,
            ],
            dim=1,
        )

    def __call__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        return self.physics_residual(x, y)


class Boundary:
    def __init__(
        self,
        species: list[str],
    ) -> None:
        self.species = species
        self.N_comp = len(species)

    def bnd_residual(
        self,
        x: torch.Tensor,
        y_pred: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute inlet-boundary residuals for the non-isothermal PFR.

        Inputs
        ------
        x : torch.Tensor
            Boundary input tensor with columns:

            [F0_A, C0_A, T0, Ta0, V]

            For inlet boundary points, V = 0.

        y_pred : torch.Tensor
            Output tensor with columns:

            [F_A, F_B, F_C, C_A, C_B, C_C, T, Ta]

        Returns
        -------
        torch.Tensor
            Boundary residual tensor with columns:

            [F_A - F0_A,
             F_B - 0,
             F_C - 0,
             C_A - C0_A,
             C_B - 0,
             C_C - 0,
             T - T0,
             Ta - Ta0]
        """

        N = self.N_comp

        F0_A = x[:, 0:1]
        C0_A = x[:, 1:2]
        T0 = x[:, 2:3]
        Ta0 = x[:, 3:4]

        F0 = torch.cat(
            [
                F0_A,
                torch.zeros_like(F0_A),
                torch.zeros_like(F0_A),
            ],
            dim=1,
        )

        C0 = torch.cat(
            [
                C0_A,
                torch.zeros_like(C0_A),
                torch.zeros_like(C0_A),
            ],
            dim=1,
        )

        F = y_pred[:, :N]
        C = y_pred[:, N : 2 * N]
        T = y_pred[:, 2 * N : 2 * N + 1]
        Ta = y_pred[:, 2 * N + 1 : 2 * N + 2]

        res_F = F - F0
        res_C = C - C0
        res_T = T - T0
        res_Ta = Ta - Ta0

        return torch.cat(
            [
                res_F,
                res_C,
                res_T,
                res_Ta,
            ],
            dim=1,
        )

    def __call__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        return self.bnd_residual(x, y)
