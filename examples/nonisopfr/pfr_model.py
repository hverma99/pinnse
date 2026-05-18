import numpy as np
from scipy.integrate import solve_ivp


class NonisoPFR:
    def __init__(
        self,
        reactions,  # list of reaction
        species,  # list of species
        C0,  # inlet species molar concentration dictionary {species: C0} [mol/L]
        F0,  # inlet molar flowrate dictionary {species: F0} [mol/s]
        T0,  # inlet temperature [K]
        Ua,  # heat transfer coefficient [J/(m3-s-K)]
        Ta,  # reactor coolant fluid temperature [K]
        V_span,  # reactor volume span (0, V_total) [L]
        Cp,  # species heat capacity dictionary {species: Cp} [J/(mol-K)]
        delta_Hrxn,  # heat of reaction dictionary {rxn: ΔH} [J/mol]
        V_ch,  # characteristic scale for volume [L]
        F_ch,  # characteristic scale for flows [mol/s]
        T_ch,  # characteristic scale for temperature [K]
        Cp_ch,  # characteristic scale for heat capacities [J/(mol-K)]
        H_ch,  # characteristic scale for heat of reaction [J/mol]
    ):
        self.reactions = reactions
        self.species = species
        self.C0 = C0
        self.F0 = F0
        self.T0 = T0
        self.Ua = Ua
        self.Ta = Ta
        self.V_span = V_span
        self.Cp = Cp
        self.delta_Hrxn = delta_Hrxn
        self.delta_H = np.array([delta_Hrxn[i] for i in range(len(self.reactions))])
        self.V_ch = V_ch
        self.F_ch = F_ch
        self.T_ch = T_ch
        self.Cp_ch = Cp_ch
        self.H_ch = H_ch

    def concentrations(self, F, T):
        """
        Given molar flows F (array): return concentrations assuming ideal gas behavior
        """
        F_tot = np.sum(F)
        C_Tot_0 = sum(self.C0.values())
        C_arr = C_Tot_0 * (F / F_tot) * (self.T0 / T)
        return {sp: C_arr[i] for i, sp in enumerate(self.species)}

    def net_rate(self, T, C):
        """
        Returns the overall net rate of individual species across reactions
        """
        r_net = np.zeros(len(self.species), dtype=float)
        for rxn in self.reactions:
            r_net += np.array(rxn.rate(T, C))
        return r_net

    def odes(self, V, y):
        """
        y = [F1, F2, ..., Fn, T, Ta]
        returns: dF/dV, dT/dV, dTa/dV
        """
        N = len(self.species)
        F = y[:N]
        T = y[N]
        Ta = y[N + 1]
        C = self.concentrations(F, T)

        r_net = self.net_rate(T, C)

        # material balance equations
        dF_dV = (self.V_ch / self.F_ch) * r_net

        rate = np.array([rxn.rate(T, C) for rxn in self.reactions])
        rate_A = rate[:, 0]

        # energy balance equations
        heat_gen = np.dot(-self.delta_H, -rate_A)
        Cp_flow = np.dot(F, np.array([self.Cp[sp] for sp in self.species]))
        heat_rem = Ta - T

        dT_dV = (
            heat_rem * self.Ua * self.V_ch / (self.F_ch * self.Cp_ch)
            + heat_gen * (self.H_ch * self.V_ch / (self.F_ch * self.Cp_ch * self.T_ch))
        ) / Cp_flow

        dTa_dV = 0

        return np.concatenate([dF_dV, [dT_dV, dTa_dV]])

    def simulate(self, t_eval: np.ndarray | None = None):
        # initial state vector
        y0 = np.array([self.F0[sp] for sp in self.species] + [self.T0] + [self.Ta])
        sol = solve_ivp(
            fun=self.odes,
            t_span=(self.V_span[0], self.V_span[1]),
            y0=y0,
            method="BDF",
            atol=1e-8,
            rtol=1e-6,
            t_eval=t_eval,
        )
        return sol


class Reaction:
    def __init__(
        self,
        species,  # species
        stoich,  # stoichiometric coeffiecient dictionary {species: nu}
        k0,  # pre-exponential factor [units]
        E,  # activation energy at reference temperature T_r (J/mol)
        T_R,  # reference temperature [K]
        T_ch,  # characteristic temperature [K]
        C_ch,  # characterisitc concentration [mol/L]
    ):
        self.species = species
        self.stoich = stoich
        self.k0 = k0
        self.E = E
        self.R = 8.314
        self.T_R = T_R
        self.T_ch = T_ch
        self.C_ch = C_ch

    def rate(self, T, C):

        exp = self.k0 * np.exp((self.E / (self.R * self.T_ch)) * (1 / self.T_R - 1 / T))

        r_A = exp
        for sp, v in self.stoich.items():
            if v < 0:
                r_A *= C[sp] ** (-v) * self.C_ch ** (-v)
        r_A = -r_A

        rates = []
        for sp in self.species:
            v_ij = self.stoich.get(sp, 0.0)
            v_Aj = self.stoich.get("A", None)
            rates.append((v_ij / v_Aj) * r_A)
        return rates
