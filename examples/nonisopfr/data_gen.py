import numpy as np
import pandas as pd
from scipy.stats import qmc
from pfr_model import NonisoPFR, Reaction
from tqdm import tqdm

species = ["A", "B", "C"]
N_species = len(species)

# Process specifications
F0_ = {"A": 100, "B": 0.0, "C": 0.0}  # mol/s
C0_ = {"A": 0.1, "B": 0.0, "C": 0.0}  # mol/L
T0_ = (273.15) + 120  # K
Ta_ = (273.15) + 50  # K
V_span_ = (0.0, 1)  # L
T_R_ = 300  # K

Cp_ = {"A": 90, "B": 90, "C": 180}  # J/(mol-K)
delta_Hrxn_ = {0: -20000, 1: -60000}  # J/(mol reactant A reacted)
Ua = 4000.0  # J/(m3-s-K)
R = 8.314  # J/(mol-K)

# Characteristic scales for process variables
F_ch = 200  # mol/s
C_ch = 0.1  # mol/L
T_ch = 800  # K
V_ch = 1  # m3
H_ch = -100000  # J/(mol A reacted)
Cp_ch = 200  # J/(mol-K)

# Evaluate nondimensional variables
F0 = {key: value / F_ch for key, value in F0_.items()}
C0 = {key: value / C_ch for key, value in C0_.items()}
T0 = T0_ / T_ch
Ta = Ta_ / T_ch
V_span = tuple(v / V_ch for v in V_span_)
T_R = T_R_ / T_ch
Cp = {key: value / Cp_ch for key, value in Cp_.items()}
delta_Hrxn = {key: value / H_ch for key, value in delta_Hrxn_.items()}

# Reaction system specifications
r1 = Reaction(
    species=species,
    stoich={"A": -1, "B": 1},
    k0=10,
    E=4000 * R,
    T_R=T_R,
    T_ch=T_ch,
    C_ch=C_ch,
)

r2 = Reaction(
    species=species,
    stoich={"A": -2, "C": 3},
    k0=0.5,
    E=9000 * R,
    T_R=T_R,
    T_ch=T_ch,
    C_ch=C_ch,
)

reactions = [r1, r2]

# Admissible operating ranges
F0_lbs, F0_ubs = np.array([50, 0, 0]) / F_ch, np.array([100, 0, 0]) / F_ch
C0_lbs, C0_ubs = np.array([0.05, 0.0, 0.0]) / C_ch, np.array([0.10, 0.0, 0.0]) / C_ch
T0_lb, T0_ub = 350.0 / T_ch, 450.0 / T_ch
Ta0_lb, Ta0_ub = 300.0 / T_ch, 350.0 / T_ch
V_ub = 1.0

N_inlets, N_v = 400, 101


def I_S_samples(seed=42):

    lower = np.array([F0_lbs[0], C0_lbs[0], T0_lb, Ta0_lb])
    upper = np.array([F0_ubs[0], C0_ubs[0], T0_ub, Ta0_ub])

    vary = np.where(upper > lower)[0]
    d_vary = len(vary)

    if d_vary > 0:
        engine = qmc.LatinHypercube(d=d_vary, seed=seed)  # type: ignore
        raw = engine.random(N_inlets)
        lb_v = lower[vary]
        ub_v = upper[vary]
        scaled_v = qmc.scale(raw, lb_v, ub_v)
    else:
        scaled_v = np.empty((N_inlets, 0))

    I_S = np.tile(lower, (N_inlets, 1))

    if d_vary > 0:
        I_S[:, vary] = scaled_v

    I_S_cols = ["F0_A", "C0_A", "T0", "Ta0"]

    I_S = pd.DataFrame(I_S, columns=I_S_cols)

    return I_S


def D_S_samples(I_S_samples: pd.DataFrame):

    I_S_rows = []
    D_S_rows = []

    for _, sample in tqdm(
        I_S_samples.iterrows(),
        total=len(I_S_samples),
        desc="Simulating trajectories",
    ):

        F0_vals = {"A": sample["F0_A"], "B": 0.0, "C": 0.0}
        C0_vals = {"A": sample["C0_A"], "B": 0.0, "C": 0.0}

        T0 = sample["T0"]
        Ta0 = sample["Ta0"]

        model = NonisoPFR(
            reactions=reactions,
            species=species,
            C0=C0_vals,
            F0=F0_vals,
            T0=T0,
            Ua=Ua,
            Ta=Ta0,
            V_span=(0.0, V_ub),
            Cp=Cp,
            delta_Hrxn=delta_Hrxn,
            V_ch=V_ch,
            F_ch=F_ch,
            T_ch=T_ch,
            Cp_ch=Cp_ch,
            H_ch=H_ch,
        )

        t_eval = np.linspace(0.0, V_ub, N_v)

        try:
            sol = model.simulate(t_eval=t_eval)
        except TypeError:
            sol = model.simulate()

        for V_i, Y_i in zip(sol.t, sol.y.T):
            I_row = {
                "F0_A": float(F0_vals["A"]),
                "C0_A": float(C0_vals["A"]),
                "T0": float(T0),
                "Ta0": float(Ta0),
                "V": float(V_i),
            }
            D_row = {}

            for idx, sp in enumerate(species):
                D_row[f"F_{sp}"] = float(Y_i[idx])

            C_dict = model.concentrations(Y_i[:N_species], Y_i[N_species])

            for sp, Ci in C_dict.items():
                D_row[f"C_{sp}"] = float(Ci)

            D_row["T"] = float(Y_i[N_species])
            D_row["Ta"] = float(Y_i[N_species + 1])

            I_S_rows.append(I_row)
            D_S_rows.append(D_row)

    I_S = pd.DataFrame(I_S_rows)
    D_S = pd.DataFrame(D_S_rows)

    return I_S, D_S


if __name__ == "__main__":

    I_S_inlet = I_S_samples()

    I_S, D_S = D_S_samples(I_S_inlet)

    I_S.to_excel("I_S_data.xlsx", index=False)
    D_S.to_excel("D_S_data.xlsx", index=False)
