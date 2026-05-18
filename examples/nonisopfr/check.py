from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from pfr_model import NonisoPFR, Reaction
from pinnse import SANN
from phys_res import Physics

# -----------------------------------------------------------------------------
# System specifications: keep these consistent with data_gen.py and main.py
# -----------------------------------------------------------------------------
SPECIES: list[str] = ["A", "B", "C"]
N_SPECIES: int = len(SPECIES)

# Dimensional process constants
R: float = 8.314  # J/(mol-K)
T_R_DIM: float = 300.0  # K
UA: float = 4000.0  # J/(m3-s-K)

CP_DIM: dict[str, float] = {"A": 90.0, "B": 90.0, "C": 180.0}  # J/(mol-K)
DELTA_HRXN_DIM: dict[int, float] = {
    0: -20000.0,
    1: -60000.0,
}  # J/(mol A reacted)

# Characteristic scales
F_CH: float = 200.0  # mol/s
C_CH: float = 0.1  # mol/L
T_CH: float = 800.0  # K
V_CH: float = 1.0  # L, consistent with data_gen.py
H_CH: float = -100000.0  # J/(mol A reacted)
CP_CH: float = 200.0  # J/(mol-K)


class Check:
    """Compare a trained PINN checkpoint against the non-isothermal PFR model."""

    def __init__(
        self,
        model: torch.nn.Module,
        checkpoints: Iterable[str],
        F0_dim: list[float],
        C0_dim: list[float],
        T0_dim: float,
        Ta0_dim: float,
        V_max_dim: float,
        n_grid_points: int = 101,
        root_dir: str | Path = ".",
        output_dir: str | Path = "plots",
    ) -> None:
        self.model = model
        self.checkpoints = list(checkpoints)
        self.F0_dim = F0_dim
        self.C0_dim = C0_dim
        self.T0_dim = T0_dim
        self.Ta0_dim = Ta0_dim
        self.V_max_dim = V_max_dim
        self.n_grid_points = n_grid_points

        self.root_dir = Path(root_dir).resolve()
        self.output_dir = self.root_dir / output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.physics = Physics(
            species=SPECIES,
            delta_Hrxn=DELTA_HRXN_DIM,
            Cp=CP_DIM,
            Ua=UA,
            T_R=T_R_DIM,
            R=R,
            F_ch=F_CH,
            C_ch=C_CH,
            T_ch=T_CH,
            V_ch=V_CH,
            H_ch=H_CH,
            Cp_ch=CP_CH,
        )

        # Nondimensional inlet conditions used by the NN and ODE model
        self.F0_nd = {
            sp: val / F_CH for sp, val in zip(SPECIES, self.F0_dim, strict=True)
        }
        self.C0_nd = {
            sp: val / C_CH for sp, val in zip(SPECIES, self.C0_dim, strict=True)
        }
        self.T0_nd = self.T0_dim / T_CH
        self.Ta0_nd = self.Ta0_dim / T_CH
        self.V_arr_nd = np.linspace(0.0, self.V_max_dim / V_CH, self.n_grid_points)

    def build_reactions(self) -> list[Reaction]:
        """Create reaction objects with nondimensional reference temperature."""

        T_R_nd = T_R_DIM / T_CH

        r1 = Reaction(
            species=SPECIES,
            stoich={"A": -1, "B": 1},
            k0=10,
            E=4000 * R,
            T_R=T_R_nd,
            T_ch=T_CH,
            C_ch=C_CH,
        )

        r2 = Reaction(
            species=SPECIES,
            stoich={"A": -2, "C": 3},
            k0=0.5,
            E=9000 * R,
            T_R=T_R_nd,
            T_ch=T_CH,
            C_ch=C_CH,
        )

        return [r1, r2]

    def exact_solution(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate the first-principles non-isothermal PFR model.

        Returns
        -------
        F_exact_nd : np.ndarray
            Nondimensional flowrates with shape (n_grid_points, N_species).

        C_exact_nd : np.ndarray
            Nondimensional concentrations with shape (n_grid_points, N_species).

        T_exact_nd : np.ndarray
            Nondimensional reactor temperature with shape (n_grid_points,).

        Ta_exact_nd : np.ndarray
            Nondimensional coolant temperature with shape (n_grid_points,).
        """

        Cp_nd = {sp: val / CP_CH for sp, val in CP_DIM.items()}
        delta_Hrxn_nd = {idx: val / H_CH for idx, val in DELTA_HRXN_DIM.items()}

        sim_model = NonisoPFR(
            reactions=self.build_reactions(),
            species=SPECIES,
            C0=self.C0_nd,
            F0=self.F0_nd,
            T0=self.T0_nd,
            Ua=UA,
            Ta=self.Ta0_nd,
            V_span=(0.0, self.V_max_dim / V_CH),
            Cp=Cp_nd,
            delta_Hrxn=delta_Hrxn_nd,
            V_ch=V_CH,
            F_ch=F_CH,
            T_ch=T_CH,
            Cp_ch=CP_CH,
            H_ch=H_CH,
        )

        sol = sim_model.simulate(t_eval=self.V_arr_nd)

        if not sol.success:
            raise RuntimeError(f"ODE simulation failed: {sol.message}")

        F_exact_nd = sol.y[:N_SPECIES, :].T
        T_exact_nd = sol.y[N_SPECIES, :]
        Ta_exact_nd = sol.y[N_SPECIES + 1, :]

        C_exact_list = []
        for F_i, T_i in zip(F_exact_nd, T_exact_nd, strict=True):
            C_dict = sim_model.concentrations(F_i, T_i)
            C_exact_list.append([C_dict[sp] for sp in SPECIES])

        C_exact_nd = np.asarray(C_exact_list)

        return F_exact_nd, C_exact_nd, T_exact_nd, Ta_exact_nd

    def checkpoint_path(self, checkpoint: str) -> Path:
        """Resolve a checkpoint name or path."""

        path = Path(checkpoint)

        if path.is_absolute() and path.exists():
            return path

        candidates = [
            self.root_dir / path,
            self.root_dir / "logs" / path,
        ]

        for candidate in candidates:
            if candidate.exists():
                return candidate

        raise FileNotFoundError(
            f"Could not find checkpoint '{checkpoint}'. Checked: "
            + ", ".join(str(c) for c in candidates)
        )

    def load_checkpoint(self, checkpoint: str) -> None:
        """Load model weights from a checkpoint file."""

        ckpt_path = self.checkpoint_path(checkpoint)
        ckpt = torch.load(ckpt_path, map_location=self.device)

        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif isinstance(ckpt, dict) and "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt

        self.model.load_state_dict(state_dict)
        self.model.eval()

    def pinn_solution(
        self,
        checkpoint: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluate the PINN and its physics residual on the current V-grid.

        Returns
        -------
        F_pinn_nd, C_pinn_nd, T_pinn_nd, Ta_pinn_nd, phys_res
            Nondimensional PINN outputs and physics residuals.
        """

        self.load_checkpoint(checkpoint)

        n_pts = self.V_arr_nd.shape[0]

        F0_A = np.full((n_pts, 1), self.F0_nd["A"], dtype=np.float32)
        C0_A = np.full((n_pts, 1), self.C0_nd["A"], dtype=np.float32)
        T0 = np.full((n_pts, 1), self.T0_nd, dtype=np.float32)
        Ta0 = np.full((n_pts, 1), self.Ta0_nd, dtype=np.float32)
        V = self.V_arr_nd.astype(np.float32).reshape(-1, 1)

        # Current reduced input structure:
        # x = [F0_A, C0_A, T0, Ta0, V]
        X_np = np.hstack([F0_A, C0_A, T0, Ta0, V]).astype(np.float32)
        X = torch.from_numpy(X_np).to(self.device)
        X.requires_grad_(True)

        Y_pinn = self.model(X)
        phys_res = self.physics(X, Y_pinn)

        F_pinn_nd = Y_pinn[:, :N_SPECIES].detach().cpu().numpy()
        C_pinn_nd = Y_pinn[:, N_SPECIES : 2 * N_SPECIES].detach().cpu().numpy()
        T_pinn_nd = Y_pinn[:, 2 * N_SPECIES].detach().cpu().numpy()
        Ta_pinn_nd = Y_pinn[:, 2 * N_SPECIES + 1].detach().cpu().numpy()
        phys_res_np = phys_res.detach().cpu().numpy()

        return F_pinn_nd, C_pinn_nd, T_pinn_nd, Ta_pinn_nd, phys_res_np

    @staticmethod
    def metrics(
        exact: np.ndarray,
        pred: np.ndarray,
        prefix: str,
        columns: list[str],
    ) -> dict[str, float]:
        """Compute MAE and RMSE for a group of variables."""

        out: dict[str, float] = {}
        err = pred - exact

        for i, col in enumerate(columns):
            out[f"{prefix}_{col}_MAE"] = float(np.mean(np.abs(err[:, i])))
            out[f"{prefix}_{col}_RMSE"] = float(np.sqrt(np.mean(err[:, i] ** 2)))

        out[f"{prefix}_Overall_MAE"] = float(np.mean(np.abs(err)))
        out[f"{prefix}_Overall_RMSE"] = float(np.sqrt(np.mean(err**2)))

        return out

    def analyze(self, checkpoint: str) -> dict[str, float]:
        """Analyze one checkpoint and write Excel/PNG outputs."""

        F_exact_nd, C_exact_nd, T_exact_nd, Ta_exact_nd = self.exact_solution()
        F_pinn_nd, C_pinn_nd, T_pinn_nd, Ta_pinn_nd, phys_res = self.pinn_solution(
            checkpoint
        )

        # Dimensionalize for plotting/reporting
        F_exact = F_exact_nd * F_CH
        F_pinn = F_pinn_nd * F_CH
        C_exact = C_exact_nd * C_CH
        C_pinn = C_pinn_nd * C_CH
        T_exact = T_exact_nd * T_CH
        T_pinn = T_pinn_nd * T_CH
        Ta_exact = Ta_exact_nd * T_CH
        Ta_pinn = Ta_pinn_nd * T_CH
        V_arr = self.V_arr_nd * V_CH

        ckpt_stem = Path(checkpoint).stem

        df_F = pd.DataFrame(
            {
                "V": V_arr,
                **{f"F_exact_{sp}": F_exact[:, i] for i, sp in enumerate(SPECIES)},
                **{f"F_pinn_{sp}": F_pinn[:, i] for i, sp in enumerate(SPECIES)},
            }
        )

        df_C = pd.DataFrame(
            {
                "V": V_arr,
                **{f"C_exact_{sp}": C_exact[:, i] for i, sp in enumerate(SPECIES)},
                **{f"C_pinn_{sp}": C_pinn[:, i] for i, sp in enumerate(SPECIES)},
            }
        )

        df_T = pd.DataFrame(
            {
                "V": V_arr,
                "T_exact": T_exact,
                "T_pinn": T_pinn,
                "Ta_exact": Ta_exact,
                "Ta_pinn": Ta_pinn,
            }
        )

        df_phys_res = pd.DataFrame(
            phys_res,
            columns=[f"res_mb_{sp}" for sp in SPECIES]
            + [f"res_c_{sp}" for sp in SPECIES]
            + ["res_eb", "res_ceb"],
        )
        df_phys_res.insert(0, "V", V_arr)

        summary = {"checkpoint": checkpoint}
        summary.update(self.metrics(F_exact, F_pinn, "F", SPECIES))
        summary.update(self.metrics(C_exact, C_pinn, "C", SPECIES))
        summary.update(
            self.metrics(
                T_exact.reshape(-1, 1),
                T_pinn.reshape(-1, 1),
                "T",
                ["T"],
            )
        )
        summary.update(
            self.metrics(
                Ta_exact.reshape(-1, 1),
                Ta_pinn.reshape(-1, 1),
                "Ta",
                ["Ta"],
            )
        )
        summary["phys_res_Overall_MAE"] = float(np.mean(np.abs(phys_res)))
        summary["phys_res_Overall_RMSE"] = float(np.sqrt(np.mean(phys_res**2)))

        excel_path = self.output_dir / f"check_{ckpt_stem}.xlsx"
        with pd.ExcelWriter(excel_path) as writer:
            pd.DataFrame([summary]).to_excel(writer, sheet_name="summary", index=False)
            df_F.to_excel(writer, sheet_name="F", index=False)
            df_C.to_excel(writer, sheet_name="C", index=False)
            df_T.to_excel(writer, sheet_name="T_Ta", index=False)
            df_phys_res.to_excel(writer, sheet_name="phys_res", index=False)

        self.plot_flows(V_arr, F_exact, F_pinn, ckpt_stem)
        self.plot_concentrations(V_arr, C_exact, C_pinn, ckpt_stem)
        self.plot_temperatures(V_arr, T_exact, T_pinn, Ta_exact, Ta_pinn, ckpt_stem)

        print(f"Finished {checkpoint}")
        print(f"  Excel: {excel_path}")

        return summary

    def plot_flows(
        self,
        V_arr: np.ndarray,
        F_exact: np.ndarray,
        F_pinn: np.ndarray,
        ckpt_stem: str,
    ) -> None:
        """Plot exact and PINN molar flowrates."""

        plt.figure(figsize=(8, 6), dpi=300)
        cmap = plt.get_cmap("tab10")

        for i, sp in enumerate(SPECIES):
            color = cmap(i)
            plt.plot(V_arr, F_exact[:, i], "--", color=color, label=f"$F_{sp}$ Exact")
            plt.plot(V_arr, F_pinn[:, i], "-", color=color, label=f"$F_{sp}$ PINN")

        plt.xlabel(r"$V\;(\mathrm{L})$")
        plt.ylabel(r"Flowrate $\left(\mathrm{mol}\,\mathrm{s}^{-1}\right)$")
        plt.title("Flowrates: Exact vs PINN")
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / f"flows_{ckpt_stem}.png")
        plt.close()

    def plot_concentrations(
        self,
        V_arr: np.ndarray,
        C_exact: np.ndarray,
        C_pinn: np.ndarray,
        ckpt_stem: str,
    ) -> None:
        """Plot exact and PINN concentrations."""

        plt.figure(figsize=(8, 6), dpi=300)
        cmap = plt.get_cmap("tab10")

        for i, sp in enumerate(SPECIES):
            color = cmap(i)
            plt.plot(V_arr, C_exact[:, i], "--", color=color, label=f"$C_{sp}$ Exact")
            plt.plot(V_arr, C_pinn[:, i], "-", color=color, label=f"$C_{sp}$ PINN")

        plt.xlabel(r"$V\;(\mathrm{L})$")
        plt.ylabel(r"Concentration $\left(\mathrm{mol}\,\mathrm{L}^{-1}\right)$")
        plt.title("Concentrations: Exact vs PINN")
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / f"concentrations_{ckpt_stem}.png")
        plt.close()

    def plot_temperatures(
        self,
        V_arr: np.ndarray,
        T_exact: np.ndarray,
        T_pinn: np.ndarray,
        Ta_exact: np.ndarray,
        Ta_pinn: np.ndarray,
        ckpt_stem: str,
    ) -> None:
        """Plot exact and PINN reactor/coolant temperatures."""

        plt.figure(figsize=(8, 6), dpi=300)
        cmap = plt.get_cmap("tab10")
        c_t = cmap(N_SPECIES)
        c_ta = cmap(N_SPECIES + 1)

        plt.plot(V_arr, T_exact, "--", color=c_t, label="$T$ Exact")
        plt.plot(V_arr, T_pinn, "-", color=c_t, label="$T$ PINN")
        plt.plot(V_arr, Ta_exact, "--", color=c_ta, label="$T_a$ Exact")
        plt.plot(V_arr, Ta_pinn, "-", color=c_ta, label="$T_a$ PINN")

        plt.xlabel(r"$V\;(\mathrm{L})$")
        plt.ylabel(r"Temperature $\left(\mathrm{K}\right)$")
        plt.title("Temperatures: Exact vs PINN")
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / f"temperatures_{ckpt_stem}.png")
        plt.close()

    def run(self) -> pd.DataFrame:
        """Analyze all available checkpoints and write an overall summary."""

        summaries: list[dict[str, float]] = []

        for checkpoint in self.checkpoints:
            try:
                summaries.append(self.analyze(checkpoint))
            except FileNotFoundError as exc:
                print(f"Skipping missing checkpoint: {exc}")

        if not summaries:
            raise FileNotFoundError("No requested checkpoints were found.")

        summary_df = pd.DataFrame(summaries)
        summary_path = self.output_dir / "check_summary.xlsx"
        summary_df.to_excel(summary_path, index=False)
        print(f"Summary written to: {summary_path}")

        return summary_df


def infer_dimensions(root_dir: Path) -> tuple[int, int]:
    """Infer input/output dimensions from Excel data files when available."""

    i_s_path = root_dir / "I_S_data.xlsx"
    d_s_path = root_dir / "D_S_data.xlsx"

    if i_s_path.exists() and d_s_path.exists():
        dim_in = pd.read_excel(i_s_path, nrows=1).shape[1]
        dim_out = pd.read_excel(d_s_path, nrows=1).shape[1]
        return dim_in, dim_out

    return 5, 8


if __name__ == "__main__":
    ROOT_DIR = Path(__file__).resolve().parent

    MODEL_CHECKPOINTS = [
        "best_model.pth",
        "combined_lbfgs_model.pth",
        "data_lbfgs_model.pth",
        "physics_lbfgs_model.pth",
    ]

    F0_DIMENSIONAL = [100.0, 0.0, 0.0]  # mol/s
    C0_DIMENSIONAL = [0.1, 0.0, 0.0]  # mol/L
    T0_DIMENSIONAL = 273.15 + 125.0  # K
    Ta0_DIMENSIONAL = 273.15 + 50.0  # K
    V_MAX_DIMENSIONAL = 1.0  # L
    N_GRID_POINTS = 101

    depth, width = 6, 64
    dim_in, dim_out = infer_dimensions(ROOT_DIR)
    layer_size = [dim_in] + [width] * depth + [dim_out]

    model = SANN(layer_size, activation=nn.Tanh)

    checker = Check(
        model=model,
        checkpoints=MODEL_CHECKPOINTS,
        F0_dim=F0_DIMENSIONAL,
        C0_dim=C0_DIMENSIONAL,
        T0_dim=T0_DIMENSIONAL,
        Ta0_dim=Ta0_DIMENSIONAL,
        V_max_dim=V_MAX_DIMENSIONAL,
        n_grid_points=N_GRID_POINTS,
        root_dir=ROOT_DIR,
        output_dir="plots",
    )

    checker.run()
