"""
Physics and boundary residuals of the inverse case: the shared effluent-flow
(EFM) residuals of ../common/residuals.py, with the activation energies taken
from the trainable parameter theta (in kJ/mol) instead of fixed values.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # examples/isopfr

from common.residuals import EFMBoundary as Boundary
from common.residuals import EFMPhysics

__all__ = ["Boundary", "Physics"]


class Physics(EFMPhysics):
    def __init__(
        self,
        I_S_metrics: dict,
        D_S_metrics: dict,
        species: list[str],
        nu,
        key_species: list[str],
        k,
        theta,
        R: float = 8.314,
    ):
        super().__init__(I_S_metrics, D_S_metrics, species, nu, key_species, k, None, R)
        self.theta = theta

    def activation_energies(self, device, dtype):
        # inverse parameter estimation - trainable parameter
        return self.theta.to(device=device, dtype=dtype) * 1e3  # (nr,)
