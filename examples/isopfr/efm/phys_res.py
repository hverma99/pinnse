"""
Physics and boundary residuals of this case: the shared effluent-flow (EFM)
residuals of ../common/residuals.py. Redefine Physics or Boundary here to
change them for this case only.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # examples/isopfr

from common.residuals import EFMBoundary as Boundary
from common.residuals import EFMPhysics as Physics

__all__ = ["Boundary", "Physics"]
