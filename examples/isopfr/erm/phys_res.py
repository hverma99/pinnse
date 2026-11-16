"""
Physics and boundary residuals of this case: the shared extent of reaction-based (ERM)
residuals of ../common/residuals.py. Redefine Physics or Boundary here to
change them for this case only.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # examples/isopfr

from common.residuals import ERMBoundary as Boundary
from common.residuals import ERMPhysics as Physics

__all__ = ["Boundary", "Physics"]
