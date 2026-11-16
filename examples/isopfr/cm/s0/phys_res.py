"""
Physics and boundary residuals of this case: the shared conversion-based (CM)
residuals of ../../common/residuals.py. Redefine Physics or Boundary here to
change them for this case only.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # examples/isopfr

from common.residuals import CMBoundary as Boundary
from common.residuals import CMPhysics as Physics

__all__ = ["Boundary", "Physics"]
