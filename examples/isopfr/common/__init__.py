"""
Code shared by the isothermal PFR example cases.

Each case folder (efm, erm, cm/s0, cm/s1, efm.inverse, efm.noise) keeps its
own settings in main.py and check.py and its residuals in phys_res.py, and
imports the rest from here:

- process.py   : reaction network, kinetics and operating ranges
- pfr_model.py : first-principles reactor models (EFM, ERM, CM)
- data_gen.py  : generation of the labeled datasets
- residuals.py : physics and boundary residuals of each formulation
- train.py     : training pipeline used by main.py
- evaluate.py  : evaluation and plots used by check.py

The case scripts put examples/isopfr on sys.path, so they still run from
inside their folder (python main.py, python check.py, sbatch run.sh).
"""

# On Windows/Anaconda, torch must load before NumPy/SciPy (MKL), or importing
# torch fails (see pyproject.toml). Importing it here makes every
# `from common... import` load torch first.
import torch  # noqa: F401
