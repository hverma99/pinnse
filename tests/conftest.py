import os
import random

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# On Windows/Anaconda, torch must load before NumPy (imported by matplotlib),
# or importing torch fails with "Error loading ...\torch\lib\shm.dll".
import torch

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def _seed_and_cleanup():
    """Make every test deterministic and release figures it opens."""
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    yield
    plt.close("all")


@pytest.fixture
def tabular_data():
    """
    A small steady-state dataset in the layout used by `DataModule`: ordinary
    bounded inputs, a mole-fraction group (prefix `Z_`) that sums to one, and
    the boundary variable `V` in the last column.
    """
    rng = np.random.default_rng(0)
    n = 100
    z = rng.dirichlet([1.0, 1.0], size=n)
    I_S = pd.DataFrame(
        {
            "P": rng.uniform(1.0, 5.0, n),
            "T": rng.uniform(300.0, 400.0, n),
            "Z_A": z[:, 0],
            "Z_B": z[:, 1],
            "V": rng.uniform(-1.0, 1.0, n),
        }
    )
    D_S = pd.DataFrame(
        {"F_A": I_S["P"] * I_S["Z_A"], "F_B": I_S["T"] * I_S["Z_B"] / 100.0}
    )
    return I_S, D_S
