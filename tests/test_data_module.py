import numpy as np
import pandas as pd
import pytest

from pinnse import DataModule


@pytest.fixture
def module():
    rng = np.random.default_rng(0)
    n = 200
    I_S = pd.DataFrame(
        {
            "T": rng.uniform(300.0, 500.0, size=n),
            "P": rng.uniform(1.0, 5.0, size=n),
            "V": rng.uniform(0.0, 1.0, size=n),
        }
    )
    D_S = pd.DataFrame(
        {
            "y1": rng.normal(size=n),
            "y2": rng.normal(size=n),
        }
    )
    return DataModule(
        I_S_data=I_S,
        D_S_data=D_S,
        labeled_data_batch_size=16,
        physics_coll_data_size=64,
        physics_coll_batch_size=16,
        boundary_coll_data_size=32,
        boundry_coll_batch_size=16,
        test_frac=0.1,
        val_frac=0.1,
    )


def test_collocation_loader_before_labeled_loader(module):
    """phys_colloc_loader and bnd_colloc_loader must not depend on labeled_data_loader
    having been called first (they used to access self.lower_bnd set inside that method)."""
    phys = module.phys_colloc_loader()
    bnd = module.bnd_colloc_loader()

    phys_batch = next(iter(phys))[0]
    bnd_batch = next(iter(bnd))[0]

    assert phys_batch.shape[1] == 3
    assert bnd_batch.shape[1] == 3
    # bnd_colloc_loader fixes the last column to bnd_value (default -1.0)
    assert bnd_batch[:, -1].unique().tolist() == pytest.approx([-1.0])


def test_labeled_and_collocation_loaders_agree(module):
    """Calling labeled_data_loader() before/after collocation loaders must not
    change the bounds used for sampling."""
    lb_before = module.lower_bnd.copy()
    ub_before = module.upper_bnd.copy()
    module.labeled_data_loader()
    np.testing.assert_array_equal(module.lower_bnd, lb_before)
    np.testing.assert_array_equal(module.upper_bnd, ub_before)


def test_bnd_colloc_loader_respects_bounds(module):
    """Free columns of boundary-collocation samples must lie inside [lower_bnd, upper_bnd]."""
    bnd = module.bnd_colloc_loader()
    for (x,) in bnd:
        x_np = x.numpy()
        for j in range(x_np.shape[1] - 1):
            assert (x_np[:, j] >= module.lower_bnd[j] - 1e-6).all()
            assert (x_np[:, j] <= module.upper_bnd[j] + 1e-6).all()
