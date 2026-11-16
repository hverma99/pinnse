"""
Names deprecated in pinnse 1.0 keep working until 2.0, with a warning that
names their replacement.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import pandas as pd
import pytest

from pinnse import (
    ANN,
    DataModule,
    Denormalization,
    Fourier_ANN,
    FourierANN,
    Normalization,
    Training,
)


def test_fourier_ann_alias(recwarn):
    torch.manual_seed(0)
    with pytest.warns(DeprecationWarning, match="use FourierANN instead"):
        old = Fourier_ANN([2, 8, 1], fourier_levels=2)
    torch.manual_seed(0)
    new = FourierANN([2, 8, 1], fourier_levels=2)
    assert isinstance(old, FourierANN)
    x = torch.randn(4, 2)
    torch.testing.assert_close(old(x), new(x))


@pytest.mark.parametrize(
    "old, new, normalize",
    [("mean_abs", "mean_norm", "mean_norm"), ("z_norm", "z_score", "z_score")],
)
def test_denormalization_aliases(old, new, normalize):
    frame = pd.DataFrame({"a": [1.0, 2.0, 4.0], "b": [-3.0, 0.0, 6.0]})
    norm, metrics = getattr(Normalization, normalize)(frame)
    with pytest.warns(DeprecationWarning, match=f"use Denormalization.{new} instead"):
        restored = getattr(Denormalization, old)(norm, metrics)
    pd.testing.assert_frame_equal(
        restored, getattr(Denormalization, new)(norm, metrics)
    )


def test_boundry_coll_batch_size_keyword_and_attributes(tabular_data):
    I_S, D_S = tabular_data
    with pytest.warns(DeprecationWarning, match="use boundary_coll_batch_size"):
        dm = DataModule(
            I_S, D_S, 8, boundary_coll_data_size=16, boundry_coll_batch_size=4
        )
    assert dm.boundary_coll_batch_size == 4
    assert len(dm.bnd_colloc_loader()) == 4

    with pytest.warns(DeprecationWarning):
        assert dm.boundry_coll_batch_size == 4
    with pytest.warns(DeprecationWarning):
        dm.boundry_coll_batch_size = 8
    assert dm.boundary_coll_batch_size == 8

    with pytest.warns(DeprecationWarning):
        assert dm.boundry_coll_data_size == 16
    with pytest.warns(DeprecationWarning):
        dm.boundry_coll_data_size = 32
    assert dm.boundary_coll_data_size == 32

    with pytest.raises(TypeError, match="boundary_coll_batch_size only"):
        DataModule(I_S, D_S, 8, boundary_coll_batch_size=4, boundry_coll_batch_size=4)


def _trainer(tmp_path):
    torch.manual_seed(0)
    X = torch.linspace(-1, 1, 16).unsqueeze(1)
    loader = DataLoader(TensorDataset(X, X**2), batch_size=8)
    model = ANN([1, 8, 1], nn.Tanh)
    t = Training(
        model,
        loader,
        loader,
        loader,
        torch.optim.Adam(model.parameters(), 1e-2),
        nn.MSELoss(),
        torch.device("cpu"),
        ckpt_path=str(tmp_path / "b.pth"),
    )
    t._save_checkpoint()
    return t


def test_lbfgs_n_lbfgs_keyword(tmp_path):
    t = _trainer(tmp_path)
    with pytest.warns(DeprecationWarning, match="use n_steps instead"):
        h = t.lbfgs_step(mode="data", N_LBFGS=2, verbose=False)
    assert len(h["loss_t"]) == 2
    with pytest.raises(TypeError, match="n_steps only"):
        t.lbfgs_step(mode="data", n_steps=2, N_LBFGS=2)


def test_lbfgs_step_count_default_and_validation(tmp_path):
    t = _trainer(tmp_path)
    assert len(t.lbfgs_step("data", verbose=False)["loss_t"]) == 50
    with pytest.raises(ValueError, match="n_steps must be a positive integer"):
        t.lbfgs_step("data", n_steps=0)
    assert np.isfinite(t.lbfgs_step("data", 1, verbose=False)["loss_t"][0])
