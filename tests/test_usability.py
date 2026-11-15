"""
Tests of the usability features of pinnse 1.0: input checks, residual
verification, logging and progress control, resumable training, checkpoint
metadata and loading, and the extra history entries.
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
    SequenceDataModule,
    Training,
)


# --------------------------------------------------------------------- helpers
def loader(X, Y=None, batch_size=8, generator=None):
    ds = TensorDataset(X) if Y is None else TensorDataset(X, Y)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=generator is not None, generator=generator
    )


def labeled(n, **kw):
    X = torch.linspace(-1, 1, n).unsqueeze(1)
    return loader(X, X**2, **kw)


def phys_fn(x, y):
    dy = torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]
    return dy - 2.0 * x


def bnd_fn(x, y):
    return y - x**2


def make(tmp_path, model=None, theta=None, **kw):
    torch.manual_seed(0)
    model = model or ANN([1, 8, 1], nn.Tanh)
    params = list(model.parameters()) + ([theta] if theta is not None else [])
    args = dict(
        model=model,
        train_loader=labeled(32),
        val_loader=labeled(16),
        test_loader=labeled(8),
        optimizer=torch.optim.Adam(params, lr=1e-2),
        loss_fn=nn.MSELoss(),
        device=torch.device("cpu"),
        phys_coll_loader=loader(torch.rand(32, 1) * 2 - 1),
        bnd_coll_loader=loader(torch.tensor([[-1.0], [1.0]] * 4), batch_size=4),
        phys_residual=phys_fn,
        bnd_residual=bnd_fn,
        ckpt_path=str(tmp_path / "best.pth"),
        phys_weight=1.0,
        bnd_weight=1.0,
        theta=theta,
    )
    args.update(kw)
    return Training(**args)


# ---------------------------------------------------------- data-module checks
def test_data_module_rejects_inconsistent_or_invalid_data():
    I_S = pd.DataFrame({"x": [0.0, 1.0, 2.0]})
    D_S = pd.DataFrame({"y": [0.0, 1.0, 2.0]})
    with pytest.raises(ValueError, match="same number of rows"):
        DataModule(I_S, D_S.iloc[:2], 2)
    with pytest.raises(ValueError, match="non-numeric columns \\['name'\\]"):
        DataModule(I_S.assign(name=["a", "b", "c"]), D_S, 2)
    with pytest.raises(ValueError, match="D_S_data has missing or non-finite values"):
        DataModule(I_S, D_S.assign(y=[0.0, np.nan, 1.0]), 2)


def test_sequence_module_rejects_invalid_data():
    X = np.zeros((4, 3, 2))
    with pytest.raises(ValueError, match="must be numeric"):
        SequenceDataModule(X.astype(str), X, 2)
    bad = X.copy()
    bad[0, 0, 0] = np.inf
    with pytest.raises(ValueError, match="D_S_data has missing or non-finite"):
        SequenceDataModule(X, bad, 2)
