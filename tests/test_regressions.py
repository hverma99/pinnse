"""
Regression tests for the bugs fixed in pinnse 1.0, one test per bug ID of the
1.0 audit. Each test fails on pinnse 0.2.1 and passes once its bug is fixed.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import pinnse.train
from pinnse import ANN, Training


def _loader(X, Y=None, batch_size=8):
    ds = TensorDataset(X) if Y is None else TensorDataset(X, Y)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


def _labeled(n):
    X = torch.linspace(-1, 1, n).unsqueeze(1)
    return _loader(X, X**2)


def _phys(x, y):
    dy = torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]
    return dy - 2.0 * x


def _trainer(tmp_path, **kw):
    torch.manual_seed(0)
    model = ANN([1, 8, 1], activation=nn.Tanh)
    args = dict(
        model=model,
        train_loader=_labeled(32),
        val_loader=_labeled(16),
        test_loader=_labeled(8),
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-2),
        loss_fn=nn.MSELoss(),
        device=torch.device("cpu"),
        ckpt_path=str(tmp_path / "best.pth"),
    )
    args.update(kw)
    return Training(**args)


# ----------------------------------- B13 -------------------------------------
def test_B13_train_module_docstring():
    assert pinnse.train.__doc__ and "Training module" in pinnse.train.__doc__


# ----------------------------------- B14 -------------------------------------
def test_B14_checkpoints_are_loaded_weights_only(tmp_path, monkeypatch):
    t = _trainer(tmp_path)
    t._save_checkpoint()
    seen = []
    real_load = torch.load

    def spy(*args, **kwargs):
        seen.append(kwargs.get("weights_only"))
        return real_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", spy)
    t._load_checkpoint()
    assert seen == [True]
