"""
Regression tests for the bugs fixed in pinnse 1.0, one test per bug ID of the
1.0 audit. Each test fails on pinnse 0.2.1 and passes once its bug is fixed.
"""

import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import pytest

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


# ----------------------------------- B01 -------------------------------------
def test_B01_checkpoint_selection_not_skewed_by_adapted_weights(tmp_path):
    # The physics residual is the same for every model, so it cannot change
    # which model is best; only its weight grows. The restored checkpoint must
    # therefore be the epoch with the lowest validation data loss.
    t = _trainer(
        tmp_path,
        phys_coll_loader=_loader(torch.rand(16, 1)),
        phys_residual=lambda x, y: 0.0 * y + 1.0,
        phys_weight=1.0,
        adapt_wts=True,
        weight_scheme=lambda gd, gp, gb: (100.0, None),
    )
    t.wt_update_every = 1
    h = t.adam_step(epochs=60, val_every=5, verbose=False)

    assert h["wt_phys"][-1] > 50.0  # the weight did drift
    restored = t.validate_test(t.val_loader)[0]
    assert restored == pytest.approx(min(h["val_loss_d"]), rel=1e-6)


# ----------------------------------- B02 -------------------------------------
def test_B02_validation_sampling_reproducible_under_torch_seed(tmp_path):
    def run(python_seed):
        random.seed(python_seed)  # state of Python's global generator
        torch.manual_seed(0)
        coll = _loader(torch.rand(128, 1) * 2 - 1, batch_size=8)
        t = _trainer(
            tmp_path,
            phys_coll_loader=coll,
            phys_residual=_phys,
            phys_weight=1.0,
        )
        return t.adam_step(epochs=4, val_every=1, verbose=False)["val_loss_p"]

    assert run(1) == run(2)


# ----------------------------------- B07 -------------------------------------
def test_B07_adam_step_rejects_bad_epochs_before_training(tmp_path, monkeypatch):
    t = _trainer(tmp_path)
    monkeypatch.setattr(t, "train_epoch", lambda: pytest.fail("trained anyway"))
    with pytest.raises(ValueError, match="exceeds epochs"):
        t.adam_step(epochs=3, val_every=5, verbose=False)
    with pytest.raises(ValueError, match="val_every must be a positive integer"):
        t.adam_step(epochs=3, val_every=0, verbose=False)


# ----------------------------------- B08 -------------------------------------
def test_B08_lbfgs_rejects_mode_without_active_term(tmp_path):
    t = _trainer(tmp_path)  # no physics residual, no collocation data
    t._save_checkpoint()
    before = [p.detach().clone() for p in t.model.parameters()]
    with pytest.raises(ValueError, match="no active loss term"):
        t.lbfgs_step(mode="physics", N_LBFGS=2, verbose=False)
    for a, b in zip(before, t.model.parameters()):
        assert torch.equal(a, b)


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
