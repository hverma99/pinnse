import os

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from pinnse import ANN, Training


def _make_loader(n=64, dim_in=2, dim_out=1, batch=16, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, size=(n, dim_in)).astype(np.float32)
    Y = X.sum(axis=1, keepdims=True).astype(np.float32)
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
    return DataLoader(ds, batch_size=batch, shuffle=True)


def _make_training(ckpt_path):
    model = ANN([2, 8, 8, 1], activation=nn.Tanh)
    loader = _make_loader()
    return Training(
        model=model,
        train_loader=loader,
        val_loader=_make_loader(seed=1),
        test_loader=_make_loader(seed=2),
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-2),
        loss_fn=nn.MSELoss(),
        device=torch.device("cpu"),
        ckpt_path=ckpt_path,
    )


def test_ckpt_dir_autocreated(tmp_path):
    """Training must create the checkpoint directory if it doesn't exist."""
    nested = tmp_path / "does" / "not" / "exist" / "best.pth"
    _make_training(str(nested))
    assert nested.parent.is_dir()


def test_adam_step_writes_checkpoint_even_without_validation(tmp_path):
    """When val_every > epochs, no validation occurs and previously the load
    at the end of adam_step would fail (or load a stale checkpoint). A final
    checkpoint must always be written so evaluation succeeds."""
    ckpt = tmp_path / "best.pth"
    trainer = _make_training(str(ckpt))
    # val_every larger than epochs -> no validation iteration will run
    history = trainer.adam_step(epochs=2, val_every=100, verbose=False)
    assert ckpt.is_file()
    # No validation happened
    assert history["val_loss_t"] == []
    # But test evaluation must have completed without raising
    assert history["loss_t"]  # training losses recorded


def test_adam_step_normal_flow_writes_best(tmp_path):
    """When validation does run, the file exists and can be reloaded."""
    ckpt = tmp_path / "best.pth"
    trainer = _make_training(str(ckpt))
    trainer.adam_step(epochs=4, val_every=2, verbose=False)
    assert ckpt.is_file()
    loaded = torch.load(ckpt, map_location="cpu")
    assert "model_state_dict" in loaded
