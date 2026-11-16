"""
Tests of the usability features of pinnse 1.0: input checks, residual
verification, logging and progress control, resumable training, checkpoint
metadata and loading, and the extra history entries.
"""

import io
import logging
import sys

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import pandas as pd
import pytest

import pinnse.train as train_module
from pinnse import (
    ANN,
    SANN,
    Analyze,
    DataModule,
    Plotter,
    Save,
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


# --------------------------------------------------------- constructor checks
@pytest.mark.parametrize(
    "kw, match",
    [
        ({"phys_weight": -1.0}, "phys_weight must be a finite, non-negative"),
        ({"bnd_weight": float("nan")}, "bnd_weight must be a finite, non-negative"),
        ({"wt_ema": 0.0}, "wt_ema must lie in"),
        ({"val_colloc_frac": 1.5}, "val_colloc_frac must lie in"),
        ({"wt_update_every": 0}, "wt_update_every must be a positive integer"),
        ({"grad_norm_every": 0}, "grad_norm_every must be a positive integer"),
        ({"metadata": {"a": np.zeros(2)}}, "metadata must be a dict of JSON-like"),
        ({"metadata": {1: "x"}}, "metadata must be a dict of JSON-like"),
    ],
)
def test_constructor_rejects_invalid_arguments(tmp_path, kw, match):
    with pytest.raises(ValueError, match=match):
        make(tmp_path, **kw)


@pytest.mark.parametrize(
    "kw, match",
    [
        ({"phys_residual": None}, "phys_weight=1.0 but no phys_residual"),
        ({"bnd_coll_loader": None}, "bnd_weight=1.0 but no bnd_coll_loader"),
    ],
)
def test_constructor_warns_when_a_weighted_loss_cannot_be_computed(tmp_path, kw, match):
    with pytest.warns(UserWarning, match=match):
        make(tmp_path, **kw)


def test_adaptive_weight_settings_are_constructor_arguments(tmp_path):
    t = make(tmp_path, adapt_wts=True, wt_update_every=1, wt_ema=0.5)
    h = t.adam_step(epochs=2, val_every=2, verbose=False)
    gd, gp = h["grad_data_hist"][0], h["grad_phys_hist"][0]
    assert h["wt_phys"][0] == pytest.approx(0.5 * 1.0 + 0.5 * gd / (gp + 1e-12))


# ------------------------------------------------------------ verify_residuals
@pytest.mark.parametrize(
    "residual, error, match",
    [
        (lambda x, y: 1.0, TypeError, "must return a torch.Tensor"),
        (lambda x, y: y.sum(), ValueError, "one row per collocation point"),
        (lambda x, y: y[:2], ValueError, "one row per collocation point"),
        (lambda x, y: y / 0.0 - y / 0.0, ValueError, "not finite"),
    ],
)
def test_verify_residuals_rejects_bad_residuals(tmp_path, residual, error, match):
    t = make(tmp_path, phys_residual=residual)
    with pytest.raises(error, match=match):
        t.adam_step(epochs=1, val_every=1, verbose=False)


def test_verify_residuals_warns_when_independent_of_the_output(tmp_path):
    t = make(tmp_path, bnd_residual=lambda x, y: x * 0 + 1)
    with pytest.warns(UserWarning, match="does not depend on the network output"):
        t.verify_residuals()
    t = make(tmp_path, bnd_residual=lambda x, y: torch.ones(x.shape[0], 1))
    with pytest.warns(UserWarning, match="does not depend on the network output"):
        t.verify_residuals()


def test_verify_residuals_leaves_model_mode_and_rng_unchanged(tmp_path):
    t = make(tmp_path)
    t.model.train()
    state = torch.get_rng_state()
    t.verify_residuals()
    assert t.model.training and torch.equal(state, torch.get_rng_state())
    t.model.eval()
    t.verify_residuals()
    assert not t.model.training


def test_residuals_are_verified_once_and_can_be_skipped(tmp_path, monkeypatch):
    t = make(tmp_path)
    calls = []
    monkeypatch.setattr(t, "verify_residuals", lambda: calls.append(1))
    t.adam_step(epochs=1, val_every=1, verbose=False)
    t._residuals_verified = True
    t.lbfgs_step("data", 1, verbose=False)
    assert calls == [1]

    t = make(tmp_path, check_residuals=False)
    monkeypatch.setattr(t, "verify_residuals", lambda: pytest.fail("checked"))
    t.adam_step(epochs=1, val_every=1, verbose=False)


# ------------------------------------------------------- logging and progress
def test_messages_are_logged_and_printed_only_when_verbose(tmp_path, caplog, capsys):
    t = make(tmp_path)
    with caplog.at_level(logging.INFO, logger="pinnse"):
        t.adam_step(epochs=2, val_every=1, verbose=False)
    assert "Epoch: 1 | Training | Total Loss" in caplog.text
    assert (
        caplog.text.count("| Validation |") == 2 and "Test | Total Loss" in caplog.text
    )
    assert capsys.readouterr().out == ""
    assert "\033[" not in caplog.text  # no terminal colour codes


@pytest.mark.parametrize(
    "progress, isatty, notebook, disabled",
    [
        (True, False, False, False),
        (False, True, False, True),
        (None, True, False, False),
        (None, False, False, True),
        (None, False, True, False),
    ],
)
def test_progress_bar_policy(monkeypatch, progress, isatty, notebook, disabled):
    stream = io.StringIO()
    stream.isatty = lambda: isatty
    monkeypatch.setattr(sys, "stderr", stream)
    if notebook:
        monkeypatch.setitem(sys.modules, "ipykernel", object())
    else:
        monkeypatch.delitem(sys.modules, "ipykernel", raising=False)
    assert train_module._progress_disabled(progress) is disabled


def test_progress_bar_disabled_without_isatty(monkeypatch):
    monkeypatch.setattr(sys, "stderr", object())
    monkeypatch.delitem(sys.modules, "ipykernel", raising=False)
    assert train_module._progress_disabled(None)


def test_progress_argument_reaches_tqdm(tmp_path, monkeypatch):
    seen = []
    real = train_module.tqdm

    def spy(*args, **kwargs):
        seen.append(kwargs["disable"])
        return real(*args, **kwargs)

    monkeypatch.setattr(train_module, "tqdm", spy)
    t = make(tmp_path)
    t.adam_step(epochs=1, val_every=1, verbose=False, progress=False)
    t.lbfgs_step("data", 1, verbose=False, progress=False)
    assert seen == [True, True]


# --------------------------------------------------------- gradient-norm cost
def test_grad_norm_every_samples_batches(tmp_path, monkeypatch):
    t = make(tmp_path, log_grad_norms=True, grad_norm_every=2)
    norms = iter(float(v) for v in range(1, 100))
    monkeypatch.setattr(t, "grad_norm", lambda loss: torch.tensor(next(norms)))
    stats = t.train_epoch()  # 4 training batches -> batches 0 and 2
    # calls per sampled batch: data, physics, boundary -> (1, 2, 3), (4, 5, 6)
    assert (stats["avg_grad_d"], stats["avg_grad_p"], stats["avg_grad_b"]) == (
        2.5,
        3.5,
        4.5,
    )


def test_aggregator_may_return_a_python_number(tmp_path):
    def aggregator(losses, weights, params):
        total = sum(w * losses[k] for k, w in weights.items() if losses[k] is not None)
        total.backward()
        return float(total.detach())

    stats = make(tmp_path, grad_aggregator=aggregator).train_epoch()
    assert stats["total_loss"] > 0


# ---------------------------------------------------- held-out validation sets
def test_held_out_collocation_sets_are_used_for_validation(tmp_path):
    held_phys = loader(torch.linspace(-1, 1, 12).unsqueeze(1), batch_size=4)
    held_bnd = loader(torch.tensor([[-1.0], [1.0]]), batch_size=2)
    t = make(tmp_path, val_phys_coll_loader=held_phys, val_bnd_coll_loader=held_bnd)
    _, phys, bnd = t.validate_test(t.val_loader)

    def mean_sq(fn, X):
        x = X.clone().requires_grad_(True)
        return torch.mean(fn(x, t.model(x)) ** 2).item()

    assert phys == pytest.approx(
        mean_sq(phys_fn, torch.linspace(-1, 1, 12).unsqueeze(1)), rel=1e-5
    )
    assert bnd == pytest.approx(
        mean_sq(bnd_fn, torch.tensor([[-1.0], [1.0]])), rel=1e-5
    )


def test_val_colloc_frac_sets_the_default_subset(tmp_path):
    calls = []
    coll = loader(torch.rand(64, 1), batch_size=4)  # 16 batches

    def counting(x, y):
        calls.append(1)
        return phys_fn(x, y)

    t = make(
        tmp_path, phys_coll_loader=coll, phys_residual=counting, val_colloc_frac=0.5
    )
    calls.clear()
    t.validate_test(t.val_loader)
    assert len(calls) == 8


# ------------------------------------------------------------- history entries
def test_history_records_validation_epochs_and_test_losses(tmp_path):
    t = make(tmp_path)
    h = t.adam_step(epochs=4, val_every=2, verbose=False)
    assert h["val_epochs"] == [2, 4]
    assert all(h[f"test_loss_{k}"] >= 0 for k in "tdpb")
    assert h["test_loss_t"] == pytest.approx(
        h["test_loss_d"] + h["test_loss_p"] + h["test_loss_b"]
    )
    assert t.history == h and t.history is not h

    t.lbfgs_step("combined", 2, verbose=False)
    assert len(t.history["lbfgs_loss_t"]) == 2 and "lbfgs_loss_t" not in h

    Save.excel(t.history, foldername=str(tmp_path))  # scalars and lists both save


def test_plotter_uses_recorded_validation_epochs():
    h = {"val_loss_t": [3.0, 2.0, 1.0], "val_epochs": [5, 10, 20]}
    np.testing.assert_array_equal(Plotter(h)._val_epochs(3), [5, 10, 20])
    np.testing.assert_array_equal(Plotter({"val_loss_t": [1.0]})._val_epochs(2), [1, 2])
    # an incomplete record falls back to val_every
    np.testing.assert_array_equal(
        Plotter({"val_epochs": [7]}, val_every=4)._val_epochs(2), [4, 8]
    )


# ------------------------------------------------------------ resumable training
def interrupted_then_resumed(tmp_path, monkeypatch, make_kw, epochs, stop_after):
    """Train with save_every=1, kill after `stop_after` epochs, then resume."""
    t = make(tmp_path, **make_kw())
    calls = {"n": 0}
    real = t.train_epoch

    def dying():
        if calls["n"] == stop_after:
            raise KeyboardInterrupt
        calls["n"] += 1
        return real()

    monkeypatch.setattr(t, "train_epoch", dying)
    with pytest.raises(KeyboardInterrupt):
        t.adam_step(epochs=epochs, val_every=2, verbose=False, save_every=1)

    resumed = make(tmp_path, **make_kw())  # a new process: fresh objects
    h = resumed.adam_step(epochs=epochs, val_every=2, verbose=False, resume=True)
    return resumed, h


@pytest.mark.parametrize("with_extras", [False, True])
def test_resume_reproduces_the_uninterrupted_run(tmp_path, monkeypatch, with_extras):
    def make_kw():
        kw = {"adapt_wts": True, "wt_update_every": 2}
        if with_extras:
            gen = torch.Generator().manual_seed(1)
            kw.update(
                train_loader=labeled(32, generator=gen),
                theta=nn.Parameter(torch.tensor([1.5])),
            )
        return kw

    def with_scheduler(t):
        t.scheduler = StepLR(t.optimizer, step_size=2, gamma=0.5)
        return t

    ref_dir, run_dir = tmp_path / "ref", tmp_path / "run"
    reference = with_scheduler(make(ref_dir, **make_kw()))
    expected = reference.adam_step(epochs=6, val_every=2, verbose=False)

    real_make = make

    def make_with_scheduler(path, **kw):
        return with_scheduler(real_make(path, **kw))

    monkeypatch.setitem(globals(), "make", make_with_scheduler)
    resumed, h = interrupted_then_resumed(run_dir, monkeypatch, make_kw, 6, 3)

    for key in ("loss_t", "val_loss_t", "val_metric", "wt_phys", "grad_data_hist"):
        assert h[key] == expected[key], key
    assert h["best_epoch"] == expected["best_epoch"]
    for a, b in zip(resumed.model.parameters(), reference.model.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    if with_extras:
        np.testing.assert_array_equal(
            np.stack(h["inv_param"]), np.stack(expected["inv_param"])
        )


def test_resume_without_saved_state_starts_over(tmp_path):
    t = make(tmp_path)
    assert t.last_ckpt_path.endswith("best_last.pth")
    with pytest.warns(UserWarning, match="no saved training state"):
        h = t.adam_step(epochs=2, val_every=1, verbose=False, resume=True)
    assert len(h["loss_t"]) == 2


def test_resume_of_a_finished_run_only_reevaluates(tmp_path, monkeypatch):
    first = make(tmp_path).adam_step(epochs=2, val_every=1, verbose=False, save_every=5)
    t = make(tmp_path)
    monkeypatch.setattr(t, "train_epoch", lambda: pytest.fail("trained again"))
    again = t.adam_step(epochs=2, val_every=1, verbose=False, resume=True)
    assert again["loss_t"] == first["loss_t"]
    assert again["best_epoch"] == first["best_epoch"]
    assert again["test_loss_t"] == pytest.approx(first["test_loss_t"])


def test_resume_with_a_loader_generator_not_in_the_saved_state(tmp_path):
    make(tmp_path).adam_step(epochs=2, val_every=1, verbose=False, save_every=1)
    gen = torch.Generator().manual_seed(3)
    t = make(tmp_path, train_loader=labeled(32, generator=gen))
    h = t.adam_step(epochs=3, val_every=1, verbose=False, resume=True)
    assert len(h["loss_t"]) == 3


def test_save_every_must_be_positive(tmp_path):
    with pytest.raises(ValueError, match="save_every must be a positive integer"):
        make(tmp_path).adam_step(epochs=2, val_every=1, save_every=0)


def test_last_state_path_without_extension(tmp_path):
    t = make(tmp_path, ckpt_path=str(tmp_path / "ckpt"))
    assert t.last_ckpt_path == str(tmp_path / "ckpt_last.pth")


# ----------------------------------------------------- metadata and loading
def test_checkpoint_metadata_and_generic_loading(tmp_path):
    torch.manual_seed(0)
    model = SANN([1, 8, 1], nn.Tanh)
    theta = nn.Parameter(torch.tensor([2.0]))
    meta = {"I_S_metrics": {"x": {"min": -1.0, "max": 1.0}}, "species": ["A"]}
    t = make(tmp_path, model=model, theta=theta, metadata=meta)
    t.adam_step(epochs=2, val_every=1, verbose=False, save_every=1)

    fresh = SANN([1, 8, 1], nn.Tanh)
    ckpt = Analyze.load_checkpoint(t.ckpt_path, fresh, "cpu")
    assert ckpt["metadata"] == meta
    torch.testing.assert_close(ckpt["theta"], theta.detach())
    assert not fresh.training
    for a, b in zip(fresh.parameters(), model.parameters()):
        torch.testing.assert_close(a, b)

    other = Analyze.load_model(SANN([1, 8, 1], nn.Tanh), t.ckpt_path)
    assert isinstance(other, SANN) and not other.training
    assert torch.load(t.last_ckpt_path, weights_only=True)["metadata"] == meta


def test_load_checkpoint_accepts_a_bare_state_dict(tmp_path):
    model = ANN([2, 4, 1], nn.Tanh)
    path = tmp_path / "raw.pth"
    torch.save(model.state_dict(), path)
    ckpt = Analyze.load_checkpoint(str(path))
    assert set(ckpt) == {"model_state_dict"}
    loaded = Analyze.load_ann_pinn(2, 1, 1, 4, nn.Tanh, str(path), torch.device("cpu"))
    for a, b in zip(loaded.parameters(), model.parameters()):
        torch.testing.assert_close(a, b)


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


# ------------------------------------------------------------- small helpers
def test_plain_and_json_like_helpers():
    assert train_module._plain({"a": (np.float32(1.5), np.array([1, 2]))}) == {
        "a": [1.5, [1, 2]]
    }
    assert train_module._json_like({"a": [1, 2.0, "x", None, True, (1,)]})
    assert not train_module._json_like({"a": object()})
