import copy
import math

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader, TensorDataset

import pytest

from pinnse import ANN, PCGrad, Training

"""
Training tests use a one-dimensional problem with a known solution, y = x^2,
a physics residual dy/dx - 2x and a boundary residual y - x^2.
"""


class Residual:
    """Callable residual that records how it was called."""

    def __init__(self, fn):
        self.fn = fn
        self.calls = []

    def __call__(self, x, y):
        self.calls.append(x.requires_grad)
        return self.fn(x, y)


def phys_fn(x, y):
    dy = torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]
    return dy - 2.0 * x


def bnd_fn(x, y):
    return y - x**2


def loader(X, Y=None, batch_size=8):
    ds = TensorDataset(X) if Y is None else TensorDataset(X, Y)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


def labeled(n):
    X = torch.linspace(-1, 1, n).unsqueeze(1)
    return loader(X, X**2)


def make_trainer(tmp_path, phys=True, bnd=True, lr=1e-2, model=None, **kw):
    torch.manual_seed(0)
    model = model or ANN([1, 8, 1], activation=nn.Tanh)
    params = list(model.parameters())
    if kw.get("theta") is not None:
        params.append(kw["theta"])
    defaults = dict(
        model=model,
        train_loader=labeled(32),
        val_loader=labeled(16),
        test_loader=labeled(8),
        optimizer=torch.optim.Adam(params, lr=lr),
        loss_fn=nn.MSELoss(),
        device=torch.device("cpu"),
        phys_coll_loader=loader(torch.rand(16, 1) * 2 - 1) if phys else None,
        bnd_coll_loader=loader(torch.tensor([[-1.0], [1.0]] * 4), batch_size=4)
        if bnd
        else None,
        phys_residual=Residual(phys_fn) if phys else None,
        bnd_residual=Residual(bnd_fn) if bnd else None,
        ckpt_path=str(tmp_path / "best.pth"),
        phys_weight=1.0 if phys else 0.0,
        bnd_weight=1.0 if bnd else 0.0,
        log_grad_norms=True,
    )
    defaults.update(kw)
    return Training(**defaults)


def params_of(model):
    return [p.detach().clone() for p in model.parameters()]


def mean_sq_residual(trainer, batches, fn):
    x = torch.cat(batches).requires_grad_(True)
    return torch.mean(fn(x, trainer.model(x)) ** 2).item()


# --------------------------------- grad_norm ---------------------------------
def test_grad_norm_matches_manual_norm(tmp_path):
    t = make_trainer(tmp_path)
    X = torch.linspace(-1, 1, 10).unsqueeze(1)
    loss = t.loss_fn(t.model(X), X**2)

    grads = torch.autograd.grad(loss, t.trainable_params, retain_graph=True)
    expected = math.sqrt(sum(g.pow(2).sum().item() for g in grads))
    assert t.grad_norm(loss).item() == pytest.approx(expected, rel=1e-5)


def test_grad_norm_is_zero_without_dependence_on_parameters(tmp_path):
    t = make_trainer(tmp_path)
    assert t.grad_norm(None).item() == 0.0
    unrelated = torch.tensor(2.0, requires_grad=True) ** 2
    assert t.grad_norm(unrelated).item() == 0.0


# -------------------------------- train_epoch --------------------------------
def test_gradient_norms_logged_only_on_request(tmp_path, monkeypatch):
    # By default (no adaptive weights) the extra backward passes are skipped
    # and the gradient histories are recorded as zeros.
    t = make_trainer(tmp_path, log_grad_norms=False)
    monkeypatch.setattr(
        t, "grad_norm", lambda loss: pytest.fail("grad_norm should not run")
    )
    stats = t.train_epoch()
    assert (stats["avg_grad_d"], stats["avg_grad_p"], stats["avg_grad_b"]) == (
        0.0,
        0.0,
        0.0,
    )
    assert stats["data_loss"] > 0

    # Adaptive weighting needs the norms, so it switches the logging on.
    assert make_trainer(tmp_path, log_grad_norms=False, adapt_wts=True).log_grad_norms


def test_checkpoint_directory_is_created(tmp_path):
    ckpt = tmp_path / "nested" / "dir" / "best.pth"
    t = make_trainer(tmp_path, phys=False, bnd=False, ckpt_path=str(ckpt))
    t.adam_step(epochs=1, val_every=1, verbose=False)
    assert ckpt.exists()


def test_train_epoch_data_only(tmp_path):
    t = make_trainer(tmp_path, phys=False, bnd=False)
    before = params_of(t.model)
    stats = t.train_epoch()

    assert stats["phys_loss"] == 0.0 and stats["bnd_loss"] == 0.0
    assert stats["avg_grad_p"] == 0.0 and stats["avg_grad_b"] == 0.0
    assert stats["total_loss"] == pytest.approx(stats["data_loss"])
    assert stats["data_loss"] > 0 and stats["avg_grad_d"] > 0
    assert any(not torch.equal(a, b) for a, b in zip(before, params_of(t.model)))


def test_train_epoch_combines_weighted_losses(tmp_path):
    t = make_trainer(tmp_path, phys_weight=0.5, bnd_weight=2.0)
    stats = t.train_epoch()

    assert stats["phys_loss"] > 0 and stats["bnd_loss"] > 0
    assert stats["avg_grad_p"] > 0 and stats["avg_grad_b"] > 0
    # equal batch sizes, so the averaged total is the weighted sum of averages
    assert stats["total_loss"] == pytest.approx(
        stats["data_loss"] + 0.5 * stats["phys_loss"] + 2.0 * stats["bnd_loss"],
        rel=1e-5,
    )
    # one collocation batch per labeled batch, cycling over the shorter
    # collocation loaders, and always with gradients enabled on the inputs
    assert t.phys_residual.calls == [True] * 4
    assert t.bnd_residual.calls == [True] * 4


@pytest.mark.parametrize(
    "overrides",
    [
        {"phys_weight": 0.0},
        {"phys_residual": None},
        {"phys_coll_loader": None},
    ],
)
def test_train_epoch_skips_physics_when_disabled(tmp_path, overrides):
    t = make_trainer(tmp_path, bnd=False, **overrides)
    stats = t.train_epoch()
    assert stats["phys_loss"] == 0.0
    if t.phys_residual is not None:
        assert t.phys_residual.calls == []


def test_train_epoch_skips_boundary_when_disabled(tmp_path):
    t = make_trainer(tmp_path, phys=False, bnd_weight=0.0)
    assert t.train_epoch()["bnd_loss"] == 0.0
    assert t.bnd_residual.calls == []


def test_train_epoch_on_empty_loader(tmp_path):
    t = make_trainer(
        tmp_path, train_loader=loader(torch.empty(0, 1), torch.empty(0, 1))
    )
    stats = t.train_epoch()
    assert all(v == 0.0 for v in stats.values())


# ----------------------------- grad aggregation ------------------------------
class SpyAggregator:
    def __init__(self):
        self.calls = []

    def __call__(self, losses, weights, params):
        self.calls.append((dict(losses), dict(weights), params))
        total = sum(weights[k] * loss for k, loss in losses.items() if loss is not None)
        total.backward()
        return total


def test_custom_aggregator_receives_losses_weights_and_params(tmp_path):
    agg = SpyAggregator()
    t = make_trainer(tmp_path, phys_weight=0.5, bnd=False, grad_aggregator=agg)
    stats = t.train_epoch()

    assert len(agg.calls) == 4
    losses, weights, params = agg.calls[0]
    assert set(losses) == {"data", "physics", "boundary"}
    assert losses["boundary"] is None
    assert weights == {"data": 1.0, "physics": 0.5, "boundary": 0.0}
    assert params is t.agg_params
    assert stats["total_loss"] == pytest.approx(
        stats["data_loss"] + 0.5 * stats["phys_loss"], rel=1e-5
    )


def test_inverse_parameter_is_aggregated_only_when_trainable(tmp_path):
    theta = nn.Parameter(torch.tensor([1.0]))
    t = make_trainer(tmp_path, theta=theta)
    assert t.agg_params[-1] is theta
    assert len(t.agg_params) == len(t.trainable_params) + 1

    frozen = nn.Parameter(torch.tensor([1.0]), requires_grad=False)
    t = make_trainer(tmp_path, theta=frozen)
    assert all(p is not frozen for p in t.agg_params)


def test_pcgrad_trains_network_and_inverse_parameter(tmp_path):
    theta = nn.Parameter(torch.tensor([1.5]))

    def phys_theta(x, y):
        dy = torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]
        return dy - theta * x

    t = make_trainer(
        tmp_path, theta=theta, phys_residual=phys_theta, grad_aggregator=PCGrad()
    )
    before = params_of(t.model)
    stats = t.train_epoch()

    assert math.isfinite(stats["total_loss"])
    assert theta.item() != 1.5
    assert any(not torch.equal(a, b) for a, b in zip(before, params_of(t.model)))


# ----------------------------------- PCGrad ----------------------------------
def test_pcgrad_projects_conflicting_gradients():
    p = torch.zeros(2, requires_grad=True)
    a, b = torch.tensor([1.0, 0.0]), torch.tensor([-1.0, 1.0])
    losses = {"data": p @ a, "physics": p @ b}

    total = PCGrad()(losses, {"data": 1.0, "physics": 1.0}, [p])

    # g_a projected off g_b: (0.5, 0.5); g_b projected off g_a: (0, 1)
    torch.testing.assert_close(p.grad, torch.tensor([0.5, 1.5]))
    assert total.item() == 0.0
    assert not total.requires_grad


def test_pcgrad_leaves_agreeing_gradients_untouched():
    p = torch.ones(2, requires_grad=True)
    losses = {
        "data": p @ torch.tensor([1.0, 0.0]),
        "physics": p @ torch.tensor([1.0, 1.0]),
    }

    total = PCGrad()(losses, {"data": 1.0, "physics": 3.0}, [p])

    torch.testing.assert_close(p.grad, torch.tensor([4.0, 3.0]))
    assert total.item() == pytest.approx(1.0 + 3.0 * 2.0)


def test_pcgrad_without_weights_and_with_missing_losses():
    p = torch.ones(2, requires_grad=True)
    unused = torch.ones(3, requires_grad=True)
    losses = {"data": p.sum(), "physics": None, "boundary": 2 * p.sum()}

    total = PCGrad(apply_weights=False)(
        losses, {"data": 1.0, "physics": 9.0, "boundary": 9.0}, [p, unused]
    )

    torch.testing.assert_close(p.grad, torch.tensor([3.0, 3.0]))
    torch.testing.assert_close(unused.grad, torch.zeros(3))
    assert total.item() == pytest.approx(2.0 + 4.0)


def test_pcgrad_requires_a_loss():
    p = torch.ones(1, requires_grad=True)
    with pytest.raises(ValueError, match="at least one"):
        PCGrad()({"data": None}, {}, [p])


# ------------------------------- validate_test -------------------------------
def test_validate_test_data_loss_is_sample_weighted(tmp_path):
    t = make_trainer(tmp_path, phys=False, bnd=False, val_loader=None)
    X = torch.linspace(-1, 1, 12).unsqueeze(1)
    uneven = loader(X, X**2, batch_size=5)  # batches of 5, 5 and 2

    data, phys, bnd = t.validate_test(uneven)
    with torch.no_grad():
        expected = t.loss_fn(t.model(X), X**2).item()
    assert data == pytest.approx(expected, rel=1e-5)
    assert (phys, bnd) == (0.0, 0.0)
    assert not t.model.training


def test_validate_test_residual_losses_over_all_batches(tmp_path):
    t = make_trainer(tmp_path)
    _, phys, bnd = t.validate_test(
        t.val_loader, phys_colloc_frac=1.0, bnd_colloc_frac=5.0
    )

    assert phys == pytest.approx(
        mean_sq_residual(t, t.all_phys_colloc_batches, phys_fn), rel=1e-5
    )
    assert bnd == pytest.approx(
        mean_sq_residual(t, t.all_bnd_colloc_batches, bnd_fn), rel=1e-5
    )


def test_validate_test_samples_a_fraction_of_collocation_batches(tmp_path):
    t = make_trainer(tmp_path, phys_coll_loader=loader(torch.rand(32, 1), batch_size=4))
    t.phys_residual.calls.clear()
    t.validate_test(t.val_loader, phys_colloc_frac=0.25)
    assert len(t.phys_residual.calls) == 2  # 25% of 8 batches


@pytest.mark.parametrize("n_batches", [1, 2])
def test_validate_test_evaluates_physics_with_few_collocation_batches(
    tmp_path, n_batches
):
    # With the default 25% fraction, one or two collocation batches round to
    # zero sampled batches; at least one must still be evaluated, as for the
    # boundary loss, or the validation total silently ignores the physics.
    t = make_trainer(tmp_path, phys_coll_loader=loader(torch.rand(8 * n_batches, 1)))
    _, phys, bnd = t.validate_test(t.val_loader)
    assert phys > 0 and bnd > 0


def test_validate_test_ignores_disabled_residuals(tmp_path):
    t = make_trainer(tmp_path, phys_weight=0.0, bnd_weight=0.0)
    assert t.validate_test(t.val_loader)[1:] == (0.0, 0.0)
    assert t.phys_residual.calls == [] and t.bnd_residual.calls == []


# --------------------------------- adam_step ---------------------------------
def test_adam_step_history_checkpoint_and_logging(tmp_path, capsys):
    t = make_trainer(tmp_path, phys_weight=0.5, bnd_weight=0.25)
    h = t.adam_step(epochs=4, val_every=2, verbose=True)

    assert {
        len(h[k]) for k in ("loss_t", "loss_d", "loss_p", "loss_b", "wt_phys", "wt_bnd")
    } == {4}
    assert {
        len(h[k]) for k in ("grad_data_hist", "grad_phys_hist", "grad_bnd_hist")
    } == {4}
    assert {
        len(h[k]) for k in ("val_loss_t", "val_loss_d", "val_loss_p", "val_loss_b")
    } == {2}
    assert h["inv_param"] == []
    assert h["wt_phys"] == [0.5] * 4 and h["wt_bnd"] == [0.25] * 4
    for tot, d, p, b in zip(
        h["val_loss_t"], h["val_loss_d"], h["val_loss_p"], h["val_loss_b"]
    ):
        assert tot == pytest.approx(d + 0.5 * p + 0.25 * b)

    ckpt = torch.load(tmp_path / "best.pth")
    for k, v in t.model.state_dict().items():
        torch.testing.assert_close(v, ckpt["model_state_dict"][k])

    out = capsys.readouterr().out
    assert "Epoch: 1 | Training" in out
    assert out.count("| Validation |") == 2
    assert "Test | Total Loss" in out


def test_adam_step_quiet(tmp_path, capsys):
    t = make_trainer(tmp_path)
    t.adam_step(epochs=2, val_every=1, verbose=False)
    out = capsys.readouterr().out
    assert "Epoch:" not in out and "Test |" not in out


def test_adam_step_restores_best_validation_checkpoint(tmp_path):
    t = make_trainer(tmp_path, phys=False, bnd=False)
    scripted = iter([(3.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)])
    snapshots = []

    def fake_validate(loader):
        if loader is t.test_loader:
            return 0.0, 0.0, 0.0
        snapshots.append(copy.deepcopy(t.model.state_dict()))
        return next(scripted)

    t.validate_test = fake_validate
    h = t.adam_step(epochs=3, val_every=1, verbose=False)

    assert h["val_loss_t"] == [3.0, 1.0, 2.0]
    for k, v in t.model.state_dict().items():
        torch.testing.assert_close(v, snapshots[1][k])
        assert not torch.equal(v, snapshots[2][k])


def test_adam_step_default_adaptive_weights(tmp_path):
    t = make_trainer(tmp_path, adapt_wts=True)
    t.wt_update_every = 1
    h = t.adam_step(epochs=3, val_every=3, verbose=False)

    wp, wb = 1.0, 1.0
    for e in range(3):
        gd, gp, gb = (
            h["grad_data_hist"][e],
            h["grad_phys_hist"][e],
            h["grad_bnd_hist"][e],
        )
        wp = 0.9 * wp + 0.1 * gd / (gp + 1e-12)
        wb = 0.9 * wb + 0.1 * gd / (gb + 1e-12)
        assert h["wt_phys"][e] == pytest.approx(wp)
        assert h["wt_bnd"][e] == pytest.approx(wb)


def test_adaptive_weights_update_only_every_n_epochs(tmp_path):
    t = make_trainer(tmp_path, adapt_wts=True)
    t.wt_update_every = 2
    h = t.adam_step(epochs=3, val_every=3, verbose=False)
    assert h["wt_phys"][0] == 1.0 and h["wt_bnd"][0] == 1.0
    assert h["wt_phys"][1] != 1.0 and h["wt_bnd"][1] != 1.0
    assert h["wt_phys"][2] == h["wt_phys"][1]


@pytest.mark.parametrize("vanishing", ["phys", "bnd"])
def test_adaptive_weights_keep_weight_when_its_gradient_vanishes(tmp_path, vanishing):
    # a residual that does not depend on the network has a zero gradient norm
    constant = {f"{vanishing}_residual": lambda x, y: x * 0 + 1}
    t = make_trainer(tmp_path, adapt_wts=True, **constant)
    t.wt_update_every = 1
    h = t.adam_step(epochs=2, val_every=2, verbose=False)

    frozen, adapted = (
        ("wt_phys", "wt_bnd") if vanishing == "phys" else ("wt_bnd", "wt_phys")
    )
    assert h[frozen] == [1.0, 1.0]
    assert h[adapted][0] != 1.0


def test_adam_step_custom_weight_scheme(tmp_path):
    seen = []

    def scheme(gd, gp, gb):
        seen.append((gd, gp, gb))
        return 5.0, 7.0

    t = make_trainer(tmp_path, adapt_wts=True, weight_scheme=scheme)
    t.wt_update_every = 1
    h = t.adam_step(epochs=2, val_every=2, verbose=False)

    assert seen[0] == (
        h["grad_data_hist"][0],
        h["grad_phys_hist"][0],
        h["grad_bnd_hist"][0],
    )
    assert h["wt_phys"] == pytest.approx([1.4, 0.9 * 1.4 + 0.5])
    assert h["wt_bnd"] == pytest.approx([1.6, 0.9 * 1.6 + 0.7])


def test_adaptive_weights_adapt_only_active_weights(tmp_path):
    # A zero weight disables its loss and stays zero; the other weight adapts.
    called = []
    t = make_trainer(
        tmp_path,
        bnd=False,
        adapt_wts=True,
        weight_scheme=lambda *g: called.append(g) or (9.0, 9.0),
    )
    t.wt_update_every = 1
    h = t.adam_step(epochs=2, val_every=2, verbose=False)
    assert h["wt_phys"] == pytest.approx([1.8, 0.9 * 1.8 + 0.9])
    assert h["wt_bnd"] == [0.0, 0.0]
    assert len(called) == 2


def test_default_adaptive_weights_with_single_residual(tmp_path):
    t = make_trainer(tmp_path, phys=False, adapt_wts=True)
    t.wt_update_every = 1
    h = t.adam_step(epochs=2, val_every=2, verbose=False)
    assert h["wt_phys"] == [0.0, 0.0]
    assert h["wt_bnd"][0] != 1.0 and h["wt_bnd"][1] != h["wt_bnd"][0]


def test_adam_step_raises_when_no_validation_epoch_is_reached(tmp_path):
    t = make_trainer(tmp_path, phys=False, bnd=False)
    with pytest.raises(RuntimeError, match="No checkpoint was written"):
        t.adam_step(epochs=1, val_every=2, verbose=False)
    assert not (tmp_path / "best.pth").exists()


def test_adam_step_ignores_non_finite_validation_losses(tmp_path):
    t = make_trainer(tmp_path, phys=False, bnd=False)
    nan = float("nan")
    scripted = iter([(nan, 0.0, 0.0), (1.0, 0.0, 0.0), (nan, 0.0, 0.0)])
    snapshots = []

    def fake_validate(loader):
        if loader is t.test_loader:
            return 0.0, 0.0, 0.0
        snapshots.append(copy.deepcopy(t.model.state_dict()))
        return next(scripted)

    t.validate_test = fake_validate
    t.adam_step(epochs=3, val_every=1, verbose=False)
    for k, v in t.model.state_dict().items():
        torch.testing.assert_close(v, snapshots[1][k])


def test_adam_step_raises_when_every_validation_loss_is_non_finite(tmp_path):
    # A stale checkpoint from an earlier run must not be reloaded silently.
    t = make_trainer(tmp_path, phys=False, bnd=False)
    stale = {"model_state_dict": {"marker": torch.tensor(1.0)}}
    torch.save(stale, t.ckpt_path)
    t.validate_test = lambda loader: (float("inf"), 0.0, 0.0)
    with pytest.raises(RuntimeError, match="non-finite"):
        t.adam_step(epochs=2, val_every=1, verbose=False)
    assert torch.load(t.ckpt_path)["model_state_dict"].keys() == {"marker"}


def test_checkpoint_restores_inverse_parameter_of_best_epoch(tmp_path):
    theta = nn.Parameter(torch.tensor([1.5]))

    def phys_theta(x, y):
        dy = torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]
        return dy - theta * x

    t = make_trainer(tmp_path, theta=theta, phys_residual=phys_theta, bnd=False)
    scripted = iter([(3.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)])
    thetas = []
    real_validate = t.validate_test

    def fake_validate(loader):
        if loader is t.test_loader:
            return real_validate(loader)
        thetas.append(theta.detach().clone())
        return next(scripted)

    t.validate_test = fake_validate
    h = t.adam_step(epochs=3, val_every=1, verbose=False)

    assert not torch.equal(thetas[1], thetas[2])
    torch.testing.assert_close(theta.detach(), thetas[1])
    torch.testing.assert_close(torch.load(t.ckpt_path)["theta"], thetas[1])
    # the recorded history still follows the parameter epoch by epoch
    assert len(h["inv_param"]) == 3


def test_adam_step_records_inverse_parameter(tmp_path):
    theta = nn.Parameter(torch.tensor([1.0, 2.0]))

    def phys_theta(x, y):
        dy = torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]
        return dy - theta[0] * theta[1] * x

    t = make_trainer(tmp_path, theta=theta, phys_residual=phys_theta, bnd=False)
    h = t.adam_step(epochs=4, val_every=2, verbose=False)

    assert len(h["inv_param"]) == 2
    assert h["inv_param"][0].shape == (2,)
    assert not (h["inv_param"][0] == [1.0, 2.0]).all()
    assert (h["inv_param"][-1] == theta.detach().numpy()).all()


def test_adam_step_steps_epoch_scheduler_every_epoch(tmp_path):
    t = make_trainer(tmp_path, lr=0.1)
    t.scheduler = StepLR(t.optimizer, step_size=1, gamma=0.5)
    t.adam_step(epochs=3, val_every=2, verbose=False)
    assert t.optimizer.param_groups[0]["lr"] == pytest.approx(0.1 * 0.5**3)


def test_adam_step_steps_plateau_scheduler_on_validation_loss(tmp_path):
    class Spy(ReduceLROnPlateau):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.metrics = []

        def step(self, metrics):
            self.metrics.append(metrics)
            super().step(metrics)

    t = make_trainer(tmp_path)
    t.scheduler = Spy(t.optimizer)
    h = t.adam_step(epochs=4, val_every=2, verbose=False)
    assert t.scheduler.metrics == h["val_loss_t"]


# --------------------------------- lbfgs_step --------------------------------
def test_lbfgs_rejects_unknown_mode(tmp_path):
    with pytest.raises(ValueError, match="mode must be one of"):
        make_trainer(tmp_path).lbfgs_step(mode="everything")


@pytest.mark.parametrize(
    "mode, uses",
    [
        ("data", "d"),
        ("physics", "p"),
        ("boundary", "b"),
        ("overall physics", "pb"),
        ("Combined", "dpb"),
    ],
)
def test_lbfgs_modes_select_loss_terms(tmp_path, mode, uses):
    t = make_trainer(tmp_path, phys_weight=0.5, bnd_weight=2.0)
    torch.save({"model_state_dict": t.model.state_dict()}, t.ckpt_path)
    h = t.lbfgs_step(mode=mode, N_LBFGS=3, verbose=False)

    assert {len(v) for v in h.values()} == {3}
    for name, key in (("d", "loss_d"), ("p", "loss_p"), ("b", "loss_b")):
        assert all(v > 0 for v in h[key]) == (name in uses)
        assert all(v == 0 for v in h[key]) == (name not in uses)
    for tot, d, p, b in zip(h["loss_t"], h["loss_d"], h["loss_p"], h["loss_b"]):
        assert tot == pytest.approx(d + 0.5 * p + 2.0 * b)


def test_lbfgs_reduces_loss_and_keeps_best_state(tmp_path, capsys):
    t = make_trainer(tmp_path, phys=False, bnd=False)
    torch.save({"model_state_dict": t.model.state_dict()}, t.ckpt_path)
    X, Y = (
        torch.linspace(-1, 1, 32).unsqueeze(1),
        torch.linspace(-1, 1, 32).unsqueeze(1) ** 2,
    )
    with torch.no_grad():
        initial = t.loss_fn(t.model(X), Y).item()

    h = t.lbfgs_step(mode="data", N_LBFGS=10, verbose=True)

    assert min(h["loss_t"]) < initial / 2
    with torch.no_grad():
        final = t.loss_fn(t.model(X), Y).item()
    assert final == pytest.approx(min(h["loss_t"]), rel=1e-5)
    assert "LBFGS Step: 10" in capsys.readouterr().out


def test_lbfgs_optimizes_and_restores_inverse_parameter(tmp_path):
    theta = nn.Parameter(torch.tensor([1.5]))

    def phys_theta(x, y):
        dy = torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]
        return dy - theta * x

    t = make_trainer(tmp_path, theta=theta, phys_residual=phys_theta, bnd=False)
    t._save_checkpoint()
    assert any(p is theta for p in t.agg_params)

    h = t.lbfgs_step(mode="physics", N_LBFGS=5, verbose=False)
    assert theta.item() != 1.5
    assert min(h["loss_p"]) < h["loss_p"][0] or h["loss_p"][-1] == min(h["loss_p"])
    torch.testing.assert_close(torch.load(t.ckpt_path)["theta"], theta.detach())


def test_lbfgs_without_collocation_data(tmp_path):
    t = make_trainer(tmp_path, phys=False, bnd=False, phys_weight=1.0, bnd_weight=1.0)
    torch.save({"model_state_dict": t.model.state_dict()}, t.ckpt_path)
    h = t.lbfgs_step(mode="overall physics", N_LBFGS=2, verbose=False)
    assert h == {
        "loss_t": [0.0, 0.0],
        "loss_d": [0.0, 0.0],
        "loss_p": [0.0, 0.0],
        "loss_b": [0.0, 0.0],
    }
    assert t.X_phys_all is None and t.X_bnd_all is None
