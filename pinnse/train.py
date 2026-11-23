"""
Training module for supervised and PINN models.

This module defines the `Training` class used to train, validate, and test
PyTorch-based PINN models with optional physics-residual and
boundary-residual constraints. The class supports:
- supervised data loss evaluation,
- optional physics and boundary collocation losses,
- adaptive loss-weight updates based on gradient norms,
- model checkpointing using validation performance, and
- storage of training, validation, and gradient histories.

The implementation is designed for PINN workflows in which the total
training objective may combine labeled-data loss with additional residual-based
penalty terms.

By default the individual losses are combined into a scalar weighted sum. For
training strategies that operate on the per-loss gradients themselves rather
than on scalar weights, the `grad_aggregator` extension point accepts a
user-supplied callable that receives the individual losses, the current loss
weights, and the trainable parameters, and is responsible for populating the
parameter gradients. `PCGrad` is provided as a reference implementation; any
other multi-objective strategy can be supplied without modifying this module.
"""

import logging
import math
import os
import random
import sys
import warnings
from itertools import cycle
from typing import Callable, Optional, Union

import torch
from torch import nn
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau

import numpy as np
from tqdm import tqdm

from ._deprecation import warn_renamed

CKPT_METRICS = ("total", "total_current", "data")


# Progress messages are logged here (INFO); `verbose=True` also prints them.
_LOGGER = logging.getLogger("pinnse")

# History lists built by `adam_step` (and saved with the resumable state).
_ADAM_LISTS = (
    "loss_t",
    "loss_d",
    "loss_p",
    "loss_b",
    "val_loss_t",
    "val_loss_d",
    "val_loss_p",
    "val_loss_b",
    "val_metric",
    "val_epochs",
    "wt_phys",
    "wt_bnd",
    "inv_param",
)


def _first_tensors(loader) -> list:
    """The first tensor of every batch of `loader` (empty for None)."""
    return [batch[0] for batch in loader] if loader is not None else []


def _value(x) -> float:
    """Python float of a 0-d tensor or a number."""
    return x.item() if torch.is_tensor(x) else float(x)


def _plain(obj):
    """Convert arrays and NumPy scalars to lists and Python numbers, recursively."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {key: _plain(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(value) for value in obj]
    return obj


def _json_like(obj) -> bool:
    """Whether `obj` holds only dicts with string keys, lists and scalars."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return True
    if isinstance(obj, (list, tuple)):
        return all(_json_like(value) for value in obj)
    if isinstance(obj, dict):
        return all(isinstance(k, str) and _json_like(v) for k, v in obj.items())
    return False


def _progress_disabled(progress: Optional[bool]) -> bool:
    """
    tqdm's `disable` flag: the explicit choice, or else on in terminals and
    notebooks and off when stderr is not a terminal (e.g. batch-job logs).
    """
    if progress is not None:
        return not progress
    if "ipykernel" in sys.modules:
        return False
    isatty = getattr(sys.stderr, "isatty", None)
    return not (isatty is not None and isatty())


def _require_positive_int(value, name: str) -> int:
    """Return `value` as an int, rejecting non-integers and values below one."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer; received {value!r}.")
    return int(value)


class PCGrad(object):
    """
    Projecting-Conflicting-Gradients (PCGrad) loss aggregation.

    Reference implementation of the gradient-surgery strategy of Yu et al.
    (2020), provided as a ready-to-use `grad_aggregator` for `Training`. Where
    gradient-norm-based adaptive weighting balances only the magnitudes of the
    individual loss gradients, PCGrad additionally resolves conflicts in their
    directions: whenever two loss gradients have a negative inner product, each
    is projected onto the normal plane of the other before the gradients are
    summed.

    Inputs
    ------
    apply_weights : bool, optional, default=True
        Whether to scale each loss by its current weight before computing the
        per-loss gradients. If False, the raw losses are used and the physics
        and boundary weights act only as on/off switches.
    eps : float, optional, default=1e-12
        Small positive constant used for numerical stability.

    Notes
    -----
    Supplying a `grad_aggregator` replaces the default scalar weighted-sum
    update. A custom strategy only needs to match the call signature
    `(losses, weights, params) -> torch.Tensor`, populate `p.grad` for every
    parameter in `params`, and return the scalar total loss for logging.
    """

    def __init__(self, apply_weights: bool = True, eps: float = 1e-12):
        self.apply_weights = apply_weights
        self.eps = eps

    def __call__(self, losses: dict, weights: dict, params: list):
        names = [name for name, loss in losses.items() if loss is not None]
        if not names:
            raise ValueError("PCGrad requires at least one non-None loss.")

        flat_grads = []
        for name in names:
            weight = weights.get(name, 1.0) if self.apply_weights else 1.0
            raw = torch.autograd.grad(
                weight * losses[name],
                params,
                retain_graph=True,
                allow_unused=True,
            )
            grads = [
                torch.zeros_like(p) if g is None else g for p, g in zip(params, raw)
            ]
            flat_grads.append(torch.cat([g.reshape(-1) for g in grads]))

        # Project each loss gradient off any conflicting loss gradient
        projected = [g.clone() for g in flat_grads]
        n_tasks = len(flat_grads)
        for i in range(n_tasks):
            for j in torch.randperm(n_tasks).tolist():
                if i == j:
                    continue
                inner = torch.dot(projected[i], flat_grads[j])
                if inner < 0:
                    projected[i] = (
                        projected[i]
                        - (inner / (torch.dot(flat_grads[j], flat_grads[j]) + self.eps))
                        * flat_grads[j]
                    )

        # Write the aggregated gradient back onto the parameters
        aggregated = torch.stack(projected).sum(dim=0)
        offset = 0
        for p in params:
            numel = p.numel()
            p.grad = aggregated[offset : offset + numel].view_as(p).clone()
            offset += numel

        total = sum(
            (weights.get(name, 1.0) if self.apply_weights else 1.0)
            * losses[name].detach()
            for name in names
        )
        return total


class Training(object):
    """
    Train, validate and test a PyTorch model with an optional physics loss and
    boundary loss.

    Inputs
    ------
    model : nn.Module
        Network to train. Its forward pass maps an input batch to one tensor.
    train_loader, val_loader, test_loader : DataLoader
        Labeled loaders yielding `(x, y)` batches.
    optimizer : torch.optim.Optimizer
        Optimizer of `adam_step`. For inverse problems it must also hold `theta`.
    loss_fn : nn.Module
        Data loss, e.g. `nn.MSELoss()`.
    device : torch.device
        Device the batches are moved to.
    scheduler : LRScheduler or ReduceLROnPlateau, optional
        Stepped every epoch, or, for ReduceLROnPlateau, on every validation
        with the checkpoint metric.
    phys_coll_loader, bnd_coll_loader : DataLoader, optional
        Physics- and boundary-collocation loaders yielding `(x,)` batches.
    phys_residual, bnd_residual : callable, optional
        `(x, y) -> residual` with one row per point; the loss is the mean
        squared residual.
    ckpt_path : str, optional, default="./logs/best_model.pth"
        File of the best checkpoint.
    phys_weight, bnd_weight : float, optional, default=0.0
        Non-negative weights of the physics and boundary losses; zero disables
        a loss.
    adapt_wts : bool, optional, default=False
        Adapt the non-zero weights from the gradient norms of the losses.
    theta : nn.Parameter, optional
        Unknown process parameters of an inverse problem, saved with and
        restored from the checkpoints.
    grad_aggregator : callable, optional
        Strategy combining the per-loss gradients (see `PCGrad`).
    weight_scheme : callable, optional
        `(grad_d, grad_p, grad_b) -> (target_wp, target_wb)` replacing the
        default adaptive-weight rule.
    log_grad_norms : bool, optional, default=False
        Whether to record the per-epoch gradient norms of the data, physics
        and boundary losses (`grad_data_hist`, `grad_phys_hist`,
        `grad_bnd_hist`). Each norm costs an additional backward pass, so they
        are computed only when requested here or when `adapt_wts=True`, which
        needs them; otherwise the histories are recorded as zeros.
    seed : int, optional, default=None
        Seed of the random generator owned by this trainer, which selects
        the collocation batches evaluated during validation. If None, it is
        taken from `torch.initial_seed()`, so a run seeded with
        `torch.manual_seed` (or `pinnse.seed_everything`) is reproducible.
    ckpt_metric : str, optional, default="total"
        Validation quantity used to select the best checkpoint and to step
        a `ReduceLROnPlateau` scheduler:

        - ``"total"``: data loss plus the physics and boundary losses
          weighted by the *initial* weights. With adaptive weights this
          keeps every validation epoch on the same scale.
        - ``"total_current"``: as above, but with the weights current at
          that epoch (the behaviour of pinnse < 1.0).
        - ``"data"``: validation data loss only.
    wt_update_every : int, optional, default=10
        Epochs between two updates of the adaptive weights.
    wt_ema : float, optional, default=0.1
        Smoothing factor in (0, 1] of the adaptive-weight update: each update
        moves a weight this fraction of the way to its target.
    val_colloc_frac : float, optional, default=0.25
        Fraction of the training collocation batches on which the validation
        and test residual losses are evaluated (at least one batch). The
        subset is drawn once, so every validation uses the same points.
    val_phys_coll_loader, val_bnd_coll_loader : DataLoader, optional
        Held-out collocation points. When given, the validation and test
        residual losses are evaluated on all of these instead of a subset of
        the training points.
    grad_norm_every : int, optional, default=1
        Compute the logged gradient norms on every n-th training batch only;
        each epoch's value is the mean over those batches. Larger values cut
        the cost of `log_grad_norms` and of adaptive weighting.
    check_residuals : bool, optional, default=True
        Run `verify_residuals` before the first training step.
    metadata : dict, optional
        JSON-like data saved in every checkpoint, e.g. the normalization
        metrics, so that a model can be evaluated without the training data
        (see `Analyze.load_checkpoint`).
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        device: torch.device,
        scheduler: Optional[Union[LRScheduler, ReduceLROnPlateau]] = None,
        phys_coll_loader: Optional[torch.utils.data.DataLoader] = None,
        bnd_coll_loader: Optional[torch.utils.data.DataLoader] = None,
        phys_residual: Optional[Callable] = None,
        bnd_residual: Optional[Callable] = None,
        ckpt_path: str = "./logs/best_model.pth",
        phys_weight: float = 0.0,
        bnd_weight: float = 0.0,
        adapt_wts: bool = False,
        theta: Optional[torch.nn.Parameter] = None,
        grad_aggregator: Optional[Callable] = None,
        weight_scheme: Optional[Callable] = None,
        log_grad_norms: bool = False,
        seed: Optional[int] = None,
        ckpt_metric: str = "total",
        wt_update_every: int = 10,
        wt_ema: float = 0.1,
        val_colloc_frac: float = 0.25,
        val_phys_coll_loader: Optional[torch.utils.data.DataLoader] = None,
        val_bnd_coll_loader: Optional[torch.utils.data.DataLoader] = None,
        grad_norm_every: int = 1,
        check_residuals: bool = True,
        metadata: Optional[dict] = None,
    ):
        if ckpt_metric not in CKPT_METRICS:
            raise ValueError(
                f"ckpt_metric must be one of {CKPT_METRICS}; received {ckpt_metric!r}."
            )
        for name, weight in (("phys_weight", phys_weight), ("bnd_weight", bnd_weight)):
            if not (math.isfinite(weight) and weight >= 0):
                raise ValueError(
                    f"{name} must be a finite, non-negative number; received {weight!r}."
                )
        if not 0 < wt_ema <= 1:
            raise ValueError(f"wt_ema must lie in (0, 1]; received {wt_ema!r}.")
        if not 0 < val_colloc_frac <= 1:
            raise ValueError(
                f"val_colloc_frac must lie in (0, 1]; received {val_colloc_frac!r}."
            )
        if metadata is not None and not _json_like(metadata):
            raise ValueError(
                "metadata must be a dict of JSON-like values (dicts with string "
                "keys, lists, str, int, float, bool or None), so that checkpoints "
                "stay loadable with weights_only=True; convert arrays with "
                ".tolist()."
            )
        for kind, weight, residual, loader in (
            ("phys", phys_weight, phys_residual, phys_coll_loader),
            ("bnd", bnd_weight, bnd_residual, bnd_coll_loader),
        ):
            if weight != 0 and (residual is None or loader is None):
                missing = "residual" if residual is None else "coll_loader"
                warnings.warn(
                    f"{kind}_weight={weight} but no {kind}_{missing} was given, so "
                    f"this loss term is disabled.",
                    UserWarning,
                    stacklevel=2,
                )

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.device = device
        self.phys_coll_loader = phys_coll_loader
        self.bnd_coll_loader = bnd_coll_loader
        self.phys_residual = phys_residual
        self.bnd_residual = bnd_residual
        self.ckpt_path = ckpt_path
        self.phys_weight = phys_weight
        self.bnd_weight = bnd_weight
        self.adapt_wts = adapt_wts
        self.theta = theta
        self.grad_aggregator = grad_aggregator
        self.weight_scheme = weight_scheme
        self.log_grad_norms = bool(log_grad_norms or adapt_wts)
        self.ckpt_metric = ckpt_metric
        self.seed = seed if seed is not None else torch.initial_seed()
        self._rng = random.Random(self.seed)
        self._val_subsets: dict = {}
        self.wt_update_every = _require_positive_int(wt_update_every, "wt_update_every")
        self.wt_ema = wt_ema
        self.val_colloc_frac = val_colloc_frac
        self.grad_norm_every = _require_positive_int(grad_norm_every, "grad_norm_every")
        self.check_residuals = check_residuals
        self._residuals_verified = False
        self.metadata = metadata
        self.history: dict = {}

        # Reference weights of the "total" checkpoint metric; the working
        # weights below may be changed by the adaptive weighting.
        self.ref_phys_weight = phys_weight
        self.ref_bnd_weight = bnd_weight
        self.best_epoch: Optional[int] = None

        os.makedirs(os.path.dirname(self.ckpt_path) or ".", exist_ok=True)

        self.grad_data_hist: list = []
        self.grad_phys_hist: list = []
        self.grad_bnd_hist: list = []
        self.all_phys_colloc_batches = _first_tensors(self.phys_coll_loader)
        self.all_bnd_colloc_batches = _first_tensors(self.bnd_coll_loader)
        self.val_phys_colloc_batches = _first_tensors(val_phys_coll_loader)
        self.val_bnd_colloc_batches = _first_tensors(val_bnd_coll_loader)
        self.trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        # Parameters seen by a custom gradient aggregator.
        self.agg_params = list(self.trainable_params)

        # Unknown process parameters estimated in inverse problems are included here so that
        # multi-objective strategies also act on them, whereas the gradient-norm-based
        # adaptive weights are computed with respect to the network parameters only.
        if self.theta is not None and self.theta.requires_grad:
            self.agg_params.append(self.theta)

    # ------------------------------------------------------------- checkpoints
    @property
    def last_ckpt_path(self) -> str:
        """File of the resumable training state written by `adam_step(save_every=...)`."""
        root, ext = os.path.splitext(self.ckpt_path)
        return f"{root}_last{ext or '.pth'}"

    def _save_checkpoint(self):
        """Save the model state and, for inverse problems, the current theta."""
        state: dict[str, object] = {"model_state_dict": self.model.state_dict()}
        if self.theta is not None:
            state["theta"] = self.theta.detach().cpu().clone()
        if self.metadata is not None:
            state["metadata"] = self.metadata
        torch.save(state, self.ckpt_path)

    def _load_checkpoint(self):
        """Restore the model state and, when stored, theta from the checkpoint."""
        ckpt = torch.load(self.ckpt_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["model_state_dict"])
        if self.theta is not None and "theta" in ckpt:
            with torch.no_grad():
                self.theta.copy_(ckpt["theta"])
        return ckpt

    def _loaders_with_generators(self) -> dict:
        loaders = {
            "train": self.train_loader,
            "physics": self.phys_coll_loader,
            "boundary": self.bnd_coll_loader,
        }
        return {
            name: loader
            for name, loader in loaders.items()
            if getattr(loader, "generator", None) is not None
        }

    def _save_last_state(
        self, epoch: int, history: dict, best_total: float, ckpt_saved: bool
    ):
        """Write everything needed to resume `adam_step` after `epoch`."""
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": (
                self.scheduler.state_dict() if self.scheduler is not None else None
            ),
            "theta": None if self.theta is None else self.theta.detach().cpu().clone(),
            "weights": [float(self.phys_weight), float(self.bnd_weight)],
            "best": [best_total, ckpt_saved, self.best_epoch],
            "history": _plain(history),
            "grad_hists": _plain(
                [self.grad_data_hist, self.grad_phys_hist, self.grad_bnd_hist]
            ),
            "rng": {
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else [],
                "trainer": self._rng.getstate(),
                "val_subsets": dict(self._val_subsets),
                "loaders": {
                    name: loader.generator.get_state()
                    for name, loader in self._loaders_with_generators().items()
                },
            },
            "metadata": self.metadata,
        }
        # Write to a temporary file first, so that a job killed while saving
        # leaves the previous state intact.
        tmp = self.last_ckpt_path + ".tmp"
        torch.save(state, tmp)
        os.replace(tmp, self.last_ckpt_path)

    def _load_last_state(self) -> Optional[dict]:
        """Restore the state written by `_save_last_state`, or return None."""
        if not os.path.exists(self.last_ckpt_path):
            warnings.warn(
                f"resume=True but no saved training state was found at "
                f"{self.last_ckpt_path!r}; training starts at epoch 1.",
                UserWarning,
                stacklevel=3,
            )
            return None
        state = torch.load(self.last_ckpt_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state["model_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        if self.scheduler is not None and state["scheduler_state_dict"] is not None:
            self.scheduler.load_state_dict(state["scheduler_state_dict"])
        if self.theta is not None and state["theta"] is not None:
            with torch.no_grad():
                self.theta.copy_(state["theta"])
        self.phys_weight, self.bnd_weight = state["weights"]
        self.best_epoch = state["best"][2]
        self.grad_data_hist, self.grad_phys_hist, self.grad_bnd_hist = state[
            "grad_hists"
        ]
        rng = state["rng"]
        torch.set_rng_state(rng["torch"])
        if rng["cuda"] and torch.cuda.is_available():  # pragma: no cover - GPU only
            torch.cuda.set_rng_state_all(rng["cuda"])
        self._rng.setstate(rng["trainer"])
        self._val_subsets = dict(rng["val_subsets"])
        for name, loader in self._loaders_with_generators().items():
            if name in rng["loaders"]:
                loader.generator.set_state(rng["loaders"][name])
        history = state["history"]
        history["inv_param"] = [
            np.asarray(p, dtype=np.float32) for p in history["inv_param"]
        ]
        state["history"] = history
        return state

    # --------------------------------------------------------------- residuals
    def _residual_on(self, kind: str) -> bool:
        """Whether the "physics" or "boundary" loss has a residual and a non-zero weight."""
        if kind == "physics":
            return self.phys_weight != 0 and self.phys_residual is not None
        return self.bnd_weight != 0 and self.bnd_residual is not None

    def _residual_loss(self, kind: str, x: torch.Tensor) -> torch.Tensor:
        """
        Mean squared "physics" or "boundary" residual on the collocation points `x`.

        The points are moved to the device and detached into a leaf that
        requires gradients, so that a residual may differentiate the network
        output with respect to its inputs.
        """
        residual = self.phys_residual if kind == "physics" else self.bnd_residual
        x = x.detach().to(self.device).requires_grad_(True)
        return torch.mean(residual(x, self.model(x)) ** 2)  # type: ignore[misc]

    def verify_residuals(self) -> None:
        """
        Check every active residual once, on its first collocation batch.

        A residual must return a tensor with one row per collocation point and
        finite values; a residual that does not depend on the network output,
        and so cannot train it, triggers a warning. The check runs with the
        model in evaluation mode and leaves the model, its mode and the random
        number generators unchanged. `adam_step` and `lbfgs_step` call it
        automatically unless `check_residuals=False`.
        """
        for kind, batches in (
            ("physics", self.all_phys_colloc_batches),
            ("boundary", self.all_bnd_colloc_batches),
        ):
            if not self._residual_on(kind) or not batches:
                continue
            residual = self.phys_residual if kind == "physics" else self.bnd_residual
            x = batches[0].detach().to(self.device).requires_grad_(True)
            was_training = self.model.training
            devices = [self.device] if self.device.type == "cuda" else []
            self.model.eval()
            try:
                with torch.random.fork_rng(devices=devices):
                    y = self.model(x)
                    res = residual(x, y)  # type: ignore[misc]
                    name = f"The {kind} residual"
                    if not torch.is_tensor(res):
                        raise TypeError(
                            f"{name} must return a torch.Tensor; it returned "
                            f"{type(res).__name__}."
                        )
                    if res.dim() == 0 or res.shape[0] != x.shape[0]:
                        raise ValueError(
                            f"{name} must return one row per collocation point, "
                            f"i.e. a leading dimension of {x.shape[0]}; it returned "
                            f"shape {tuple(res.shape)}."
                        )
                    if not torch.isfinite(res).all():
                        raise ValueError(
                            f"{name} is not finite on the first collocation batch "
                            "(check units, normalization and divisions); pass "
                            "check_residuals=False to skip this check."
                        )
                    grad = (
                        torch.autograd.grad(res.sum(), y, allow_unused=True)[0]
                        if res.requires_grad
                        else None
                    )
                    if grad is None:
                        warnings.warn(
                            f"{name} does not depend on the network output, so its "
                            "loss cannot train the model.",
                            UserWarning,
                            stacklevel=2,
                        )
            finally:
                self.model.train(was_training)
        self._residuals_verified = True

    def _verify_once(self) -> None:
        if self.check_residuals and not self._residuals_verified:
            self.verify_residuals()

    def grad_norm(self, loss):
        if loss is not None:
            raw = torch.autograd.grad(
                loss, self.trainable_params, retain_graph=True, allow_unused=True
            )
            g_ = [g for g in raw if g is not None]
            norm = (
                torch.norm(torch.stack([g.norm() for g in g_]))
                if g_
                else torch.tensor(0.0, device=self.device)
            )
        else:
            norm = torch.tensor(0.0, device=self.device)
        return norm

    # ---------------------------------------------------------------- training
    def train_epoch(self):
        self.model.train()
        phys_coll_iter = cycle(
            self.phys_coll_loader if self.phys_coll_loader is not None else []
        )
        bnd_coll_iter = cycle(
            self.bnd_coll_loader if self.bnd_coll_loader is not None else []
        )

        # Losses are kept on the device and read back once, at the end of the
        # epoch, rather than synchronizing with `.item()` after every batch.
        sizes, totals, data_losses = [], [], []
        phys_terms, bnd_terms, grad_norms = [], [], []

        for i, (x_batch, y_batch) in enumerate(self.train_loader):
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            batch_size = x_batch.size(0)

            y_pred = self.model(x_batch)
            ld = self.loss_fn(y_pred, y_batch)

            lp, phys_coll_size = None, 0
            if self._residual_on("physics") and self.phys_coll_loader is not None:
                x_phys = next(phys_coll_iter)[0]
                phys_coll_size = x_phys.size(0)
                lp = self._residual_loss("physics", x_phys)

            lb, bnd_coll_size = None, 0
            if self._residual_on("boundary") and self.bnd_coll_loader is not None:
                x_bnd = next(bnd_coll_iter)[0]
                bnd_coll_size = x_bnd.size(0)
                lb = self._residual_loss("boundary", x_bnd)

            if self.log_grad_norms and i % self.grad_norm_every == 0:
                grad_norms.append(
                    (self.grad_norm(ld), self.grad_norm(lp), self.grad_norm(lb))
                )

            if self.grad_aggregator is not None:
                self.optimizer.zero_grad()
                loss = self.grad_aggregator(
                    {"data": ld, "physics": lp, "boundary": lb},
                    {
                        "data": 1.0,
                        "physics": self.phys_weight,
                        "boundary": self.bnd_weight,
                    },
                    self.agg_params,
                )
                self.optimizer.step()
            else:
                loss = (
                    ld
                    + (self.phys_weight * lp if lp is not None else 0.0)
                    + (self.bnd_weight * lb if lb is not None else 0.0)
                )
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            sizes.append(batch_size)
            totals.append(loss.detach() if torch.is_tensor(loss) else loss)
            data_losses.append(ld.detach())
            if lp is not None:
                phys_terms.append((lp.detach(), phys_coll_size))
            if lb is not None:
                bnd_terms.append((lb.detach(), bnd_coll_size))

        n_samples = sum(sizes)
        n_phys_colloc = sum(n for _, n in phys_terms)
        n_bnd_colloc = sum(n for _, n in bnd_terms)

        def weighted_sum(pairs):
            return sum(_value(v) * n for v, n in pairs)

        avg_total = weighted_sum(zip(totals, sizes)) / n_samples if n_samples else 0.0
        avg_data = (
            weighted_sum(zip(data_losses, sizes)) / n_samples if n_samples else 0.0
        )
        avg_phys = weighted_sum(phys_terms) / n_phys_colloc if n_phys_colloc else 0.0
        avg_bnd = weighted_sum(bnd_terms) / n_bnd_colloc if n_bnd_colloc else 0.0

        n_grad = len(grad_norms)
        grads = [[_value(g) for g in triple] for triple in grad_norms]
        avg_grad_d, avg_grad_p, avg_grad_b = (
            (sum(g[k] for g in grads) / n_grad if n_grad else 0.0) for k in range(3)
        )

        return {
            "total_loss": avg_total,
            "data_loss": avg_data,
            "phys_loss": avg_phys,
            "bnd_loss": avg_bnd,
            "avg_grad_d": avg_grad_d,
            "avg_grad_p": avg_grad_p,
            "avg_grad_b": avg_grad_b,
        }

    # -------------------------------------------------------------- validation
    def _colloc_subset(self, batches: list, kind: str, frac: float) -> list:
        """
        Return the collocation batches evaluated during validation and testing.

        A fraction `frac` of the batches (at least one) is drawn once, with
        the trainer's own seeded generator, and reused at every later call, so
        that successive validation losses are computed on the same points and
        are comparable, and so that the selection is reproducible.
        """
        n_total = len(batches)
        frac = max(0.0, min(1.0, float(frac)))
        n_sample = max(1, min(n_total, int(round(n_total * frac))))
        key = (kind, n_sample)
        if key not in self._val_subsets:
            self._val_subsets[key] = sorted(self._rng.sample(range(n_total), n_sample))
        return [batches[i] for i in self._val_subsets[key]]

    def _selection_metric(self, data: float, phys: float, bnd: float) -> float:
        """Validation quantity used for checkpoint selection (see `ckpt_metric`)."""
        if self.ckpt_metric == "data":
            return data
        if self.ckpt_metric == "total_current":
            return data + self.phys_weight * phys + self.bnd_weight * bnd
        return data + self.ref_phys_weight * phys + self.ref_bnd_weight * bnd

    def validate_test(
        self,
        loader: torch.utils.data.DataLoader,
        phys_colloc_frac: Optional[float] = None,
        bnd_colloc_frac: Optional[float] = None,
    ):
        """
        Data, physics and boundary losses of the model on `loader`.

        The residual losses are evaluated on the held-out collocation loaders
        when given, and otherwise on a fixed fraction of the training
        collocation batches (`phys_colloc_frac` / `bnd_colloc_frac`, by
        default `val_colloc_frac`).
        """
        self.model.eval()

        tot_data_loss, n_data = 0.0, 0
        with torch.no_grad():
            for x_batch, y_batch in loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                batch_size = x_batch.size(0)
                y_pred = self.model(x_batch)
                ld = self.loss_fn(y_pred, y_batch)
                tot_data_loss += ld.item() * batch_size
                n_data += batch_size
        avg_data = tot_data_loss / n_data if n_data else 0.0

        avg_phys = self._colloc_loss(
            "physics",
            self.val_phys_colloc_batches or self.all_phys_colloc_batches,
            None if self.val_phys_colloc_batches else phys_colloc_frac,
        )
        avg_bnd = self._colloc_loss(
            "boundary",
            self.val_bnd_colloc_batches or self.all_bnd_colloc_batches,
            None if self.val_bnd_colloc_batches else bnd_colloc_frac,
        )
        return avg_data, avg_phys, avg_bnd

    def _colloc_loss(self, kind: str, batches: list, frac: Optional[float]) -> float:
        """
        Sample-weighted residual loss over the validation subset of `batches`,
        or over all of them when `batches` is a held-out set (`frac` is None).
        """
        if not self._residual_on(kind) or not batches:
            return 0.0
        held_out = (
            self.val_phys_colloc_batches
            if kind == "physics"
            else self.val_bnd_colloc_batches
        )
        if held_out:
            subset = batches
        else:
            frac = self.val_colloc_frac if frac is None else frac
            subset = self._colloc_subset(batches, kind, frac)
        total, n = 0.0, 0
        for x in subset:
            total += self._residual_loss(kind, x).item() * x.size(0)
            n += x.size(0)
        return total / n

    # --------------------------------------------------------------- reporting
    @staticmethod
    def _report(verbose: bool, message: str) -> None:
        """Log `message` to the "pinnse" logger and, when verbose, print it."""
        _LOGGER.info(message)
        if verbose:
            tqdm.write(message)

    @staticmethod
    def _losses_line(prefix: str, total: float, data: float, phys: float, bnd: float):
        return (
            f"{prefix} | Total Loss: {total:.2e}, Data Loss: {data:.2e}, "
            f"Physics Loss: {phys:.2e}, Boundary Loss: {bnd:.2e}"
        )

    # ---------------------------------------------------------------- adam_step
    def adam_step(
        self,
        epochs: int,
        val_every: int,
        verbose: bool = True,
        *,
        progress: Optional[bool] = None,
        save_every: Optional[int] = None,
        resume: bool = False,
    ):
        """
        First-phase training with `optimizer`, validating every `val_every`
        epochs, keeping the best checkpoint and finally restoring and testing it.

        Inputs
        ------
        epochs : int
            Total number of epochs (including those already done when resuming).
        val_every : int
            Validation period, in epochs; must not exceed `epochs`.
        verbose : bool, optional, default=True
            Print the first training epoch, every validation and the test
            losses. The same messages are always logged to the "pinnse"
            logger at INFO level.
        progress : bool, optional, default=None
            Show a progress bar. None shows it in terminals and notebooks and
            hides it when stderr is not a terminal, e.g. in batch jobs.
        save_every : int, optional
            Every `save_every` epochs, and after the last one, write the
            complete training state (model, optimizer, scheduler, weights,
            history, random-generator states) to `last_ckpt_path`.
        resume : bool, optional, default=False
            Continue from the state in `last_ckpt_path`, if present. A run
            whose data and trainer are rebuilt identically continues exactly
            as the uninterrupted run would have.

        Returns
        -------
        history : dict
            Training, validation and test losses, weights, gradient norms and
            inverse parameters (also stored as `self.history`).
        """
        epochs = _require_positive_int(epochs, "epochs")
        val_every = _require_positive_int(val_every, "val_every")
        if val_every > epochs:
            raise ValueError(
                f"val_every={val_every} exceeds epochs={epochs}: no validation "
                "epoch would be reached, so no checkpoint could be selected."
            )
        if save_every is not None:
            save_every = _require_positive_int(save_every, "save_every")
        self._verify_once()

        h: dict = {key: [] for key in _ADAM_LISTS}
        best_total, ckpt_saved, start = float("inf"), False, 1
        self.best_epoch = None
        if resume:
            state = self._load_last_state()
            if state is not None:
                h = state["history"]
                best_total, ckpt_saved, _ = state["best"]
                start = state["epoch"] + 1

        pbar = tqdm(
            range(start, epochs + 1),
            initial=start - 1,
            total=epochs,
            desc="Training",
            unit="epoch",
            disable=_progress_disabled(progress),
        )

        for epoch in pbar:
            loss_stats = self.train_epoch()
            h["loss_t"].append(loss_stats["total_loss"])
            h["loss_d"].append(loss_stats["data_loss"])
            h["loss_p"].append(loss_stats["phys_loss"])
            h["loss_b"].append(loss_stats["bnd_loss"])

            self.grad_data_hist.append(loss_stats["avg_grad_d"])
            self.grad_phys_hist.append(loss_stats["avg_grad_p"])
            self.grad_bnd_hist.append(loss_stats["avg_grad_b"])

            postfix = {
                "L_T": f"{loss_stats['total_loss']:.2e}",
                "L_D": f"{loss_stats['data_loss']:.2e}",
                "L_P": f"{loss_stats['phys_loss']:.2e}",
                "L_B": f"{loss_stats['bnd_loss']:.2e}",
            }
            pbar.set_description(f"Epoch {epoch}/{epochs}")
            pbar.set_postfix(postfix)

            if epoch == 1:
                self._report(
                    verbose,
                    self._losses_line(
                        f"Epoch: {epoch} | Training",
                        loss_stats["total_loss"],
                        loss_stats["data_loss"],
                        loss_stats["phys_loss"],
                        loss_stats["bnd_loss"],
                    ),
                )

            if (
                self.adapt_wts
                and (epoch % self.wt_update_every == 0)
                and (self.phys_weight != 0 or self.bnd_weight != 0)
            ):
                grad_d = loss_stats["avg_grad_d"]
                grad_p = loss_stats["avg_grad_p"]
                grad_b = loss_stats["avg_grad_b"]

                if self.weight_scheme is not None:
                    # User-supplied rule: (grad_d, grad_p, grad_b) -> (wp, wb).
                    # The returned values are targets, smoothed below by the
                    # same exponential moving average as the default rule.
                    target_wp, target_wb = self.weight_scheme(grad_d, grad_p, grad_b)
                else:
                    # Default rule: move each weight toward the ratio of the
                    # data-loss gradient norm to its own; a vanishing gradient
                    # norm leaves that weight unchanged.
                    target_wp = grad_d / (grad_p + 1e-12) if grad_p > 0.0 else None
                    target_wb = grad_d / (grad_b + 1e-12) if grad_b > 0.0 else None

                # A weight of zero disables its loss, so only active weights adapt.
                if self.phys_weight and target_wp is not None:
                    self.phys_weight = (
                        1.0 - self.wt_ema
                    ) * self.phys_weight + self.wt_ema * target_wp

                if self.bnd_weight and target_wb is not None:
                    self.bnd_weight = (
                        1.0 - self.wt_ema
                    ) * self.bnd_weight + self.wt_ema * target_wb

            h["wt_phys"].append(self.phys_weight)
            h["wt_bnd"].append(self.bnd_weight)

            if epoch % val_every == 0:
                val_data, val_phys, val_bnd = self.validate_test(loader=self.val_loader)
                val_total = (
                    val_data + self.phys_weight * val_phys + self.bnd_weight * val_bnd
                )
                h["val_loss_t"].append(val_total)
                h["val_loss_d"].append(val_data)
                h["val_loss_p"].append(val_phys)
                h["val_loss_b"].append(val_bnd)
                h["val_epochs"].append(epoch)
                self._report(
                    verbose,
                    self._losses_line(
                        f"Epoch: {epoch} | Validation",
                        val_total,
                        val_data,
                        val_phys,
                        val_bnd,
                    ),
                )
                # The checkpoint is selected on a quantity whose scale does not
                # change with the adaptive weights (see `ckpt_metric`). A
                # non-finite value never counts as an improvement, so a run
                # that diverges does not overwrite, or later reload, a usable
                # checkpoint.
                metric = self._selection_metric(val_data, val_phys, val_bnd)
                h["val_metric"].append(metric)
                if math.isfinite(metric) and metric < best_total:
                    best_total = metric
                    self._save_checkpoint()
                    ckpt_saved = True
                    self.best_epoch = epoch

                if self.theta is not None:
                    h["inv_param"].append(self.theta.detach().cpu().numpy().copy())

                if self.scheduler is not None and isinstance(
                    self.scheduler, ReduceLROnPlateau
                ):
                    self.scheduler.step(metric)

            if self.scheduler is not None and not isinstance(
                self.scheduler, ReduceLROnPlateau
            ):
                self.scheduler.step()

            if save_every is not None and (epoch % save_every == 0 or epoch == epochs):
                self._save_last_state(epoch, h, best_total, ckpt_saved)
        pbar.close()

        if not ckpt_saved:
            raise RuntimeError(
                "No checkpoint was written during training: every validation "
                f"{self.ckpt_metric!r} loss was non-finite. The model has not been "
                f"restored from {self.ckpt_path!r}."
            )
        self._load_checkpoint()
        test_data, test_phys, test_bnd = self.validate_test(loader=self.test_loader)
        test_total = (
            test_data + self.phys_weight * test_phys + self.bnd_weight * test_bnd
        )
        self._report(
            verbose,
            self._losses_line("Test", test_total, test_data, test_phys, test_bnd),
        )

        history = {
            "loss_t": h["loss_t"],
            "loss_d": h["loss_d"],
            "loss_p": h["loss_p"],
            "loss_b": h["loss_b"],
            "val_loss_t": h["val_loss_t"],
            "val_loss_d": h["val_loss_d"],
            "val_loss_p": h["val_loss_p"],
            "val_loss_b": h["val_loss_b"],
            "val_metric": h["val_metric"],
            "val_epochs": h["val_epochs"],
            "best_epoch": self.best_epoch,
            "wt_phys": h["wt_phys"],
            "wt_bnd": h["wt_bnd"],
            "grad_data_hist": self.grad_data_hist,
            "grad_phys_hist": self.grad_phys_hist,
            "grad_bnd_hist": self.grad_bnd_hist,
            "inv_param": h["inv_param"],
            "test_loss_t": test_total,
            "test_loss_d": test_data,
            "test_loss_p": test_phys,
            "test_loss_b": test_bnd,
        }
        self.history = dict(history)
        return history

    def lbfgs_step(
        self,
        mode: str,
        n_steps: Optional[int] = None,
        verbose: bool = True,
        *,
        progress: Optional[bool] = None,
        N_LBFGS: Optional[int] = None,
    ):
        """
        Second-phase full-batch LBFGS optimization, started from the best
        checkpoint written by `adam_step`. The trainable network parameters
        and, for inverse problems, `theta` are optimized.

        Inputs
        ------
        mode : str
            Optimization mode:
            - "data"     : data loss
            - "physics"  : physics loss
            - "boundary" : boundary loss
            - "overall physics" : physics + boundary losses
            - "combined" : data + physics + boundary losses

        n_steps : int, optional, default=50
            Number of LBFGS outer steps.

        verbose : bool, optional, default=True
            Whether to print the losses after every step (they are always
            logged to the "pinnse" logger at INFO level).

        progress : bool, optional, default=None
            Show a progress bar; None decides as in `adam_step`.

        N_LBFGS : int, optional
            Deprecated name of `n_steps`, accepted as a keyword until pinnse 2.0.

        Returns
        -------
        history_lbfgs : dict
            Dictionary containing LBFGS loss histories. They are also added to
            `self.history` with an ``lbfgs_`` prefix.
        """
        if N_LBFGS is not None:
            if n_steps is not None:
                raise TypeError("Pass n_steps only; N_LBFGS is its deprecated name.")
            warn_renamed("lbfgs_step(N_LBFGS=...)", "n_steps")
            n_steps = N_LBFGS
        n_steps = _require_positive_int(50 if n_steps is None else n_steps, "n_steps")

        mode = mode.lower()
        valid_modes = ("data", "physics", "boundary", "combined", "overall physics")
        if mode not in valid_modes:
            raise ValueError(f"mode must be one of: {valid_modes}")

        # Refuse a mode whose objective is identically zero, which LBFGS would
        # otherwise "optimize" silently without changing any parameter.
        active = {
            "data": mode in ("data", "combined"),
            "physics": mode in ("physics", "combined", "overall physics")
            and len(self.all_phys_colloc_batches) > 0
            and self._residual_on("physics"),
            "boundary": mode in ("boundary", "combined", "overall physics")
            and len(self.all_bnd_colloc_batches) > 0
            and self._residual_on("boundary"),
        }
        if not any(active.values()):
            raise ValueError(
                f"lbfgs_step(mode={mode!r}) has no active loss term: the "
                "physics/boundary terms need a residual, collocation data and "
                "a non-zero weight."
            )

        self._verify_once()

        def gather_full_data(loader):
            xs, ys = [], []
            for x_batch, y_batch in loader:
                xs.append(x_batch)
                ys.append(y_batch)
            X = torch.cat(xs, dim=0).to(self.device)
            Y = torch.cat(ys, dim=0).to(self.device)
            return X, Y

        def gather_full_colloc(batch_list):
            if len(batch_list) == 0:
                return None
            return torch.cat(batch_list, dim=0).to(self.device)

        self.X_all, self.Y_all = gather_full_data(self.train_loader)
        self.X_phys_all = gather_full_colloc(self.all_phys_colloc_batches)
        self.X_bnd_all = gather_full_colloc(self.all_bnd_colloc_batches)

        self._load_checkpoint()
        self.model.train()

        # Trainable network parameters plus, for inverse problems, theta.
        self.lbfgs = torch.optim.LBFGS(
            self.agg_params,
            lr=1.0,
            max_iter=1,
            tolerance_grad=1e-9,
            tolerance_change=1e-9,
            line_search_fn="strong_wolfe",
        )

        history_lbfgs: dict[str, list[float]] = {
            "loss_t": [],
            "loss_d": [],
            "loss_p": [],
            "loss_b": [],
        }
        best_total = float("inf")
        weight = {"physics": self.phys_weight, "boundary": self.bnd_weight}

        def terms(with_grad: bool) -> dict:
            """Active loss terms of the full-batch objective."""
            out = {}
            if active["data"]:
                with torch.set_grad_enabled(with_grad):
                    out["data"] = self.loss_fn(self.model(self.X_all), self.Y_all)
            if active["physics"]:
                out["physics"] = self._residual_loss("physics", self.X_phys_all)
            if active["boundary"]:
                out["boundary"] = self._residual_loss("boundary", self.X_bnd_all)
            return out

        def closure():
            self.lbfgs.zero_grad()
            loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            for name, value in terms(with_grad=True).items():
                loss = loss + (weight[name] * value if name != "data" else value)
            loss.backward()
            return loss

        def evaluate_losses():
            self.model.train()
            values = {k: v.item() for k, v in terms(with_grad=False).items()}
            ld = values.get("data", 0.0)
            lp = values.get("physics", 0.0)
            lb = values.get("boundary", 0.0)

            total = 0.0
            if mode in ("data", "combined"):
                total += ld
            if mode in ("physics", "combined", "overall physics"):
                total += self.phys_weight * lp
            if mode in ("boundary", "combined", "overall physics"):
                total += self.bnd_weight * lb

            return total, ld, lp, lb

        pbar = tqdm(
            range(1, n_steps + 1),
            desc=f"LBFGS [{mode}]",
            unit="step",
            disable=_progress_disabled(progress),
        )

        for step in pbar:
            self.lbfgs.step(closure)

            total, ld, lp, lb = evaluate_losses()

            history_lbfgs["loss_t"].append(total)
            history_lbfgs["loss_d"].append(ld)
            history_lbfgs["loss_p"].append(lp)
            history_lbfgs["loss_b"].append(lb)

            pbar.set_postfix(
                {
                    "L_T": f"{total:.2e}",
                    "L_D": f"{ld:.2e}",
                    "L_P": f"{lp:.2e}",
                    "L_B": f"{lb:.2e}",
                }
            )

            self._report(
                verbose, self._losses_line(f"LBFGS Step: {step}", total, ld, lp, lb)
            )

            if math.isfinite(total) and total < best_total:
                best_total = total
                self._save_checkpoint()

        pbar.close()

        self._load_checkpoint()

        self.history.update({f"lbfgs_{k}": v for k, v in history_lbfgs.items()})
        return history_lbfgs
