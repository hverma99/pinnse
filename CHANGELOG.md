# Changelog

All notable changes to `pinnse` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic versioning](https://semver.org/).

## [Unreleased]

### Fixed

- Type annotations: `DataModule.phys_colloc_loader` takes `shuffle: bool` and
  `alpha: float` (`None` was never a valid value).

### Added

- The package ships a `py.typed` marker, so type checkers use its annotations.
- Static type checking with mypy (`nox -s typecheck`), run in CI.

## [1.0.0] - 11/16/2026

The first stable release. It fixes how checkpoints are selected and how
collocation points are sampled, makes runs reproducible and resumable, and
settles the public API. Upgrading from 0.2: see the
[migration guide](docs/migration.md).

### Fixed

- **Checkpoint selection with adaptive weights.** The best checkpoint was
  chosen by comparing validation totals computed with the weights current at
  each validation, so with `adapt_wts=True` or a `weight_scheme` epochs were
  compared on different scales and an early, worse model could be restored
  (12x worse validation data loss in the reproduction). The selection metric
  now uses the initial weights by default (`ckpt_metric`), and the same metric
  drives `ReduceLROnPlateau`. Unchanged for fixed weights.
- **Reproducibility of validation.** Validation and test evaluated a fresh
  random 25% of the collocation batches each time, drawn from Python's global
  `random` module, so runs seeded with `torch.manual_seed` were not
  reproducible and successive validations used different points. The subset
  is now drawn once from the trainer's own seeded generator (`seed`).
- **Composition sampling.** Input columns prefixed `Z_`, `X_` or `Y_` were
  always sampled on the simplex. Normalized compositions then summed to
  0.74-0.93 in physical units, and a single prefixed input (e.g. a conversion)
  became the constant 1.0. Groups are now validated (at least two columns,
  rows summing to one) and can be given explicitly with `simplex_groups`.
- **Boundary value.** `bnd_colloc_loader` fixed the boundary column at -1.0,
  outside the domain of data scaled to [0, 1]. The default is now the smallest
  value of that column in the data (still -1 for the centered normalization);
  explicit values outside the data range warn.
- **Error metrics.** `Analyze.error_metrics` silently dropped non-finite
  predictions, reporting a model that predicted NaN for 9 of 10 points with
  MAE = 0. Non-finite predictions now make the metrics NaN, with a warning.
- **`MultiHeadANN` with `Training`.** Its dictionary output crashed
  `Training` and `Analyze`; it now returns one concatenated tensor by default.
- `adam_step` validated `epochs`/`val_every` only after training every epoch
  (or failed with `ZeroDivisionError` for `val_every=0`); both are now checked
  up front.
- `lbfgs_step` silently did nothing for a mode without an active loss term; it
  now raises `ValueError`.
- `Normalization.max_abs` returned NaN for an all-zero column.
- Shared ("global") centered-normalization metrics were rounded to float32.
- `SequenceDataModule(n_segments=0)` and degenerate layer sizes failed with
  `ZeroDivisionError`/`IndexError` instead of a clear message.
- Checkpoints are loaded with `weights_only=True` on every supported PyTorch.
- The module docstrings of `pinnse.train` and `pinnse.PINNs` were not the
  modules' `__doc__`.
- The quickstart walkthrough used variables it never defined.

### Changed

- **Breaking:** `MultiHeadANN` concatenates its heads in the order of
  `head_dims`; pass `return_dict=True` for the previous dictionary output.
- **Breaking:** `DataModule.bnd_colloc_loader(bnd_value=None)` defaults to the
  smallest value of the boundary column instead of -1.0.
- **Breaking:** inputs prefixed `Z_`/`X_`/`Y_` whose rows do not sum to one, or
  that form a one-column group, are sampled by Latin hypercube sampling (with a
  warning) instead of on the simplex.
- **Breaking:** `Analyze.error_metrics` reports NaN, with a `RuntimeWarning`,
  for a group with non-finite predictions.
- **Breaking:** `adam_step` raises `ValueError` before training when
  `val_every > epochs` (previously `RuntimeError` after training), and
  `lbfgs_step` raises `ValueError` for a mode without an active loss term.
- **Breaking:** Python 3.11 or later is required (3.10 reached end of life in
  October 2026).
- The best checkpoint is selected on the initial-weight total by default; use
  `Training(ckpt_metric="total_current")` for the previous criterion.
- `DataModule` and `Training` check their inputs and fail fast with
  descriptive errors: mismatched row counts, non-numeric or non-finite data,
  negative weights, residuals of the wrong shape or with non-finite values.
- Progress messages are logged to the `"pinnse"` logger and printed without
  terminal colour codes; progress bars are hidden when stderr is not a
  terminal (e.g. SLURM logs).
- Default split fractions (`test_frac`, `val_frac`) are an explicit 0.25.
- Training losses are read back once per epoch rather than after every batch.
- Declared minimum dependency versions: numpy 1.24, pandas 2.0,
  scikit-learn 1.2, tqdm 4.65, openpyxl 3.1 (scipy 1.15, torch 2.0 and
  matplotlib 3.8 as before). CI runs the test suite with every dependency at
  exactly these versions on Python 3.11 (`nox -s minimums`).
- Internal simplification without behaviour change: shared builders for the
  networks, one residual-loss path in `Training`, table-driven plotting, and
  shared split/loader helpers (results verified bit-for-bit).
- The six isothermal PFR examples share their reactor models, data
  generation, residuals, training pipeline and evaluation in
  `examples/isopfr/common/`; each case folder keeps only its own settings and
  still runs from inside the folder (losses, weights and error metrics
  verified bit-for-bit).

### Added

- Resumable training: `adam_step(save_every=..., resume=True)` with the full
  state in `Training.last_ckpt_path`; an interrupted and resumed run
  reproduces the uninterrupted one exactly.
- `pinnse.seed_everything(seed, deterministic=False)`, `Training(seed=...)`
  and `DataModule(shuffle_seed=...)` / `SequenceDataModule(shuffle_seed=...)`.
- `Training(ckpt_metric=...)`, `wt_update_every`, `wt_ema`,
  `val_colloc_frac`, held-out `val_phys_coll_loader` / `val_bnd_coll_loader`,
  `grad_norm_every`, `check_residuals` and `metadata` options.
- `Training.verify_residuals()`, run automatically before training.
- `adam_step(progress=...)` and `lbfgs_step(progress=...)`.
- Histories record `val_epochs`, `val_metric`, `best_epoch` and the final test
  losses; `Training.history` also collects the L-BFGS histories. `Plotter`
  reads `val_epochs`, so `val_every` is no longer needed.
- `Analyze.load_checkpoint()` and `Analyze.load_model()` for any architecture,
  returning the stored `theta` and metadata.
- `Normalization.centered()`, the general centered scaling behind
  `min_max_pfr`.
- `DataModule(simplex_groups=...)` and a full `DataModule` docstring.
- `examples/smoke.py`, which runs an example script with training cut to a few
  epochs to check that it still works end to end. CI runs every example this
  way (`nox -s examples`).

### Deprecated

These names still work and emit a `DeprecationWarning`; they will be removed in
pinnse 2.0.

- `DataModule(boundry_coll_batch_size=...)` and the attributes
  `boundry_coll_data_size` / `boundry_coll_batch_size`: use
  `boundary_coll_batch_size` / `boundary_coll_data_size`.
- `Fourier_ANN`: use `FourierANN`.
- `Denormalization.mean_abs` and `Denormalization.z_norm`: use
  `Denormalization.mean_norm` and `Denormalization.z_score`.
- `Training.lbfgs_step(N_LBFGS=...)`: use `n_steps`.
- `pinnse.DataLoader`: import `DataLoader` from `torch.utils.data`.

### Removed

- Support for Python 3.10.
- The `examples` extra, which only listed core dependencies (pip only warns
  about the unknown extra).
- `pinnse.DataLoader` and `Fourier_ANN` from `pinnse.__all__` (both remain
  importable until 2.0).

## Earlier releases

See the [GitHub releases](https://github.com/hverma99/pinnse/releases) for
versions 0.1.0 to 0.2.1.
