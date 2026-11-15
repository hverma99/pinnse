# Migrating from 0.2 to 1.0

Most scripts written for `pinnse` 0.2 run unchanged on 1.0, printing a
`DeprecationWarning` for each renamed argument or class they use. This page
lists the changes that can alter results or raise new errors, and how to keep
the previous behaviour where that makes sense.

## Requirements

- Python 3.11 or later (3.10 is no longer supported).

## Changes that can alter results

### Checkpoint selection with adaptive weights

With `adapt_wts=True` or a `weight_scheme`, the best checkpoint is now selected
on the validation data loss plus the physics and boundary losses weighted by
the **initial** weights, so that every validation is on the same scale. Runs
with fixed weights select exactly the same checkpoint as before. To reproduce
the 0.2 criterion (weights current at each validation):

```python
Training(..., ckpt_metric="total_current")
```

`ckpt_metric="data"` selects on the validation data loss alone.

### Validation collocation points

Validation and test now evaluate the physics and boundary losses on one fixed,
seeded 25% subset of the collocation batches instead of a new random subset
each time. Validation physics and boundary losses therefore differ slightly
from 0.2 runs, and may select a different checkpoint, but are now
reproducible from `torch.manual_seed` (or `Training(seed=...)`). To evaluate
them on held-out points instead, pass `val_phys_coll_loader` /
`val_bnd_coll_loader`.

### Composition inputs

Columns prefixed `Z_`, `X_` or `Y_` are sampled from a Dirichlet distribution
only if they form a group of at least two columns whose rows sum to one. If you
normalized such columns, they are now sampled within their bounds like any
other input, with a warning. Leave mole fractions unnormalized to keep simplex
sampling, or state the groups explicitly:

```python
DataModule(..., simplex_groups=[["Z_A", "Z_B", "Z_C"]])  # or [] to disable
```

### Default boundary value

`bnd_colloc_loader()` now places the boundary at the smallest value of the last
input column. For data normalized with `Normalization.min_max_pfr` (or any
scaling to [-1, 1]) this is -1, as before. For data scaled to [0, 1] it is now
0 instead of -1 (which lay outside the domain). Pass the value explicitly to
choose another location:

```python
data.bnd_colloc_loader(bnd_value=-1.0)
```

### Error metrics

`Analyze.error_metrics` reports NaN, with a `RuntimeWarning`, for any output
group containing non-finite predictions, instead of scoring the finite ones
only. Missing reference values are still skipped.

## Changes that can raise new errors

| Situation | 0.2 behaviour | 1.0 behaviour |
|---|---|---|
| `adam_step(epochs=50, val_every=100)` | trained 50 epochs, then `RuntimeError` | `ValueError` before training |
| `lbfgs_step` mode with no active term (zero weight, no residual or no collocation data) | silently did nothing | `ValueError` |
| `MultiHeadANN(...)(x)` | `dict` of head outputs | one concatenated tensor; use `return_dict=True` for the dict |
| Labeled data with NaN, non-numeric columns or mismatched rows | NaN losses or a late error | `ValueError` in `DataModule` |
| Residual returning a non-tensor, wrong leading dimension or non-finite values | late or silent failure | error before the first step; `check_residuals=False` skips the check |
| Negative or non-finite `phys_weight` / `bnd_weight` | accepted | `ValueError` |

## Renamed APIs

The old names keep working until 2.0 with a `DeprecationWarning`.

| 0.2 | 1.0 |
|---|---|
| `DataModule(boundry_coll_batch_size=...)` | `DataModule(boundary_coll_batch_size=...)` |
| `dm.boundry_coll_data_size`, `dm.boundry_coll_batch_size` | `dm.boundary_coll_data_size`, `dm.boundary_coll_batch_size` |
| `Fourier_ANN` | `FourierANN` |
| `Denormalization.mean_abs` | `Denormalization.mean_norm` |
| `Denormalization.z_norm` | `Denormalization.z_score` |
| `training.lbfgs_step(mode, N_LBFGS=150)` | `training.lbfgs_step(mode, n_steps=150)` |
| `from pinnse import DataLoader` | `from torch.utils.data import DataLoader` |

To find remaining uses, run your script with deprecation warnings turned into
errors:

```bash
python -W error::DeprecationWarning main.py
```

## New options worth adopting

- `pinnse.seed_everything(seed)` at the top of a script seeds Python, NumPy
  and PyTorch together.
- `training.adam_step(..., save_every=500, resume=True)` makes long cluster
  jobs restartable: resubmit the same script after a time-out and training
  continues where it stopped.
- `Training(..., metadata={"I_S_metrics": I_S_metrics, "D_S_metrics": D_S_metrics})`
  stores the normalization metrics in the checkpoint;
  `Analyze.load_checkpoint(path, model)` returns them, so evaluation scripts no
  longer need the training data.
- `Plotter(history)` reads the recorded validation epochs; `val_every` is only
  needed for histories saved by 0.2.
