# Normalization and noise

## Normalization strategies

{class}`~pinnse.Normalization` offers several methods. The right choice depends
on the data distribution and the formulation.
{class}`~pinnse.Denormalization` provides the inverse transforms that recover
dimensional predictions from normalized model outputs.

| Method | Range | Best for |
|---|---|---|
| `Normalization.min_max()` | [0, 1] | General use when output magnitudes are similar. |
| `Normalization.scale_centered()` | [−1, 1] | Variables with both positive and negative ranges; PFR formulations where centered scaling improves gradient flow. |
| `Normalization.max_abs()` | [−1, 1] | Data that is already centered around zero. |
| `Normalization.mean_norm()` | Centered | When the mean is a meaningful reference point. |
| `Normalization.z_score()` | Standardized | When standard-deviation scaling is preferred (variables with near-normal distributions, for example). |
| `Normalization.min_max_pfr()` | [−1, 1] | **PFR-specific.** Groups variables by physical role (flowrates, extents, conversions, operating conditions) and applies global or local centered scaling. Supports the `"EFM"`, `"ERM"`, and `"CM"` formulations. |

:::{tip}
For the PFR examples, use `min_max_pfr()`. It gives variables in the same
physical group (all inlet flowrates, for example) a common scale, which helps
the network learn balanced representations. For flash or other general systems,
`min_max()` or `scale_centered()` are good defaults.
:::

## Measurement noise

{meth}`Noise.gaussian <pinnse.Noise.gaussian>` adds zero-mean Gaussian noise
to the labelled outputs, so a clean simulated dataset emulates plant or
experimental measurements. The collocation datasets are unaffected, since they
carry no measured values. Three noise models are available:

| `mode` | Standard deviation | Typical use |
|---|---|---|
| `"relative"` (default) | `noise_level × std(column)` | The same noise for every sample, scaled by the spread of each variable. |
| `"proportional"` | `noise_level × \|value\|` | Noise that grows with the measured magnitude, typical of flow and composition sensors. |
| `"absolute"` | `noise_level` | Noise in the units of the column. |

```python
from pinnse import Noise

D_S_noisy, noise_metrics = Noise.gaussian(
    data=D_S_data,  # labelled outputs (DataFrame)
    noise_level=0.05,  # a float, or a dict {column: level}
    mode="relative",
    clip_min=0.0,  # keep flowrates non-negative
    random_state=42,
)
```

`noise_cols` restricts the noise to selected columns. `noise_metrics` records
the mode, level and realized standard deviation of each column. Compute the
normalization metrics from the noisy data, since that is the data available in
practice (see `examples/isopfr/efm.noise`).

## Other utilities

| Class | Description |
|---|---|
| {class}`~pinnse.Analyze` | Load a trained model checkpoint, evaluate predictions, and compute error metrics (MAE, RMSE, R²). |
| {class}`~pinnse.Save` | Write training histories or results to Excel (`.xlsx`) or CSV files. |
