# Framework overview

## Why physics-informed surrogate models?

First-principles models in chemical engineering (reactor balances, phase
equilibrium, heat transfer) are often nonlinear, tightly coupled, and expensive
to solve repeatedly in optimization, control, or uncertainty-quantification
loops.

Purely data-driven surrogates approximate these models cheaply, but they can
violate conservation laws, equilibrium relationships, or boundary conditions,
especially when extrapolating beyond the training data.

Physics-informed neural networks address this by adding soft physical
constraints to the training loss. During training, the network is penalized for
mismatch with labelled data and also for violating the governing equations at
collocation points sampled across the input domain. The resulting surrogate is:

- **Data-efficient:** physical constraints reduce the labelled data needed.
- **Physically consistent:** conservation laws and boundary conditions are
  respected.
- **Differentiable:** the surrogate and its gradients are available through
  automatic differentiation, which supports sensitivity analysis and
  gradient-based optimization.

## The composite loss

`pinnse` trains a surrogate by minimizing a loss with three components:

- **Data loss** fits the network predictions to labelled input/output samples
  from a process simulator or experiment.
- **Physics loss** penalizes violations of the governing equations (material
  and energy balances, for example) at collocation points sampled across the
  input domain.
- **Boundary loss** enforces boundary or initial conditions, such as the inlet
  conditions of a PFR at reactor volume $V = 0$.

$$
\mathcal{L} = \mathcal{L}_\text{data} + w_\text{phys}\,\mathcal{L}_\text{phys}
            + w_\text{bnd}\,\mathcal{L}_\text{bnd}
$$

The weights $w_\text{phys}$ and $w_\text{bnd}$ can be fixed, or updated during
training by the built-in gradient-norm balancing or by a rule you supply. A
gradient aggregator such as {class}`~pinnse.PCGrad` can instead combine the
individual loss gradients directly (see
[](#custom-loss-weighting-and-gradient-aggregation)).

## Workflow

Each example follows the same workflow, organized across a small set of files:

| Stage | What happens | File |
|---|---|---|
| **Setup** | Define process variables, operating bounds, architecture, optimizer, loss weights | `main.py` |
| **Data generation** | Generate labelled data from Aspen Plus, SciPy ODE solvers, or other process models | `data_gen.py` |
| **Data loading** | Split labelled data; construct supervised, physics-collocation, and boundary-collocation loaders | {mod}`pinnse.data` |
| **Architecture** | Build the differentiable neural-network surrogate | {mod}`pinnse.PINNs` |
| **Physics residuals** | Encode governing equations in residual form | `phys_res.py` |
| **Training** | Optimize with Adam (and optional L-BFGS refinement); checkpoint the best model | {mod}`pinnse.train` |
| **Evaluation** | Load the trained model, denormalize outputs, compute error metrics, generate figures | `check.py` |

## Network architectures

| Class | Description | When to use |
|---|---|---|
| {class}`~pinnse.ANN` | Fully connected feedforward network with Xavier-uniform weights and zero biases. | General-purpose PINN surrogate. The default choice. |
| {class}`~pinnse.SANN` | Same as `ANN`, with `Softplus` on the output layer. | Outputs that must be non-negative (concentrations, flowrates). |
| {class}`~pinnse.MultiHeadANN` | Shared trunk with multiple independent output heads. | Outputs that group naturally (compositions vs. temperature) and benefit from a shared representation with separate specialization. |
| {class}`~pinnse.Fourier_ANN` | Feedforward network with a sinusoidal Fourier-feature embedding of the last input coordinate. | Targets that vary rapidly or at multiple scales along one input (reactor length, for example). |
| {class}`~pinnse.RecurrentANN` | Recurrent sequence-to-sequence network (RNN, GRU or LSTM) with an output head at each time step. | Dynamic processes. See {doc}`dynamic`. |

## Training

{class}`~pinnse.Training` manages the training loop.

| Method | Purpose |
|---|---|
| `adam_step(epochs, val_every)` | First-order training with Adam. Validates periodically and checkpoints the best model. |
| `lbfgs_step(mode, N_LBFGS)` | Second-order refinement with L-BFGS (full batch, strong Wolfe line search). Modes: `"data"`, `"physics"`, `"boundary"`, `"combined"`, `"overall physics"`. |
| `validate_test(loader)` | Evaluate data, physics, and boundary losses on a loader. |

Key constructor parameters:

| Parameter | Description |
|---|---|
| `phys_weight`, `bnd_weight` | Scalar weights for the physics and boundary losses. Set to `0.0` to disable a loss. |
| `adapt_wts` | If `True`, adjusts `phys_weight` and `bnd_weight` every 10 epochs by gradient-norm balancing. Requires both weights to be nonzero. |
| `theta` | An `nn.Parameter` of trainable physical parameters (activation energies, for example) for inverse problems, optimized alongside the network weights. |
| `weight_scheme` | Optional callable `(grad_d, grad_p, grad_b) -> (w_phys, w_bnd)` that replaces the built-in adaptive-weight rule. Used only when `adapt_wts=True` and both weights are nonzero. |
| `grad_aggregator` | Optional callable that combines the per-loss gradients itself, replacing the default weighted-sum update. {class}`~pinnse.PCGrad` is provided. |

### Custom loss weighting and gradient aggregation

By default, with `adapt_wts=True`, every 10 epochs each weight moves toward the
ratio of the data-loss gradient norm to its own loss's gradient norm, smoothed
by an exponential moving average (factor 0.1). To use a different rule, pass
`weight_scheme`. It receives the average gradient norms of the data, physics
and boundary losses over the last epoch, and returns target physics and
boundary weights, which are smoothed by the same moving average:

```python
def capped_ratio(grad_d, grad_p, grad_b):
    # Gradient-norm balancing, capped at 100
    w_phys = min(grad_d / (grad_p + 1e-12), 100.0)
    w_bnd = min(grad_d / (grad_b + 1e-12), 100.0)
    return w_phys, w_bnd


training = Training(
    ..., phys_weight=1.0, bnd_weight=1.0, adapt_wts=True, weight_scheme=capped_ratio
)
```

Adaptive weighting, built-in or custom, runs only when both `phys_weight` and
`bnd_weight` are nonzero. With a single residual loss (for example
`bnd_weight=0.0`), the weights stay fixed. Guard divisions against zero
gradient norms, as above.

Gradient-norm weighting balances only the *magnitudes* of the loss gradients.
{class}`~pinnse.PCGrad` (Yu et al., 2020) also resolves conflicts in their
*directions*: whenever two loss gradients have a negative inner product, each
is projected onto the normal plane of the other before they are summed.

```python
from pinnse import Training, PCGrad

training = Training(..., phys_weight=1.0, bnd_weight=1.0, grad_aggregator=PCGrad())
```

With `PCGrad(apply_weights=True)` (the default), each loss is scaled by its
current weight before its gradient is computed. With `apply_weights=False`, the
raw losses are used, and nonzero weights only switch losses on. In inverse
problems, the aggregated gradient is applied to `theta` as well as to the
network parameters.

Any other multi-objective strategy can be supplied as `grad_aggregator` without
modifying `pinnse`. It is called on every batch as
`aggregator(losses, weights, params)`, where `losses` and `weights` are
dictionaries keyed by `"data"`, `"physics"` and `"boundary"` (an absent loss is
`None`). It must set `p.grad` for every tensor in `params` and return the scalar
total loss for logging.

## Visualization

{class}`~pinnse.Plotter` generates figures from the training-history
dictionary returned by `Training`.

| Method | What it plots |
|---|---|
| `plot_all_train_losses()` | Total, data, physics, and boundary training losses |
| `plot_all_val_losses()` | Validation losses |
| `plot_weights()` | Adaptive physics and boundary weight trajectories |
| `plot_gradient_history()` | Gradient-norm histories for each loss component |
| `plot_inverse_params()` | Inverse-parameter convergence trajectories |
| `plot_everything()` | All of the above, saved to a directory |

All plots support exponential-moving-average smoothing and customizable font,
DPI, and figure-size settings.
