# pinnse: Physics-informed neural networks for process systems engineering

![python](https://img.shields.io/badge/python-%E2%89%A53.10-blue)
![pytorch](https://img.shields.io/badge/PyTorch-enabled-ee4c2c)
![license](https://img.shields.io/badge/license-MIT-green)

`pinnse` is a modular PyTorch framework for constructing physics-informed neural-network (PINN) surrogate models for chemical process modelling, simulation and process systems engineering. The package separates process-specific definitions—operating bounds, governing equations, input--output representations and residual functions—from reusable infrastructure for data handling, normalization, neural-network construction, training, validation, testing and visualization.

The repository includes representative examples for nonideal flash separation, isothermal plug-flow reactors under multiple surrogate formulations, inverse PINNs for parameter estimation and a nonisothermal plug-flow reactor with coupled mass and energy balances.

<p align="center">
  <img src="docs/framework_overview.png" alt="pinnse framework overview" width="900">
</p>

The workflow follows a simple principle: define the process physics once, expose it through residual functions, and train a neural surrogate against both labeled data and physical constraints. The resulting models can be used as differentiable, process-aware surrogates for simulation, design studies and downstream optimization workflows.

---

## Contents

- [Why pinnse?](#why-pinnse)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Framework workflow](#framework-workflow)
- [Core package API](#core-package-api)
- [Example case studies](#example-case-studies)
- [Repository structure](#repository-structure)
- [Extending pinnse to a new process](#extending-pinnse-to-a-new-process)
- [Outputs](#outputs)
- [Citation](#citation)
- [License](#license)

---

## Why pinnse?

First-principles models in chemical engineering are often nonlinear, coupled and expensive to solve repeatedly. Surrogate models can accelerate repeated evaluations, but purely data-driven surrogates may violate conservation laws, equilibrium relationships or boundary conditions. `pinnse` addresses this by combining labeled simulator or solver data with residual losses derived from governing equations.

The framework is designed for process systems engineering applications in which the same unit operation may admit multiple surrogate formulations. For example, a plug-flow reactor can be represented using effluent flows, reaction extents or conversions. `pinnse` keeps these formulation-specific choices local to the case-study directory while reusing the same training, data and plotting infrastructure.

---

## Installation

Clone the repository and install it in editable mode:

```bash
git clone <repository-url>
cd pinnse
python -m pip install -e .
```

A clean environment is recommended:

```bash
conda create -n pinnse python=3.11 -y
conda activate pinnse
python -m pip install -e .
```

The core package depends on `numpy`, `scipy`, `torch`, `pandas` and `tqdm`. The included examples and plotting utilities also use:

```bash
python -m pip install scikit-learn matplotlib openpyxl
```

Flash data generation through Aspen Plus requires Windows, Aspen Plus and COM automation:

```bash
python -m pip install pywin32
```

Training from the supplied `.xlsx` datasets does not require Aspen Plus.

---

## Quick start

The isothermal PFR examples are the most direct entry point because they do not require Aspen Plus.

```bash
cd isopfr/efm
python main.py
python check.py
```

A standard run performs the following steps:

1. loads `I_S_data.xlsx` and `D_S_data.xlsx`,
2. normalizes the input and output spaces,
3. creates supervised, physics-collocation and boundary-collocation loaders,
4. builds an `ANN` model,
5. trains with data, physics and boundary losses,
6. saves the best checkpoint to `logs/best_model.pth`,
7. writes loss histories to `logs/`, and
8. generates training and comparison figures.

Flash examples can be trained from the provided data:

```bash
cd flash/case2
python main.py
```

Regenerating flash data or running Aspen-based comparison scripts requires the Windows/Aspen setup described above.

---

## Framework workflow

`pinnse` implements the composite PINN objective

```math
\mathcal{L}_{\mathrm{total}}
= \mathcal{L}_{\mathrm{data}}
+ \lambda_P \mathcal{L}_{\mathrm{physics}}
+ \lambda_B \mathcal{L}_{\mathrm{boundary}},
```

where the data loss fits labeled examples, the physics loss penalizes residual violations at collocation points, and the boundary loss enforces initial or boundary conditions.

The framework follows the sequence shown in the figure.

| Stage | Role in the workflow | Typical file |
|---|---|---|
| Framework initialization | Specifies process variables, operating bounds, governing equations, network architecture, dataset sizes, optimizers and loss weights. | `main.py` |
| Data generation and sampling | Generates labeled input--output pairs from Aspen Plus, SciPy solvers or other first-principles models. | `data_gen.py` |
| Data loading | Splits labeled data and constructs supervised, physics-collocation and boundary-collocation loaders. | `pinnse/data.py` |
| Neural architecture | Defines the differentiable surrogate model. | `pinnse/PINNs.py` |
| Physics residual formulation | Maps normalized variables to physical variables and returns residual tensors. | `phys_res.py` |
| Training, validation and testing | Optimizes the network, evaluates validation loss, checkpoints the best model and stores histories. | `pinnse/train.py` |
| Post-processing | Loads the trained model, evaluates predictions and generates figures/metrics. | `check.py`, `pinnse/utils.py`, `pinnse/plots.py` |

---

## Core package API

The reusable package is contained in `pinnse/`. It provides the general machinery used by all case studies.

### `pinnse/PINNs.py`: neural-network architectures

This module defines the PyTorch models used as PINN surrogates. All architectures share the standard PyTorch interface: instantiate a model, move it to a device and call `model(x)`.

| Class | Functionality | Typical use |
|---|---|---|
| `ANN(layer_size, activation)` | Fully connected feedforward network with Xavier-uniform weight initialization and zero biases. | Default model for flash and isothermal PFR examples. |
| `SANN(layer_size, activation)` | Same as `ANN`, but applies a `Softplus` output activation. | Useful when outputs should remain non-negative, as in the nonisothermal PFR example. |
| `BranchedANN(in_dim, trunk_layers, head_dims, activation)` | Shared-trunk network with separate output heads. Returns a dictionary of head predictions. | Multi-output systems with related but partially distinct output blocks. |
| `Fourier_ANN(layer_size, activation, fourier_levels, positive_output)` | Augments inputs with Fourier features before feedforward regression; optionally enforces positive outputs. | Mappings with high-frequency or strongly varying structure. |

Minimal model construction:

```python
import torch.nn as nn
from pinnse import ANN

layer_size = [dim_in] + [64] * 6 + [dim_out]
model = ANN(layer_size=layer_size, activation=nn.Tanh)
```

### `pinnse/data.py`: supervised and collocation data loaders

`DataModule` converts normalized `pandas.DataFrame` objects into PyTorch `DataLoader` objects.

```python
from pinnse import DataModule

data = DataModule(
    I_S_data=norm_I_S_data,
    D_S_data=norm_D_S_data,
    labeled_data_batch_size=500,
    physics_coll_data_size=20000,
    physics_coll_batch_size=500,
    boundary_coll_data_size=5000,
    boundry_coll_batch_size=1000,
    test_frac=0.1,
    val_frac=0.1,
)
```

The current API uses the argument name `boundry_coll_batch_size`.

| Method | Functionality |
|---|---|
| `labeled_data_loader()` | Splits `I_S_data` and `D_S_data` into train, validation and test loaders. It also stores lower and upper input bounds for subsequent collocation sampling. |
| `phys_colloc_loader(shuffle=True, alpha=1.0, drop_last=False)` | Samples physics-collocation inputs over the input domain. Ordinary variables are sampled by Latin hypercube sampling; columns beginning with `Z_`, `X_` or `Y_` are sampled with Dirichlet distributions so that composition groups sum to one. |
| `bnd_colloc_loader(shuffle=True, drop_last=False, bnd_value=-1.0)` | Samples boundary-collocation inputs by fixing the last input column to `bnd_value` and sampling all preceding columns over their admissible range. This is convenient for PFR inlet boundaries when the last input is `V`. |
| `inspect_loader(name, loader)` | Prints tensor shapes and batch counts for debugging. |
| `save_loaders(loader, filename)` | Exports loader contents to Excel for inspection. |

### `pinnse/train.py`: training, validation and checkpointing

`Training` combines supervised data loss, optional physics residual loss and optional boundary residual loss.

```python
from pinnse import Training

trainer = Training(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    optimizer=optimizer,
    loss_fn=loss_fn,
    device=device,
    phys_coll_loader=phys_coll_loader,
    bnd_coll_loader=bnd_coll_loader,
    phys_residual=physics,
    bnd_residual=boundary,
    ckpt_path="./logs/best_model.pth",
    phys_weight=1.0,
    bnd_weight=1.0,
    adapt_wts=False,
)

history = trainer.adam_step(epochs=75000, val_every=100)
```

| Method | Functionality |
|---|---|
| `train_epoch()` | Performs one Adam-style training epoch over labeled batches while cycling through physics and boundary collocation batches when provided. |
| `validate_test(loader, phys_colloc_frac=0.25, bnd_colloc_frac=0.25)` | Evaluates data, physics and boundary losses on validation or test data. A fraction of stored collocation batches is sampled for efficiency. |
| `adam_step(epochs, val_every, verbose=True)` | Main optimization routine. It logs training and validation losses, gradient norms, adaptive weights and inverse parameters; it also checkpoints the model with the lowest validation total loss. |
| `lbfgs_step(mode, N_LBFGS=50, verbose=True)` | Optional full-batch second-stage optimization. Supported modes are `data`, `physics`, `boundary`, `overall physics` and `combined`. |

When `adapt_wts=True`, the trainer updates `phys_weight` and `bnd_weight` by gradient-norm matching with exponential moving averaging. For inverse problems, a trainable parameter `theta` can be passed to `Training`; its trajectory is stored in `history["inv_param"]`.

The returned history dictionary includes:

```text
loss_t, loss_d, loss_p, loss_b
val_loss_t, val_loss_d, val_loss_p, val_loss_b
wt_phys, wt_bnd
grad_data_hist, grad_phys_hist, grad_bnd_hist
inv_param
```

### `pinnse/utils.py`: normalization, evaluation and persistence

This module provides reusable utilities for preprocessing, model evaluation and saving results.

#### `Normalization`

| Function | Functionality |
|---|---|
| `min_max(data, normalize_cols=None)` | Applies `[0, 1]` min--max scaling to selected columns. Columns not listed in `normalize_cols` are left unchanged. |
| `min_max_defined_metrics(data, metrics)` | Applies stored min--max metrics to new data. |
| `scale_centered(data, col, cmin, cmax)` | Scales a column to `[-1, 1]`. |
| `apply_global(...)` | Applies centered scaling to a group of columns using a shared global minimum and maximum. |
| `apply_local(...)` | Applies centered scaling column by column. |
| `min_max_pfr(I_S_data, D_S_data, formulation, species, N_reaction=3)` | Formulation-aware centered scaling for PFR examples. Supports `EFM`, `ERM` and `CM`. |
| `scale_centered_defined_metrics(I_S, I_S_metrics)` | Applies stored centered-scaling metrics to new PFR inputs. |
| `max_abs(...)`, `mean_norm(...)`, `z_score(...)` | Additional scaling options for user-defined workflows. |

`min_max_pfr` is tailored to reactor formulations. In EFM, inlet and outlet flows share a common scale. In ERM and CM, inlet flows and reaction variables are scaled according to their formulation-specific structure.

#### `Denormalization`

| Function | Functionality |
|---|---|
| `min_max(data_norm, metrics)` | Inverts `[0, 1]` min--max scaling. |
| `min_max_col(x_norm, col, metrics)` | Denormalizes one variable and leaves it unchanged if no metric is available. This is useful inside residual functions with partially normalized data. |
| `min_max_pfr(X, norm_metrics, keys)` | Converts centered PFR variables from `[-1, 1]` to dimensional values and returns the corresponding ranges. |
| `max_abs(...)`, `mean_abs(...)`, `z_norm(...)` | Invert the corresponding normalization methods. |

#### `Analyze`

| Function | Functionality |
|---|---|
| `load_ann_pinn(dim_in, dim_out, depth, width, activation, ckpt_path, device)` | Reconstructs an `ANN` and loads a saved checkpoint. |
| `pinn_eval(...)` | Evaluates a trained model for standard min--max normalized workflows. |
| `pinn_eval_pfr(...)` | Evaluates a trained model for centered PFR workflows. |
| `error_metrics(df_true, df_pred, groups)` | Computes grouped MAE, RMSE and R² values. |

#### `Save`

| Function | Functionality |
|---|---|
| `excel(history, foldername="logs", filename="training_history.xlsx")` | Writes each history entry to a sheet in an Excel workbook. |
| `csv(history, foldername="logs", filename="csv_files")` | Writes each history entry to a separate CSV file. |

### `pinnse/plots.py`: training diagnostics

`Plotter` converts saved training histories into publication-quality diagnostic figures.

```python
from pinnse import Plotter

plots = Plotter(history=history, val_every=100)
plots.plot_everything(savepath="figures", scale=1000)
```

| Method | Functionality |
|---|---|
| `plot_individual_loss(...)` | Plots one training or validation loss history, with exponential moving-average smoothing. |
| `plot_all_train_losses(...)` | Plots total, data, physics and boundary training losses together. |
| `plot_all_val_losses(...)` | Plots total, data, physics and boundary validation losses together. |
| `plot_weights(...)` | Plots adaptive physics and boundary weights. |
| `plot_gradient_history(...)` | Plots gradient-norm histories for data, physics or boundary terms. |
| `plot_inverse_params(...)` | Plots inverse-parameter trajectories stored in `history["inv_param"]`. |
| `plot_everything(...)` | Generates all available diagnostic plots and saves them to the requested directory. |

### `pinnse/__init__.py`: public interface

The package-level imports expose the main user-facing classes and utilities:

```python
from pinnse import (
    ANN, SANN, BranchedANN, Fourier_ANN,
    DataModule, Training, Plotter,
    Normalization, Denormalization, Analyze, Save,
)
```

---

## Example case studies

Each example directory contains process-specific files. The reusable package remains unchanged.

### Nonideal flash separation

Two flash surrogates are provided.

| Directory | Inputs | Outputs | Physics residuals |
|---|---|---|---|
| `flash/case1` | `T_flash`, `P_flash`, feed composition `Z_i` | vapor fraction `VF`, vapor composition `Y_i`, liquid composition `X_i` | component material balance and phase-composition summation constraints |
| `flash/case2` | `T_flash`, `P_flash`, `T_feed`, `P_feed`, feed flow `F`, feed composition `Z_i` | `VF`, `Y_i`, `X_i`, heat duty `Q` | same equilibrium constraints, with energy-duty prediction included as a supervised output |

`data_gen.py` in these folders uses Aspen Plus through COM automation to generate simulator data. The supplied `I_S_data.xlsx` and `D_S_data.xlsx` files allow users to train the models without regenerating data.

### Isothermal plug-flow reactor

The same reactor is represented through three surrogate formulations.

| Directory | Formulation | Inputs | Outputs | Residual structure |
|---|---|---|---|---|
| `isopfr/efm` | Effluent-flow model | inlet flows `F_in_i`, pressure `P`, temperature `T`, volume `V` | outlet/local flows `F_ot_i` | species-balance ODE residuals and inlet-flow boundary residuals |
| `isopfr/erm` | Extent-of-reaction model | `F_in_i`, `P`, `T`, `V` | extents `E_R1`, `E_R2`, `E_R3` | extent-space ODE residuals and zero-extent inlet constraints |
| `isopfr/cm/s0` | Conversion model | `F_in_i`, `P`, `T`, `V` | conversions `X_R1`, `X_R2`, `X_R3` | conversion-space ODE residuals and zero-conversion inlet constraints |
| `isopfr/cm/s1` | Conversion model with scheduler | same as `s0` | same as `s0` | same residuals with a `StepLR` training schedule |

These cases illustrate formulation-aware normalization and autograd-based residual construction. They are the recommended starting point for new users.

### Inverse PINN for parameter estimation

`isopfr/efm.inverse` extends the EFM reactor example by treating a kinetic parameter vector `theta` as trainable. The optimizer updates both the neural-network weights and `theta`. The learned parameter trajectory is saved in the training history and plotted automatically.

### Nonisothermal plug-flow reactor

`nonisopfr/` demonstrates a coupled mass--energy PINN for a nonisothermal reactor with species flows, concentrations, reactor temperature and coolant temperature as outputs. The model uses `SANN` to enforce positive outputs and combines material-balance, concentration-consistency, reactor-energy and coolant-energy residuals.

---

## Repository structure

```text
pinnse/
├── pyproject.toml
├── LICENSE
├── README.md
├── docs/
│   └── framework_overview.png
│
├── pinnse/
│   ├── __init__.py
│   ├── PINNs.py
│   ├── data.py
│   ├── train.py
│   ├── utils.py
│   └── plots.py
│
├── flash/
│   ├── Aspen Simulations/
│   ├── case1/
│   │   ├── data_gen.py
│   │   ├── phys_res.py
│   │   ├── main.py
│   │   ├── check.py
│   │   ├── I_S_data.xlsx
│   │   └── D_S_data.xlsx
│   └── case2/
│       └── ...
│
├── isopfr/
│   ├── efm/
│   ├── efm.inverse/
│   ├── erm/
│   └── cm/
│       ├── s0/
│       └── s1/
│
└── nonisopfr/
    ├── data_gen.py
    ├── pfr_model.py
    ├── phys_res.py
    ├── main.py
    ├── check.py
    ├── I_S_data.xlsx
    └── D_S_data.xlsx
```

Typical case-study files have the following roles.

| File | Role |
|---|---|
| `data_gen.py` | Generates labeled datasets from Aspen Plus, SciPy ODE solvers or other process models. |
| `pfr_model.py` | Defines the first-principles PFR model used for data generation and checking. |
| `phys_res.py` | Defines `Physics` and, when needed, `Boundary` residual classes. This is the main process-specific file. |
| `main.py` | Runs the end-to-end training workflow. |
| `check.py` | Reloads the trained model, compares against first-principles or simulator truth and generates evaluation figures. |
| `run.sh` | Optional HPC submission script. |
| `I_S_data.xlsx` | Labeled surrogate inputs. |
| `D_S_data.xlsx` | Labeled surrogate outputs. |

---

## Extending pinnse to a new process

A new process model can be added without changing the core package.

1. **Define the surrogate representation.** Decide the independent input set `I_S` and dependent output set `D_S`. Include all variables needed to evaluate the governing residuals.
2. **Generate labeled data.** Write a `data_gen.py` script that samples the admissible input domain and evaluates a simulator, ODE/PDE/DAE solver or experimental-data source. Save `I_S_data.xlsx` and `D_S_data.xlsx`.
3. **Choose normalization.** Use `Normalization.min_max` for standard `[0, 1]` scaling or `Normalization.min_max_pfr` for formulation-aware centered scaling in PFR-like systems.
4. **Write residual classes.** In `phys_res.py`, define a callable `Physics` class with signature `physics(x, y) -> residual_tensor`. If needed, define a callable `Boundary` class with the same signature.
5. **Construct loaders.** Use `DataModule` to create labeled, physics-collocation and boundary-collocation loaders. Use `Z_`, `X_` or `Y_` prefixes for composition columns that should be sampled on a simplex.
6. **Train.** Instantiate an architecture from `pinnse/PINNs.py`, construct a `Training` object and call `adam_step`. Add `lbfgs_step` if a full-batch refinement stage is desired.
7. **Evaluate.** Use `Analyze` and `check.py`-style scripts to compare the trained surrogate against held-out solver or simulator results.

A minimal residual class has the form:

```python
class Physics:
    def __init__(self, I_S_metrics, D_S_metrics, ...):
        self.I_S_metrics = I_S_metrics
        self.D_S_metrics = D_S_metrics

    def __call__(self, x, y):
        # 1. denormalize x and y if needed
        # 2. compute dimensional governing-equation residuals
        # 3. return shape (batch_size, n_residuals)
        return residuals
```

For derivative-based residuals, use PyTorch autograd and ensure that collocation inputs are created with `requires_grad_(True)`. The `Training` class handles this automatically for physics and boundary collocation batches.

---

## Outputs

Training and analysis scripts typically create:

| Output | Description |
|---|---|
| `logs/best_model.pth` | Model checkpoint with the lowest validation total loss. |
| `logs/training_history.xlsx` | Multi-sheet workbook containing all available loss, gradient, weight and inverse-parameter histories. |
| `logs/csv_files/` | CSV files for each history entry. |
| `figures/` | Training, validation, gradient, adaptive-weight and inverse-parameter plots. |
| case-specific comparison figures | Generated by `check.py`; may include parity plots, axial profiles, phase-composition plots and grouped error summaries. |

---

## Notes on platform support

The package, PFR examples and training from existing `.xlsx` datasets are cross-platform. Flash data generation and Aspen-based validation require Windows and a licensed Aspen Plus installation with COM access.

GPU acceleration is supported through PyTorch. Each example selects CUDA automatically when available:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

---

## Citation

If you use `pinnse` in academic work, please cite the associated manuscript once available. Until then, cite the repository in the software or methods section and include the version or commit hash used in your work.

---

## License

This project is distributed under the MIT license. See [`LICENSE`](LICENSE) for details.

---

## Contact

**Harshit Verma**  
Princeton University, Chemical and Biological Engineering  
`harshit.verma.che@gmail.com`
