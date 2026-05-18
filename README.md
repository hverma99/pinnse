# pinnse: physics-informed neural networks for process systems engineering

![python](https://img.shields.io/badge/python-%E2%89%A53.10-blue)
![pytorch](https://img.shields.io/badge/PyTorch-enabled-ee4c2c)
![license](https://img.shields.io/badge/license-MIT-green)

`pinnse` is a modular, PyTorch-based framework for building physics-informed neural-network (PINN) surrogate models for chemical process modelling, simulation and process systems engineering. The package separates process-specific information—operating bounds, governing equations, input-output representations and residual definitions—from reusable infrastructure for data handling, normalization, neural-network construction, training, validation, testing and visualization.

The repository includes representative case studies for nonideal flash separation, isothermal plug-flow reactors under multiple surrogate formulations, inverse PINNs for parameter estimation and a nonisothermal plug-flow reactor with coupled mass and energy balances.

<p align="center">
  <img src="docs/framework_overview.png" alt="pinnse framework overview" width="900">
</p>

---

## Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Framework design](#framework-design)
- [Package anatomy](#package-anatomy)
- [Example case studies](#example-case-studies)
- [Building a new process model](#building-a-new-process-model)
- [Outputs](#outputs)
- [Repository structure](#repository-structure)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

---

## Overview

First-principles models in chemical engineering are often nonlinear, coupled and expensive to solve repeatedly. Data-driven surrogates can accelerate simulation and design, but they may violate conservation laws, equilibrium relationships or boundary conditions. `pinnse` addresses this gap by training differentiable neural surrogates against both labelled data and residuals derived from governing equations.

The framework is intended for process systems engineering applications in which the same physical system may admit multiple useful surrogate formulations. For example, a plug-flow reactor can be represented through effluent flows, reaction extents or conversions. `pinnse` keeps such formulation-specific choices local to each example directory, while reusing the same package-level machinery for data loading, collocation sampling, model construction, optimization and post-processing.

`pinnse` provides:

- supervised and physics-informed training workflows;
- interior and boundary collocation sampling;
- formulation-aware normalization utilities;
- fully connected, softplus-output, branched and Fourier-feature neural architectures;
- Adam and optional LBFGS optimization;
- adaptive physics and boundary loss weighting;
- support for inverse PINNs through trainable physical parameters;
- training diagnostics, checkpointing, history export and plotting utilities.

---

## Installation

Clone the repository and install the package in editable mode:

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

The package uses `numpy`, `scipy`, `torch`, `pandas` and `tqdm`. The data loader and example workflows also require:

```bash
python -m pip install scikit-learn matplotlib openpyxl
```

Flash data generation through Aspen Plus requires Windows, Aspen Plus and COM automation:

```bash
python -m pip install pywin32
```

Training the supplied examples from the included `.xlsx` datasets does not require Aspen Plus.

---

## Quick start

The isothermal PFR examples are the recommended entry point because they use a SciPy-based process model and do not require Aspen Plus. From the repository root:

```bash
cd examples/isopfr/efm
python main.py
python check.py
```

A standard run:

1. reads `I_S_data.xlsx` and `D_S_data.xlsx`;
2. normalizes the input and output spaces;
3. builds labelled, physics-collocation and boundary-collocation data loaders;
4. constructs a neural surrogate;
5. trains the model using data, physics and boundary losses;
6. saves the best checkpoint to `logs/best_model.pth`;
7. exports training histories to `logs/`; and
8. generates diagnostic and comparison figures.

For a short smoke test, reduce the number of epochs in `main.py` before running the script.

---

## Framework design

`pinnse` trains a neural surrogate by minimizing the composite objective

```math
\mathcal{L}_{\mathrm{total}}
= \mathcal{L}_{\mathrm{data}}
+ \lambda_P \mathcal{L}_{\mathrm{physics}}
+ \lambda_B \mathcal{L}_{\mathrm{boundary}},
```

where the data loss fits labelled input-output pairs, the physics loss penalizes governing-equation residuals at interior collocation points and the boundary loss enforces initial or boundary constraints.

The workflow shown in the figure is implemented through a small number of reusable modules:

| Stage | Purpose | Typical file |
|---|---|---|
| Framework initialization | Defines process specifications, physical parameters, input-output variables, training settings and model hyperparameters. | `main.py` |
| Data generation and sampling | Generates labelled data from Aspen Plus, SciPy solvers or other process models. | `data_gen.py` |
| Data loading | Splits labelled data and constructs supervised, physics-collocation and boundary-collocation loaders. | `pinnse/data.py` |
| Neural architecture | Defines the differentiable surrogate model. | `pinnse/PINNs.py` |
| Physics residual formulation | Converts normalized variables to physical variables and evaluates residual tensors. | `phys_res.py` |
| Training, validation and testing | Optimizes the network, evaluates validation and test losses, and stores the best checkpoint. | `pinnse/train.py` |
| Post-processing | Reloads trained models, compares predictions against process-model results and generates figures. | `check.py`, `pinnse/utils.py`, `pinnse/plots.py` |

---

## Package anatomy

The reusable source code is contained in `pinnse/`. These modules are independent of any specific unit operation.

### `pinnse/PINNs.py`

Defines neural-network architectures with a common PyTorch interface.

| Object | Role |
|---|---|
| `ANN(layer_size, activation)` | Fully connected feedforward network with Xavier-uniform weights and zero biases. This is the default surrogate architecture in most examples. |
| `SANN(layer_size, activation)` | Fully connected network with a `Softplus` output activation. This is useful when dependent variables must remain non-negative. |
| `BranchedANN(in_dim, trunk_layers, head_dims, activation)` | Shared-trunk network with multiple output heads, useful when related output groups benefit from separate prediction heads. |
| `Fourier_ANN(layer_size, activation, fourier_levels, positive_output)` | Feedforward network with Fourier features applied to the last input coordinate, useful for strongly varying or spatially structured mappings. |

Example:

```python
import torch.nn as nn
from pinnse import ANN

layer_size = [dim_in] + [64] * 6 + [dim_out]
model = ANN(layer_size=layer_size, activation=nn.Tanh)
```

### `pinnse/data.py`

Provides `DataModule`, which converts normalized `pandas.DataFrame` objects into PyTorch data loaders.

| Method | Role |
|---|---|
| `labeled_data_loader()` | Splits labelled input-output data into training, validation and test loaders. It also stores input bounds used for collocation sampling. |
| `phys_colloc_loader()` | Samples interior collocation points over the input domain. Ordinary bounded variables are sampled by Latin hypercube sampling; composition groups beginning with `Z_`, `X_` or `Y_` are sampled from Dirichlet distributions. |
| `bnd_colloc_loader()` | Samples boundary points by fixing the last input column to a prescribed normalized boundary value and sampling the remaining variables over their bounds. |
| `inspect_loader()` | Prints tensor shapes and batch counts for debugging. |
| `save_loaders()` | Exports loader contents to Excel for inspection. |

Typical use:

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

train_loader, val_loader, test_loader = data.labeled_data_loader()
phys_coll_loader = data.phys_colloc_loader()
bnd_coll_loader = data.bnd_colloc_loader()
```

### `pinnse/train.py`

Provides `Training`, the main optimization class. It combines data loss, optional physics-residual loss and optional boundary-residual loss.

| Method or feature | Role |
|---|---|
| `train_epoch()` | Performs one training epoch over labelled batches while cycling through physics and boundary collocation batches. |
| `validate_test()` | Evaluates data, physics and boundary losses on validation or test loaders. |
| `adam_step(epochs, val_every)` | Runs Adam-based training, logs histories and checkpoints the model with the lowest validation total loss. |
| `lbfgs_step(mode, N_LBFGS)` | Performs optional full-batch LBFGS refinement using data, physics, boundary, combined or overall-physics losses. |
| `adapt_wts=True` | Updates physics and boundary weights by gradient-norm matching. |
| `theta` | Allows trainable physical parameters to be optimized with the neural-network weights for inverse PINN problems. |

Typical use:

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

### `pinnse/utils.py`

Collects utilities for normalization, denormalization, evaluation and saving.

| Object | Role |
|---|---|
| `Normalization` | Provides min-max scaling, max-absolute scaling, mean normalization, z-score normalization, centered `[-1, 1]` scaling and formulation-aware PFR normalization through `min_max_pfr`. |
| `Denormalization` | Inverts the corresponding normalization operations and supports column-wise denormalization inside residual functions. |
| `Analyze` | Loads trained checkpoints, evaluates PINN predictions and computes grouped error metrics such as MAE, RMSE and R². |
| `Save` | Exports training histories to Excel workbooks or CSV files. |

The PFR helper `Normalization.min_max_pfr(...)` implements formulation-aware scaling for effluent-flow, extent-of-reaction and conversion-based reactor models.

### `pinnse/plots.py`

Provides `Plotter`, which converts training histories into diagnostic figures.

| Method | Role |
|---|---|
| `plot_individual_loss()` | Plots one training or validation loss history. |
| `plot_all_train_losses()` | Plots total, data, physics and boundary training losses. |
| `plot_all_val_losses()` | Plots total, data, physics and boundary validation losses. |
| `plot_weights()` | Plots adaptive physics and boundary weights. |
| `plot_gradient_history()` | Plots data, physics or boundary gradient norms. |
| `plot_inverse_params()` | Plots learned inverse-parameter trajectories. |
| `plot_everything()` | Saves all available diagnostic plots. |

### `pinnse/__init__.py`

Exposes the public API:

```python
from pinnse import (
    ANN, SANN, BranchedANN, Fourier_ANN,
    DataModule, Training, Plotter,
    Normalization, Denormalization, Save, Analyze,
)
```

---

## Example case studies

Each example directory contains process-specific files. The core package remains unchanged.

| Directory | System and formulation | Main features |
|---|---|---|
| `examples/flash/case1` | Nonideal flash separation with flash temperature, flash pressure and feed composition as inputs. | Predicts vapor fraction and phase compositions; enforces component balance and phase-fraction constraints. |
| `examples/flash/case2` | Extended nonideal flash separation with feed conditions and feed flow included. | Predicts vapor fraction, phase compositions and heat duty; uses Aspen-generated data. |
| `examples/isopfr/efm` | Isothermal PFR, effluent-flow model. | Predicts species flows and enforces species-balance ODE residuals. |
| `examples/isopfr/erm` | Isothermal PFR, extent-of-reaction model. | Predicts reaction extents and enforces extent-space residuals. |
| `examples/isopfr/cm/s0` | Isothermal PFR, conversion model. | Predicts reaction conversions and enforces conversion-space residuals. |
| `examples/isopfr/cm/s1` | Conversion model with a modified training schedule. | Demonstrates scheduler-based training for the same formulation. |
| `examples/isopfr/efm.inverse` | Inverse PINN for the isothermal PFR. | Learns kinetic parameters jointly with the neural surrogate. |
| `examples/nonisopfr` | Nonisothermal PFR with coupled mass and energy balances. | Uses `SANN`, physics and boundary losses, and optional LBFGS refinement. |

Typical case-study files are:

| File | Role |
|---|---|
| `data_gen.py` | Generates labelled datasets from Aspen Plus, SciPy ODE solvers or other process models. |
| `pfr_model.py` | Defines the first-principles PFR model used for data generation and validation. |
| `phys_res.py` | Defines process-specific `Physics` and, when required, `Boundary` residual classes. |
| `main.py` | Runs the complete training workflow. |
| `check.py` | Reloads the trained model and compares predictions against first-principles or simulator results. |
| `run.sh` | Optional shell script for running jobs on external compute resources. |
| `I_S_data.xlsx` | Labelled independent-variable dataset. |
| `D_S_data.xlsx` | Labelled dependent-variable dataset. |

---

## Building a new process model

A new chemical process can be added without modifying the source files in `pinnse/`.

1. **Define the surrogate representation.** Select the independent variables `I_S` and dependent variables `D_S`. Include all variables required to evaluate the governing residuals.
2. **Generate labelled data.** Write `data_gen.py` to sample the admissible input domain and evaluate a simulator, first-principles solver or experimental data source. Save `I_S_data.xlsx` and `D_S_data.xlsx`.
3. **Normalize the variables.** Use `Normalization.min_max` for standard scaling or `Normalization.min_max_pfr` for formulation-aware PFR scaling.
4. **Write residual classes.** In `phys_res.py`, define a callable `Physics` class with signature `physics(x, y) -> residuals`. Add a `Boundary` class when inlet, initial or boundary constraints are needed.
5. **Construct loaders.** Use `DataModule` to create labelled, interior-collocation and boundary-collocation loaders. Prefix composition variables with `Z_`, `X_` or `Y_` when they should be sampled on a simplex.
6. **Train the PINN.** Instantiate a model from `pinnse/PINNs.py`, construct a `Training` object and call `adam_step`. Use `lbfgs_step` for optional full-batch refinement.
7. **Evaluate and report.** Use `Analyze`, `Save`, `Plotter` and a case-specific `check.py` script to compute errors and generate figures.

A minimal residual class follows the pattern below:

```python
class Physics:
    def __init__(self, I_S_metrics, D_S_metrics, *params):
        self.I_S_metrics = I_S_metrics
        self.D_S_metrics = D_S_metrics
        self.params = params

    def __call__(self, x, y):
        # 1. Denormalize x and y if required.
        # 2. Compute dimensional conservation, equilibrium or rate residuals.
        # 3. Return a tensor of shape (batch_size, n_residuals).
        return residuals
```

For derivative-based residuals, compute derivatives with PyTorch autograd. The `Training` class automatically sets `requires_grad_(True)` for physics and boundary collocation inputs.

---

## Outputs

Training and analysis scripts typically create:

| Output | Description |
|---|---|
| `logs/best_model.pth` | Checkpoint with the lowest validation total loss. |
| `logs/training_history.xlsx` | Multi-sheet workbook containing loss, gradient, weight and inverse-parameter histories. |
| `logs/csv_files/` | CSV exports of the same histories. |
| `figures/` | Training, validation, gradient, adaptive-weight and inverse-parameter plots. |
| case-specific comparison figures | Parity plots, axial profiles, phase-composition plots or grouped error summaries generated by `check.py`. |

---

## Repository structure

```text
pinnse/
├── README.md
├── LICENSE
├── pyproject.toml
├── docs/
│   └── framework_overview.png
├── pinnse/
│   ├── __init__.py
│   ├── PINNs.py
│   ├── data.py
│   ├── train.py
│   ├── utils.py
│   └── plots.py
└── examples/
    ├── flash/
    │   ├── Aspen Simulations/
    │   ├── case1/
    │   └── case2/
    ├── isopfr/
    │   ├── efm/
    │   ├── efm.inverse/
    │   ├── erm/
    │   └── cm/
    │       ├── s0/
    │       └── s1/
    └── nonisopfr/
```

---

## Platform notes

The package, PFR examples and training from existing `.xlsx` datasets are cross-platform. Flash data generation and Aspen-based validation require Windows and a licensed Aspen Plus installation with COM access.

GPU acceleration is supported through PyTorch. The examples select CUDA automatically when available:

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
Department of Chemical and Biological Engineering  
Princeton University  
`harshit.verma.che@gmail.com`
