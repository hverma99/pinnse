<p align="center">
  <img src="docs/logo.png" alt="pinnse logo" width="1000">
</p>


![python](https://img.shields.io/badge/python-%E2%89%A53.10-blue)
![pytorch](https://img.shields.io/badge/PyTorch-enabled-ee4c2c)
![license](https://img.shields.io/badge/license-MIT-green)

`pinnse` is a modular PyTorch framework for the development of physics-informed neural-network (PINN) surrogate models for chemical process modelling, simulation, and process systems engineering. The package separates process-specific ingredients—such as operating bounds, governing equations, input-output formulations, and residual definitions—from reusable learning infrastructure for data handling, normalization, neural-network construction, training, validation, testing, and visualization.

The repository currently includes representative examples for nonideal flash separation, isothermal plug-flow reactors under multiple surrogate formulations, inverse PINNs for parameter estimation, and a nonisothermal plug-flow reactor with coupled mass and energy balances.

<p align="center">
  <img src="docs/framework_overview.png" alt="pinnse framework overview" width="950">
</p>

At its core, `pinnse` follows a simple idea: define the process physics once, expose it through residual functions, and train a differentiable surrogate against both labelled data and physical constraints. The resulting models remain data-efficient, physically informed, and suitable for downstream use in simulation, design analysis, and optimization-oriented workflows.

---

## Contents

- [Why pinnse?](#why-pinnse)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Framework workflow](#framework-workflow)
- [Core package](#core-package)
- [Example case studies](#example-case-studies)
- [Repository structure](#repository-structure)
- [Extending pinnse](#extending-pinnse)
- [Outputs](#outputs)
- [License](#license)

---

## Why pinnse?

First-principles models in chemical engineering are often nonlinear, tightly coupled, and costly to solve repeatedly. Conventional surrogate models can reduce this cost, but purely data-driven surrogates may violate conservation laws, equilibrium constraints, or boundary conditions. `pinnse` addresses this limitation by combining labelled simulator or solver data with residual losses derived from the governing equations.

The framework is designed for process systems engineering settings in which a single unit operation may admit multiple surrogate formulations. For example, a plug-flow reactor may be represented in terms of effluent flowrates, reaction extents, or conversions. In `pinnse`, such formulation-specific choices are localized within each example, while the common infrastructure for data preparation, normalization, model training, and analysis is reused across cases.

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

The core package depends on:

- `numpy`
- `scipy`
- `torch`
- `pandas`
- `tqdm`

The example workflows and plotting utilities also use:

```bash
python -m pip install matplotlib scikit-learn openpyxl
```

For flash-data generation through Aspen Plus, the following are additionally required:

```bash
python -m pip install pywin32
```

> **Note**
> Training from the supplied `.xlsx` datasets does **not** require Aspen Plus. Aspen is only needed when regenerating flash datasets or running Aspen-based workflows.

---

## Quick start

The isothermal PFR examples are the most direct entry point because they do not depend on Aspen Plus.

```bash
cd examples/isopfr/efm
python main.py
python check.py
```

A standard run performs the following steps:

1. loads `I_S_data.xlsx` and `D_S_data.xlsx`,
2. normalizes the input and output spaces,
3. constructs supervised, physics-collocation, and boundary-collocation loaders,
4. builds a PINN surrogate model,
5. trains against data and residual losses,
6. stores the best checkpoint in `logs/`, and
7. evaluates the trained model through `check.py`.

The nonisothermal PFR example follows the same pattern:

```bash
cd examples/nonisopfr
python main.py
python check.py
```

Flash examples can be trained from the supplied datasets:

```bash
cd examples/flash/case1
python main.py
```

---

## Framework workflow

`pinnse` implements a composite PINN objective of the form

```math
\mathcal{L}_{\mathrm{total}}
= \mathcal{L}_{\mathrm{data}}
+ \lambda_P\,\mathcal{L}_{\mathrm{physics}}
+ \lambda_B\,\mathcal{L}_{\mathrm{boundary}},
```

where the data loss fits labelled samples, the physics loss penalizes residual violations at collocation points, and the boundary loss enforces boundary or initial conditions.

The figure above summarizes the workflow implemented in the repository:

| Stage | Purpose | Typical file |
|---|---|---|
| **Framework initialization** | Defines the process variables, operating bounds, model formulation, network architecture, dataset sizes, optimizer, scheduler, and loss weights. | `main.py` |
| **Data generation and sampling** | Generates labelled data from Aspen Plus, SciPy-based solvers, or process models, and prepares the sampled input-output space. | `data_gen.py` |
| **Data loading** | Splits the labelled data and constructs supervised, physics-collocation, and boundary-collocation data loaders. | `pinnse/data.py` |
| **PINN architecture** | Defines the differentiable neural-network surrogate. | `pinnse/PINNs.py` |
| **Physics residual formulation** | Encodes the governing equations in residual form and evaluates them on collocation batches. | `phys_res.py` |
| **Training, validation, and testing** | Optimizes the surrogate, monitors validation performance, checkpoints the best model, and records loss histories. | `pinnse/train.py` |
| **Post-processing and analysis** | Evaluates trained models, denormalizes outputs, computes errors, and generates figures. | `check.py`, `pinnse/utils.py`, `pinnse/plots.py` |

---

## Core package

The reusable framework is implemented in the `pinnse/` package.

### `pinnse/PINNs.py`

This module defines the neural-network architectures used as PINN surrogates.

| Class | Role |
|---|---|
| `ANN` | Standard fully connected feedforward network with Xavier-uniform weight initialization and zero biases. |
| `SANN` | Feedforward network with a `Softplus` output layer, useful when outputs must remain non-negative. |
| `BranchedANN` | Shared-trunk architecture with multiple output heads for structured multi-output regression. |
| `Fourier_ANN` | Feedforward architecture with Fourier feature embedding, useful for strongly varying or oscillatory mappings. |

Minimal example:

```python
import torch.nn as nn
from pinnse import ANN

layer_size = [dim_in] + [64] * 6 + [dim_out]
model = ANN(layer_size=layer_size, activation=nn.Tanh)
```

### `pinnse/data.py`

This module provides `DataModule`, which converts normalized `pandas.DataFrame` objects into PyTorch data loaders.

Key capabilities include:

- supervised train/validation/test splits through `labeled_data_loader()`,
- physics-collocation sampling through `phys_colloc_loader()`,
- boundary-collocation sampling through `bnd_colloc_loader()`,
- data-loader inspection and export utilities.

Notably, `phys_colloc_loader()` samples the interior input space using Latin hypercube sampling, while composition variables with prefixes such as `Z_`, `X_`, or `Y_` are sampled with Dirichlet distributions so that the resulting composition groups remain physically meaningful.

### `pinnse/train.py`

This module implements the `Training` class, which couples supervised data loss with optional physics and boundary residual losses.

Its functionality includes:

- epoch-wise training with `adam_step()`,
- optional second-stage optimization with `lbfgs_step()`,
- validation and testing through `validate_test()`,
- checkpointing of the best model,
- storage of training and validation loss histories,
- gradient-norm tracking for data, physics, and boundary losses,
- optional adaptive updating of loss weights.

### `pinnse/utils.py`

This module provides common utilities for normalization, denormalization, result export, and post-training analysis.

| Class | Role |
|---|---|
| `Normalization` | Applies min-max, centred scaling, max-absolute, mean normalization, and z-score transformations. Includes routines specialized for PFR formulations. |
| `Denormalization` | Maps normalized model predictions back to dimensional variables. |
| `Analyze` | Loads trained models, evaluates predictions, and computes error metrics. |
| `Save` | Writes histories or results to Excel or CSV. |

### `pinnse/plots.py`

This module defines the `Plotter` class for visualizing model-development history. It includes methods to plot:

- individual loss histories,
- full training and validation loss traces,
- physics and boundary weight evolution,
- gradient histories,
- inverse-parameter trajectories, and
- a combined summary view through `plot_everything()`.

### `pinnse/__init__.py`

The package exports the main public API:

```python
from pinnse import (
    ANN, SANN, BranchedANN, Fourier_ANN,
    DataModule, Training, Plotter,
    Normalization, Denormalization, Save, Analyze,
)
```

---

## Example case studies

The repository includes example workflows in `examples/`.

| Example | Description |
|---|---|
| `examples/flash/case1` | PINN workflow for a flash-separation formulation using supplied labelled data. |
| `examples/flash/case2` | Alternative flash formulation with its own residual and training script. |
| `examples/isopfr/efm` | Isothermal PFR using an effluent-flow formulation. |
| `examples/isopfr/erm` | Isothermal PFR using an extent-of-reaction formulation. |
| `examples/isopfr/efm.inverse` | Inverse PINN workflow for parameter estimation in the isothermal PFR setting. |
| `examples/nonisopfr` | Nonisothermal PFR with coupled mass and energy balances. |

A typical example directory contains:

- `main.py`: model setup and training script,
- `phys_res.py`: physics and boundary residual definitions,
- `data_gen.py`: dataset generation script,
- `check.py`: post-training evaluation and comparison,
- `I_S_data.xlsx`, `D_S_data.xlsx`: labelled input and output datasets,
- `pfr_model.py`: process model helper functions where applicable,
- `run.sh`: convenience shell script.

---

## Repository structure

```text
pinnse/
├── docs/
│   ├── framework_overview.png
│   └── pinnse_logo.png
├── examples/
│   ├── flash/
│   │   ├── Aspen Simulations/
│   │   ├── case1/
│   │   └── case2/
│   ├── isopfr/
│   │   ├── efm/
│   │   ├── erm/
│   │   ├── efm.inverse/
│   │   └── cm/
│   └── nonisopfr/
├── pinnse/
│   ├── __init__.py
│   ├── PINNs.py
│   ├── data.py
│   ├── train.py
│   ├── utils.py
│   └── plots.py
├── pyproject.toml
├── README.md
└── LICENSE
```

---

## Extending pinnse

To apply `pinnse` to a new process system, the typical workflow is:

1. define the process inputs (`I_S`) and outputs (`D_S`),
2. generate or collect labelled datasets,
3. write a `phys_res.py` file that evaluates the governing-equation residuals,
4. define any required boundary residuals,
5. construct a `main.py` script that configures the architecture, data loaders, optimizer, scheduler, and loss weights,
6. train the model using `Training`, and
7. evaluate the resulting surrogate through a `check.py` script.

This design allows the process-specific physics to remain local to the new case while the rest of the pipeline is inherited directly from the package.

---

## Outputs

A typical training run generates:

- a best-model checkpoint,
- training and validation loss histories,
- optional weight and gradient histories,
- prediction-versus-reference comparison results,
- figures summarizing the model performance.

The exact output files are controlled by the example scripts and their logging conventions.

---

## License

This project is released under the **MIT License**. See [LICENSE](LICENSE) for details.
