# pinnse: Physics-Informed Neural Networks for Process Systems Engineering

`pinnse` is a modular Python framework for building **physics-informed neural networks (PINNs)** for process systems engineering (PSE). The repository combines a reusable core package with worked examples for:

- **nonideal flash separation**,
- **isothermal plug-flow reactors (PFRs)** under multiple surrogate formulations, and
- **inverse PINNs** for joint state/parameter learning.

The design philosophy is to make PINN development **systematic, extensible, and formulation-aware**. Instead of hard-coding a single problem, the framework separates:

1. **system-specific pieces**: process model, governing equations, residual construction, and dataset generation, from
2. **reusable learning infrastructure**: normalization, data handling, neural-network architectures, training, validation, evaluation, and plotting.

That separation is what makes this repository useful both as a research codebase and as a starting point for new PINN applications in PSE.

---

## What this repository contains

At a high level, the repository has two layers:

- the reusable package **`pinnse/`**, which provides the generic PINN workflow, and
- the **example case directories** (`flash/`, `isopfr/`) that define specific processes, physics, and training scripts.

### Repository layout

```text
pinnse/
├── pyproject.toml                # package metadata
├── README.md                     # repository documentation
├── LICENSE
│
├── pinnse/                       # reusable framework package
│   ├── __init__.py
│   ├── data.py                   # data splitting + collocation loaders
│   ├── PINNs.py                  # ANN, BranchedANN, Fourier_ANN
│   ├── train.py                  # training, validation, checkpointing, LBFGS
│   ├── plots.py                  # standardized history plotting
│   └── utils.py                  # normalization, denormalization, evaluation, saving
│
├── flash/
│   ├── Aspen Simulations/
│   │   ├── Flash.bkp             # Aspen template / backup file
│   │   └── Process.apwz
│   ├── case1/
│   │   ├── data_gen.py           # flash data generation (Aspen + COM)
│   │   ├── phys_res.py           # flash equilibrium residuals
│   │   ├── main.py               # training script
│   │   ├── check.py              # Aspen vs PINN comparison plots
│   │   ├── I_S_data.xlsx         # surrogate input dataset
│   │   └── D_S_data.xlsx         # surrogate output dataset
│   └── case2/
│       ├── data_gen.py
│       ├── phys_res.py
│       ├── main.py
│       ├── check.py
│       ├── I_S_data.xlsx
│       └── D_S_data.xlsx
│
├── isopfr/
│   ├── efm/                      # effluent-flow formulation
│   ├── efm.inverse/              # inverse PINN variant
│   ├── erm/                      # extent-based formulation
│   └── cm/
│       ├── s0/                   # conversion model, base training setup
│       └── s1/                   # conversion model with scheduler variant
│
├── nonisopfr/                    # placeholder for non-isothermal extensions
└── logs/                         # output folder used by scripts
```

---

## Framework concept

The framework is organized around the standard PINN objective

```math
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}} + \lambda_P\,\mathcal{L}_{\text{physics}} + \lambda_B\,\mathcal{L}_{\text{boundary}}
```

where:

- **data loss** matches labeled simulator/first-principles outputs,
- **physics loss** penalizes residual violations at collocation points, and
- **boundary loss** enforces boundary conditions where relevant.

The reusable workflow is:

1. generate or load labeled datasets `I_S` and `D_S`,
2. normalize the data,
3. build labeled and collocation data loaders,
4. define a neural network architecture,
5. define physics and optional boundary residuals,
6. train with Adam (and optionally LBFGS),
7. save histories/checkpoints, and
8. compare against simulator or first-principles truth.

The framework is intentionally agnostic to the exact process model. The example directories show how to plug in process-specific residuals without rewriting the training infrastructure.

---

## Supported example problems

### 1. Flash separation

The repository includes two flash examples.

#### `flash/case1`
A reduced flash surrogate using:

- **inputs**: `T_flash`, `P_flash`, and feed composition `Z_i`
- **outputs**: `VF`, vapor compositions `Y_i`, liquid compositions `X_i`
- **physics residual**:
  - component material balance,
  - liquid mole-fraction summation, and
  - vapor mole-fraction summation.

This formulation is useful when the surrogate only needs equilibrium-state outputs for a given flash operating point and feed composition.

#### `flash/case2`
A richer flash surrogate using:

- **inputs**: `T_flash`, `P_flash`, `T_feed`, `P_feed`, total feed flow `F`, and `Z_i`
- **outputs**: `VF`, `Y_i`, `X_i`, and heat duty `Q`
- **physics residual**: same equilibrium constraints as case 1.

This version is closer to a process-facing surrogate because it includes feed state and flowrate information and predicts energy duty.

### 2. Isothermal PFR

The repository demonstrates that the same framework can support **multiple surrogate formulations for the same underlying reactor**.

#### `isopfr/efm`
Effluent-flow formulation:

- **inputs**: `F_in_i`, `P`, `T`, `V`
- **outputs**: `F_ot_i`
- **physics residual**: reactor species balances enforced through autograd derivatives
- **boundary residual**: inlet consistency at the reactor entrance.

#### `isopfr/erm`
Extent-based formulation:

- **inputs**: `F_in_i`, `P`, `T`, `V`
- **outputs**: `E_R1`, `E_R2`, `E_R3`
- **physics residual**: ODE residuals in extent space
- **boundary residual**: zero/consistent reaction extent at the inlet.

#### `isopfr/cm/s0` and `isopfr/cm/s1`
Conversion-based formulation:

- **inputs**: `F_in_i`, `P`, `T`, `V`
- **outputs**: `X_R1`, `X_R2`, `X_R3`
- **physics residual**: ODE residuals in conversion space
- **boundary residual**: inlet conversion condition.

`s0` and `s1` use the same formulation but differ in training setup. In the current snapshot, `s1` adds a `StepLR` scheduler, making the two folders useful for controlled training comparisons.

### 3. Inverse PINN

#### `isopfr/efm.inverse`
This example performs **joint neural-network training and parameter estimation**.

- State surrogate: same as EFM (`F_ot_i` prediction)
- Trainable inverse parameters: activation-energy-like parameter vector `theta`
- The parameter history is stored in `history['inv_param']` and plotted automatically by `Plotter`.

This folder is the clearest starting point if you want to extend `pinnse` toward inverse problems or hybrid parameter/state learning.

---

## Core package modules

## `pinnse/PINNs.py`

Defines the neural-network architectures used by the framework.

### `ANN`
A standard fully connected feedforward neural network.

Use this when you want a conventional multilayer perceptron for scalar/vector regression. All current example `main.py` scripts use this class.

### `BranchedANN`
A shared-trunk architecture with multiple output heads.

Useful when outputs share common latent structure but may benefit from partially separate final mappings.

### `Fourier_ANN`
A feedforward architecture with Fourier-feature augmentation.

Useful when learning oscillatory or high-frequency mappings. It is included in the framework even though the bundled examples use `ANN`.

---

## `pinnse/data.py`

Provides the `DataModule` used to construct labeled and collocation data loaders.

### `DataModule.labeled_data_loader()`
Splits `I_S_data` and `D_S_data` into:

- training loader,
- validation loader, and
- test loader.

It also stores the lower and upper bounds of the input space, which are later reused when constructing collocation samples.

### `DataModule.phys_colloc_loader()`
Generates physics collocation points automatically from the input-space bounds.

Two sampling modes are handled internally:

- **Latin Hypercube Sampling (LHS)** for ordinary bounded variables, and
- **Dirichlet sampling** for columns whose names start with `Z_`, `X_`, or `Y_`.

That means composition-like variables automatically satisfy simplex structure during collocation generation.

### `DataModule.bnd_colloc_loader()`
Builds boundary collocation points by:

- sampling all input columns except the last one over their admissible bounds, and
- fixing the **last input column** to a specified boundary value (default `-1.0`).

This is particularly convenient for the PFR examples, where the last input variable is the normalized reactor coordinate/volume `V` and the boundary is imposed at the reactor inlet.

### `DataModule.inspect_loader()`
Utility function to print batch shapes and number of batches.

### `DataModule.save_loaders()`
Writes loader contents to disk for inspection/debugging.

---

## `pinnse/utils.py`

This file contains most of the reusable preprocessing and evaluation helpers.

### `Normalization`
There are two main normalization pathways in the repository.

#### `Normalization.min_max(...)`
Applies standard min-max scaling to `[0, 1]`.

This is used in the flash examples. A useful feature is that you can choose **which columns to normalize**. Columns not listed in `normalize_cols` are left unchanged.

#### `Normalization.min_max_pfr(...)`
Applies **centered min-max scaling to `[-1, 1]`** for PFR examples.

This method is formulation-aware:

- **EFM**: uses one shared global min/max for inlet and outlet flow variables and local scaling for `P`, `T`, `V`
- **ERM**: uses global scaling for inlet flows and local scaling for `P`, `T`, `V`, and extents
- **CM**: uses global scaling for inlet flows and local scaling for `P`, `T`, `V`, and conversions.

This design keeps related flow variables on a consistent scale while allowing thermodynamic/state variables to retain formulation-specific ranges.

#### Other helpers
- `min_max_defined_metrics(...)`: normalize new data using saved metrics
- `scale_centered_defined_metrics(...)`: reapply `[-1, 1]` scaling with saved metrics.

### `Denormalization`
Companion functions for mapping predictions/residual inputs back to physical space.

Key helpers:

- `min_max(...)`
- `min_max_col(...)`
- `min_max_pfr(...)`

These are used heavily inside the `phys_res.py` files to formulate residuals in dimensional space.

### `Analyze`
Evaluation helpers for trained models.

Important functions:

- `load_ann_pinn(...)`: rebuild an `ANN` model and load weights from `logs/best_model.pth`
- `pinn_eval(...)`: evaluate a model for standard min-max scaled data
- `pinn_eval_pfr(...)`: evaluate a PFR model trained with centered scaling
- `error_metrics(...)`: compute grouped MAE, RMSE, and R².

### `Save`
Saves training history to:

- `logs/training_history.xlsx`
- `logs/csv_files/*.csv`

---

## `pinnse/train.py`

This module defines the `Training` class.

### Main capabilities

- supervised training with labeled data loss,
- optional physics loss and boundary loss,
- adaptive weight updates based on gradient norms,
- validation-driven checkpointing,
- optional learning-rate scheduler support,
- test evaluation using the best checkpoint, and
- optional second-stage **LBFGS** optimization.

### `Training.__init__(...)`
The most important arguments are:

- `model`
- `train_loader`, `val_loader`, `test_loader`
- `optimizer`
- `loss_fn`
- `device`
- `phys_coll_loader`
- `bnd_coll_loader`
- `phys_residual`
- `bnd_residual`
- `phys_weight`, `bnd_weight`
- `adapt_wts`
- `scheduler`
- `theta` for inverse problems.

### Adaptive weighting
If `adapt_wts=True`, the trainer updates `phys_weight` and `bnd_weight` every 10 epochs using gradient-norm matching with exponential moving averaging.

This is especially useful when the data, physics, and boundary losses operate on very different scales.

### Validation/test evaluation
During validation and test evaluation, the trainer does not necessarily use all collocation points. It samples a fraction of stored collocation batches for efficiency.

### `adam_step(...)`
This is the main optimization routine used in all bundled examples.

Returned history keys include:

- `loss_t`, `loss_d`, `loss_p`, `loss_b`
- `val_loss_t`, `val_loss_d`, `val_loss_p`, `val_loss_b`
- `wt_phys`, `wt_bnd`
- `grad_data_hist`, `grad_phys_hist`, `grad_bnd_hist`
- `inv_param`.

### `lbfgs_step(...)`
Implements a second-phase full-batch LBFGS pass for selected optimization modes:

- `data`
- `physics`
- `boundary`
- `overall physics`
- `combined`.

The current example scripts do not call this function, but it is already part of the framework.

---

## `pinnse/plots.py`

Provides the `Plotter` class for standardized training diagnostics.

### Generated plots
Depending on what is present in the history dictionary, `plot_everything(...)` can generate:

- individual training loss plots,
- individual validation loss plots,
- combined training loss plot,
- combined validation loss plot,
- adaptive weight history,
- gradient-norm histories, and
- inverse-parameter trajectories.

The plotting utilities save images automatically into the target directory you provide, typically `figures/`.

---

## Installation

## 1. Clone the repository

```bash
git clone <your-repo-url>
cd pinnse
```

## 2. Create a Python environment

A clean environment is strongly recommended.

### Conda example

```bash
conda create -n pinnse python=3.11 -y
conda activate pinnse
```

### venv example

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows PowerShell
```

## 3. Install the package

```bash
pip install -e .
```

## 4. Install example dependencies

The current `pyproject.toml` lists the minimal core dependencies, but the bundled examples also rely on additional packages. Install these explicitly:

```bash
pip install matplotlib scikit-learn openpyxl
```

### Additional dependency for flash data generation on Windows

If you want to regenerate flash datasets from Aspen Plus rather than using the provided `.xlsx` files:

```bash
pip install pywin32
```

You also need:

- a **Windows** environment,
- **Aspen Plus** installed, and
- a working COM automation setup.

---

## Dependency summary

### Core package dependencies

- `numpy`
- `scipy`
- `torch`
- `pandas`
- `tqdm`

### Additional dependencies used by examples/utilities

- `matplotlib`
- `scikit-learn`
- `openpyxl`
- `pywin32` (flash data generation only, Windows only)

---

## Quick start

If you want to get something running immediately, start with one of the PFR examples. Those do **not** depend on Aspen and are easier to reproduce on any platform.

### Option A: run the EFM PFR example

```bash
cd isopfr/efm
python main.py
python check.py
```

This will:

1. load `I_S_data.xlsx` and `D_S_data.xlsx`,
2. normalize inputs/outputs,
3. build data and collocation loaders,
4. train the PINN,
5. save the best checkpoint to `logs/best_model.pth`,
6. write training histories to `logs/`, and
7. create training and comparison figures.

### Option B: run the flash example using the provided datasets

```bash
cd flash/case2
python main.py
```

This will train the model using the supplied Excel datasets.

> `flash/check.py` re-evaluates truth using Aspen-based utilities, so it generally requires the Windows + Aspen setup. Training from the provided `.xlsx` files does not.

---

## Example-by-example usage

## Flash case 1

```bash
cd flash/case1
python main.py
```

### Data conventions

- **Input file**: `I_S_data.xlsx`
  - `T_flash`, `P_flash`, `Z_O2`, `Z_CO2`, `Z_H2O`, `Z_C6H6`, `Z_C4H2O3`
- **Output file**: `D_S_data.xlsx`
  - `VF`, `Y_O2`, `Y_CO2`, `Y_H2O`, `Y_C6H6`, `Y_C4H2O3`, `X_O2`, `X_CO2`, `X_H2O`, `X_C6H6`, `X_C4H2O3`

### Training behavior

- selective input normalization: only `T_flash` and `P_flash`
- no output normalization in the shipped script
- physics collocation only; no separate boundary collocation.

## Flash case 2

```bash
cd flash/case2
python main.py
```

### Data conventions

- **Input file**: `I_S_data.xlsx`
  - `T_flash`, `P_flash`, `T_feed`, `P_feed`, `F`, `Z_i`
- **Output file**: `D_S_data.xlsx`
  - `VF`, `Y_i`, `X_i`, `Q`

### Training behavior

- selective input normalization for `T_flash`, `P_flash`, `T_feed`, `P_feed`, `F`
- only `Q` is normalized in the shipped script
- equilibrium constraints are enforced through the physics residual.

## Isothermal PFR: EFM

```bash
cd isopfr/efm
python main.py
python check.py
```

### Data conventions

- **Input file**: `I_S_data.xlsx`
  - `F_in_O2`, `F_in_CO2`, `F_in_H2O`, `F_in_C6H6`, `F_in_C4H2O3`, `P`, `T`, `V`
- **Output file**: `D_S_data.xlsx`
  - `F_ot_O2`, `F_ot_CO2`, `F_ot_H2O`, `F_ot_C6H6`, `F_ot_C4H2O3`

### Training behavior

- centered min-max normalization to `[-1, 1]`
- physics collocation + boundary collocation
- fixed physics and boundary weights in the default script.

## Isothermal PFR: ERM

```bash
cd isopfr/erm
python main.py
python check.py
```

### Output space

`E_R1`, `E_R2`, `E_R3`

### Notable training choice

The shipped ERM script uses `adapt_wts=True`, so physics and boundary weights are updated automatically during training.

## Isothermal PFR: CM

```bash
cd isopfr/cm/s0
python main.py
python check.py
```

or

```bash
cd isopfr/cm/s1
python main.py
python check.py
```

### Output space

`X_R1`, `X_R2`, `X_R3`

### Difference between `s0` and `s1`

- `s0`: base Adam training setup
- `s1`: same formulation with a `StepLR` scheduler.

## Inverse PINN example

```bash
cd isopfr/efm.inverse
python main.py
python check.py
```

### What happens here

- the network learns the EFM state mapping,
- a trainable parameter vector `theta` is optimized together with network weights,
- validation checkpoints are still driven by the total loss,
- parameter trajectories are saved in `history['inv_param']`, and
- `Plotter` creates inverse-parameter plots automatically.

---

## Typical training script structure

All `main.py` files in the examples follow the same pattern.

```python
import torch
import torch.nn as nn
import pandas as pd

from pinnse import Normalization, DataModule, ANN, Training, Plotter, Save
from phys_res import Physics, Boundary
```

### Step 1: load the labeled datasets

```python
I_S_data = pd.read_excel("I_S_data.xlsx")
D_S_data = pd.read_excel("D_S_data.xlsx")
```

### Step 2: normalize the data

For flash:

```python
norm_I_S_data, I_S_metrics = Normalization.min_max(I_S_data, normalize_cols=[...])
norm_D_S_data, D_S_metrics = Normalization.min_max(D_S_data, normalize_cols=[...])
```

For PFR:

```python
norm_I_S_data, norm_D_S_data, I_S_metrics, D_S_metrics = Normalization.min_max_pfr(
    I_S_data=I_S_data,
    D_S_data=D_S_data,
    formulation="EFM",  # or ERM / CM
    species=species,
)
```

### Step 3: build residual objects

```python
physics = Physics(...)
boundary = Boundary(...)
```

### Step 4: create the data module and loaders

```python
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

### Step 5: define the neural network

```python
layer_size = [dim_in] + [64] * 6 + [dim_out]
model = ANN(layer_size, activation=nn.Tanh).to(device)
```

### Step 6: define optimizer and trainer

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.MSELoss()

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
```

### Step 7: train

```python
history = trainer.adam_step(epochs=75000, val_every=100, verbose=True)
```

### Step 8: save logs and figures

```python
Save.excel(history)
Save.csv(history)

plots = Plotter(history=history, val_every=100)
plots.plot_everything(savepath="figures", scale=1000)
```

---

## What each example folder typically contains

A case-study folder is expected to contain most or all of the following files:

### `data_gen.py`
Builds labeled `I_S` and `D_S` datasets.

- Flash examples: use Aspen Plus via COM automation.
- PFR examples: use SciPy ODE solves.

### `phys_res.py`
Defines the physics residual class and, when needed, the boundary residual class.

This is the main file you replace when adapting the framework to a new process.

### `main.py`
End-to-end training script for the given formulation.

### `check.py`
Post-training verification script.

Typical tasks performed by `check.py` include:

- loading `logs/best_model.pth`,
- evaluating the trained PINN over structured test profiles or grids,
- evaluating simulator/first-principles truth, and
- generating comparison plots and error summaries.

### `pfr_model.py` (PFR examples)
Contains the first-principles ODE model used for generating truth data and comparison profiles.

### `I_S_data.xlsx` and `D_S_data.xlsx`
Pre-generated labeled datasets used directly by `main.py`.

### `run.sh`
SLURM submission script used on HPC systems.

---

## Output files and folders

After a standard training run, the following outputs are typically created.

### `logs/best_model.pth`
Best checkpoint based on validation total loss.

### `logs/training_history.xlsx`
Multi-sheet Excel workbook containing stored histories.

### `logs/csv_files/`
CSV export of each history vector/matrix.

### `figures/`
Standardized training-history figures from `Plotter`.

### Additional comparison figures from `check.py`
Depending on the case, these may include:

- profile comparisons,
- absolute error plots,
- grouped error summaries,
- Aspen vs PINN phase-composition plots, and
- inverse-parameter trajectories.

---

## Data generation notes

## Flash examples

The flash `data_gen.py` files:

- sample operating conditions using LHS,
- sample feed compositions using a Dirichlet distribution,
- run Aspen simulations through COM automation,
- extract converged outputs, and
- write surrogate datasets to Excel.

Because Aspen COM automation is used, **flash data generation is Windows-specific in practice**.

## PFR examples

The PFR `data_gen.py` files:

- sample operating variables with LHS,
- enforce process-feasibility constraints such as minimum `O2/C6H6` ratio,
- solve the chosen first-principles reactor model with SciPy, and
- write the resulting surrogate datasets to Excel.

These examples are much easier to reproduce across platforms.

---

## How to extend `pinnse` to a new process system

The cleanest way to extend this repository is to create a new case folder modeled after one of the examples.

## Minimal recipe

1. **Create a new case directory**
   - Example: `myprocess/case1/`

2. **Define the process inputs and outputs**
   - Decide what goes into `I_S_data.xlsx` and `D_S_data.xlsx`
   - Keep naming conventions consistent and explicit.

3. **Write `data_gen.py`**
   - Generate or load labeled samples
   - Save them as `I_S_data.xlsx` and `D_S_data.xlsx`.

4. **Write `phys_res.py`**
   - Convert normalized tensors back to dimensional variables
   - compute residuals in physical space
   - return a residual tensor of shape `(batch_size, n_residuals)`.

5. **Write a `Boundary` class if needed**
   - Use this for inlet conditions, initial conditions, or boundary constraints.

6. **Write `main.py`**
   - normalize data,
   - construct loaders,
   - instantiate the network,
   - train,
   - save logs and plots.

7. **Write `check.py`**
   - reload the best checkpoint,
   - compare PINN predictions against truth,
   - generate figures and metrics.

## Practical guidance

- If your process has **compositions** that must sum to one, using `Z_`, `X_`, and `Y_` prefixes allows the collocation sampler to treat them as Dirichlet groups automatically.
- If your process has a **spatial or temporal boundary variable**, place it as the **last input column** if you want to use the default `bnd_colloc_loader()` without modification.
- Formulate physics residuals in **dimensional space** whenever possible. The current examples follow that pattern consistently.
- Keep normalization decisions aligned with the physics. The PFR cases deliberately use centered scaling because derivatives with respect to a coordinate-like input are easier to handle consistently in the residual.

---

## Choosing a normalization strategy

Use the following rule of thumb.

### Use `Normalization.min_max(...)` when:

- you want standard `[0, 1]` scaling,
- only selected columns should be normalized, or
- some outputs are already on physically convenient scales.

This is the pattern used in the flash examples.

### Use `Normalization.min_max_pfr(...)` when:

- the problem is derivative-driven,
- outputs/inputs have grouped physical meaning,
- you want centered `[-1, 1]` scaling, or
- you need formulation-aware normalization for flow variables and reaction variables.

This is the pattern used in the PFR examples.

---

## Reusing a trained model

The recommended evaluation pattern is:

```python
from pinnse import Analyze

model = Analyze.load_ann_pinn(
    dim_in=dim_in,
    dim_out=dim_out,
    depth=6,
    width=64,
    activation=nn.Tanh,
    ckpt_path="./logs/best_model.pth",
    device=device,
)
```

Then use:

- `Analyze.pinn_eval(...)` for flash-like `[0, 1]` normalization
- `Analyze.pinn_eval_pfr(...)` for PFR-like centered normalization.

---

## Platform notes

### Works cross-platform

- package installation
- training from the shipped `.xlsx` datasets
- all PFR training and checking scripts.

### Windows-only or Windows-dependent

- flash dataset generation with Aspen via `win32com`
- flash verification workflows that call Aspen-backed truth evaluation.

---

## Current status and scope

This repository already provides a strong reusable base for:

- forward PINN surrogates,
- formulation comparisons,
- physics and boundary residual coupling,
- adaptive loss balancing,
- inverse parameter estimation, and
- standardized training diagnostics.

At the same time, a few areas are clearly evolving, which is typical for an active research repository:

- `nonisopfr/` is currently a placeholder directory,
- the package metadata in `pyproject.toml` is still minimal,
- example dependencies are broader than the minimal dependency list in the package file, and
- the paper/citation metadata is not finalized yet.

That does not reduce the usefulness of the code; it simply means the repository is currently best understood as a **research framework with reusable infrastructure**, rather than a fully polished production package.

---

## Recommended first path for new users

If you are new to the repository, this is the most reliable learning path:

1. run `isopfr/efm/main.py`
2. inspect `isopfr/efm/phys_res.py`
3. inspect `pinnse/train.py`
4. run `isopfr/efm/check.py`
5. compare with `isopfr/erm/` and `isopfr/cm/`
6. only then move to the flash cases if you need Aspen-linked workflows.

This sequence lets you understand the reusable framework first, before dealing with simulator automation.

---

## Citation

If you use this repository in academic work, cite the corresponding manuscript once finalized.

For now, you may cite the repository directly in the software/methods section of your work and update the formal citation later when the paper is available.

---

## License

This project is distributed under the terms of the `LICENSE` file included in the repository.

---

## Contact

Repository maintainer listed in `pyproject.toml`:

- **Harshit Verma**
- `harshit.verma.che@gmail.com`

---

## Final note

The main strength of `pinnse` is not only that it trains PINNs, but that it provides a **repeatable framework for building process-specific PINNs without rewriting the entire pipeline each time**. If you want to add a new unit operation, a new surrogate formulation, or an inverse-learning problem, the intended workflow is to keep the reusable package intact and replace only the case-specific process modules.
