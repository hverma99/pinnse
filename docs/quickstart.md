# Quick start

## Build a network

```python
import torch.nn as nn
from pinnse import ANN

# A feedforward PINN: 5 inputs -> 3 hidden layers (64 neurons each) -> 3 outputs
model = ANN(layer_size=[5, 64, 64, 64, 3], activation=nn.Tanh)
```

## Run a complete example

The isothermal PFR example is the fastest way to see `pinnse` in action. It is
self-contained and does not require Aspen Plus.

```bash
git clone https://github.com/hverma99/pinnse.git
cd pinnse/examples/isopfr/efm

python main.py     # Train the PINN surrogate
python check.py    # Evaluate and compare against the first-principles solution
```

This will:

1. Load labelled input/output data (`I_S_data.xlsx`, `D_S_data.xlsx`).
2. Normalize the input and output spaces.
3. Construct supervised, physics-collocation, and boundary-collocation data
   loaders.
4. Build and train a PINN surrogate with data, physics, and boundary losses.
5. Checkpoint the best model to `logs/`.
6. Evaluate the trained model and generate comparison figures.

## Walkthrough: training a PINN surrogate

This walkthrough follows the **isothermal PFR, effluent flowrate model (EFM)**
example. The full code is in `examples/isopfr/efm/main.py`.

### Step 1: Load and normalize data

```python
import pandas as pd
from pinnse import Normalization

I_S_data = pd.read_excel("I_S_data.xlsx")  # Inputs:  inlet flowrates, P, T, V
D_S_data = pd.read_excel("D_S_data.xlsx")  # Outputs: outlet flowrates

# Normalize to [-1, 1] with formulation-aware scaling
norm_I_S, norm_D_S, I_S_metrics, D_S_metrics = Normalization.min_max_pfr(
    I_S_data=I_S_data,
    D_S_data=D_S_data,
    formulation="EFM",
    species=["O2", "CO2", "H2O", "C6H6", "C4H2O3"],
)
```

See {doc}`normalization` for the other scaling strategies.

### Step 2: Create data loaders

```python
from pinnse import DataModule

data = DataModule(
    I_S_data=norm_I_S,
    D_S_data=norm_D_S,
    labeled_data_batch_size=500,
    physics_coll_data_size=20000,  # Number of collocation points
    physics_coll_batch_size=500,
    boundary_coll_data_size=5000,
    boundry_coll_batch_size=1000,
    test_frac=0.1,
    val_frac=0.1,
)

train_loader, val_loader, test_loader = data.labeled_data_loader()
phys_coll_loader = data.phys_colloc_loader()  # LHS + Dirichlet sampling
bnd_coll_loader = data.bnd_colloc_loader()  # Boundary-fixed sampling
```

### Step 3: Define physics and boundary residuals

The residuals live in a separate, process-specific `phys_res.py` file. Each
residual is a callable with signature `(x, y) -> residual_tensor`, where `x` is
the normalized input batch and `y` is the network output.

```python
from phys_res import Physics, Boundary

physics = Physics(I_S_metrics, D_S_metrics, species, nu, key_species, k, E)
boundary = Boundary(species)
```

{doc}`extending` gives a template for writing your own residuals.

### Step 4: Build the model

```python
import torch, torch.nn as nn
from pinnse import ANN

dim_in = I_S_data.shape[1]  # Number of input features
dim_ot = D_S_data.shape[1]  # Number of output targets

layer_size = [dim_in] + [64] * 6 + [dim_ot]  # 6 hidden layers, 64 neurons
model = ANN(layer_size, activation=nn.Tanh).to(device)
```

### Step 5: Train

```python
from pinnse import Training

training = Training(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    optimizer=torch.optim.Adam(model.parameters(), lr=1e-4),
    loss_fn=nn.MSELoss(),
    device=device,
    phys_coll_loader=phys_coll_loader,
    bnd_coll_loader=bnd_coll_loader,
    phys_residual=physics,
    bnd_residual=boundary,
    ckpt_path="./logs/best_model.pth",
    phys_weight=1.0,  # Weight for physics loss
    bnd_weight=1.0,  # Weight for boundary loss
    adapt_wts=False,  # Set True for adaptive gradient-norm balancing
)

history = training.adam_step(epochs=75000, val_every=100)
```

For problems that benefit from second-order optimization, append an L-BFGS
refinement stage:

```python
history_lbfgs = training.lbfgs_step(mode="combined", N_LBFGS=150)
```

### Step 6: Save and visualize

```python
from pinnse import Save, Plotter

Save.excel(history)
Save.csv(history)

plots = Plotter(history=history, val_every=100)
plots.plot_everything(savepath="figures/", scale=1000)
```
