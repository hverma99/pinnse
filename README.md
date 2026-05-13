# Physics-Informed Neural Networks for Process Systems Engineering (`pinnse`)

This repository provides `pinnse`, a Python framework for building and training Physics-Informed Neural Networks (PINNs) tailored for applications in Process Systems Engineering (PSE). This framework is designed to facilitate the integration of physical laws and domain knowledge into deep learning models, enabling more accurate and generalizable predictions for complex process systems.

## Abstract

Traditional data-driven models in PSE often require large amounts of labeled data and may not generalize well outside the training distribution. Physics-Informed Neural Networks (PINNs) offer a promising alternative by incorporating physical principles, expressed as partial differential equations (PDEs) or other governing equations, directly into the neural network's loss function. This allows the model to learn from both data and the underlying physics, leading to better performance, especially in data-scarce scenarios. The `pinnse` package provides a flexible and extensible framework for developing such models for a wide range of PSE problems.

## Key Features

*   **Flexible Neural Network Architectures**: Choose from standard `ANN`, `BranchedANN` for multi-output predictions, and `Fourier_ANN` for problems requiring feature engineering.
*   **Versatile Data Handling**: `DataModule` for seamless management of labeled data, and generation of collocation points for physics-based loss terms.
*   **Advanced Training Capabilities**: The `Training` class supports complex training loops with combined losses from supervised data, physics residuals, and boundary conditions. It also includes adaptive loss weighting to balance different loss components during training.
*   **Comprehensive Utilities**: Includes tools for data normalization, result saving, and plotting to streamline the end-to-end workflow.
*   **Example-driven**: Comes with examples for flash separation and plug flow reactors (PFR) to demonstrate the usage of the package.

## Installation

To install the `pinnse` package, clone this repository and install it using pip:

```bash
git clone https://github.com/your-username/pinnse.git
cd pinnse
pip install .
```

## How It Works

The core idea behind the `pinnse` framework is to train a neural network that approximates the solution of a system of differential equations. The loss function is composed of several terms:

1.  **Data Loss**: A standard supervised loss term (e.g., Mean Squared Error) that measures the discrepancy between the model's predictions and the available labeled data.
2.  **Physics Residual Loss**: This term penalizes the model if its predictions do not satisfy the governing physical laws (e.g., mass and energy balances). The residuals of the differential equations are evaluated at a set of collocation points distributed throughout the domain.
3.  **Boundary Condition Loss**: This term enforces the boundary conditions of the problem.

The total loss is a weighted sum of these components. The `pinnse` package provides tools to define these components and train the network to minimize the total loss.

## Package Structure

The `pinnse` package is organized into several modules:

-   `pinnse/PINNs.py`: Contains the definitions for the neural network architectures.
    -   `ANN`: A standard fully-connected feedforward neural network. Suitable for general-purpose function approximation.
    -   `BranchedANN`: A network with a shared trunk and multiple output heads. Useful when different outputs are related but have distinct characteristics.
    -   `Fourier_ANN`: A feedforward network with a Fourier feature mapping layer. This can help the network learn high-frequency components in the solution.
-   `pinnse/data.py`: Includes the `DataModule` and `DataLoader` classes. They are responsible for handling data loading, preprocessing, and generating collocation points for physics-informed training.
-   `pinnse/train.py`: Provides the `Training` class, which orchestrates the model training process. It manages the training loop, computes the different loss components (supervised data loss, physics and boundary collocation losses), and implements adaptive loss weighting to balance their contributions.
-   `pinnse/plots.py`: Contains the `Plotter` class for visualizing training history (e.g., loss curves) and model predictions.
-   `pinnse/utils.py`: Offers a collection of utility classes and functions for common tasks such as:
    -   `Normalization`: For scaling and un-scaling data.
    -   `Denormalization`: For reversing the normalization.
    -   `Save`: For saving training history and model checkpoints.
    -   `Analyze`: For performing analysis on the results.

## Usage

The following steps outline a typical workflow for using the `pinnse` package, based on the examples provided in the `flash` and `isopfr` directories.

### Step 1: Import necessary modules

```python
import torch
import torch.nn as nn
import pandas as pd
from pinnse import (
    Normalization,
    DataModule,
    ANN,
    Training,
    Plotter,
    Save,
)
from phys_res import Physics # User-defined physics residual
```

### Step 2: Load and preprocess data

Load your input (`I_S_data`) and output (`D_S_data`) datasets. Normalize the data using the `Normalization` utility.

```python
I_S_data = pd.read_excel("I_S_data.xlsx")
D_S_data = pd.read_excel("D_S_data.xlsx")

norm_I_S_data, I_S_metrics = Normalization.min_max(I_S_data)
norm_D_S_data, D_S_metrics = Normalization.min_max(D_S_data)
```

### Step 3: Define the physics-informed residual

Create a class or function that defines the physics-based residual equations. This will be used to enforce physical constraints during training.

```python
# Example from flash/case1/phys_res.py
class Physics:
    def __init__(self, I_S_metrics, D_S_metrics, species):
        # ... initialization ...
    def __call__(self, x, model):
        # ... calculate residuals ...
        return residuals
```

### Step 4: Set up the data module

Create an instance of `DataModule` to manage the training, validation, and test data, as well as the physics collocation points.

```python
data = DataModule(
    I_S_data=norm_I_S_data,
    D_S_data=norm_D_S_data,
    labeled_data_batch_size=500,
    physics_coll_data_size=25000,
    physics_coll_batch_size=500,
    test_frac=0.1,
    val_frac=0.1,
)

train_loader, val_loader, test_loader = data.labeled_data_loader()
phys_coll_loader = data.phys_colloc_loader()
```

### Step 5: Define the neural network model

Instantiate one of the neural network architectures from `pinnse.PINNs`.

```python
dim_in = I_S_data.shape[1]
dim_out = D_S_data.shape[1]
layer_size = [dim_in] + [64] * 6 + [dim_out]

model = ANN(layer_size, activation=nn.Tanh).to(device)
```

### Step 6: Configure and run the training

Create a `Training` object and call the `adam_step` method to start the training.

```python
physics = Physics(I_S_metrics, D_S_metrics, species)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

trainer = Training(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    phys_coll_loader=phys_coll_loader,
    optimizer=optimizer,
    loss_fn=nn.MSELoss(),
    device=device,
    phys_residual=physics,
    phys_weight=1.0,
)

history = trainer.adam_step(epochs=50000)
```

### Step 7: Post-process and visualize results

Use the `Plotter` and `Save` utilities to visualize and save the training results.

```python
Save.excel(history)
plots = Plotter(history=history)
plots.plot_everything(savepath="figures")
```

## Examples

The `flash` and `isopfr` directories contain example projects that demonstrate how to use the `pinnse` package for different PSE problems:
- **Flash Separation**: A flash drum model.
- **Isothermal Plug Flow Reactor (PFR)**: A model of an isothermal PFR.

These examples provide a practical guide to setting up and solving problems with `pinnse`.

## Dependencies

The package requires the following dependencies:

-   numpy
-   scipy
-   torch
-   pandas
-   tqdm

## Citation

If you use this code in your research, please cite the following paper:

```
[Paper citation will be added here once it is available]
```

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue.

## License

This project is licensed under the terms of the LICENSE file.
