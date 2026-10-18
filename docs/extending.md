# Extending pinnse to a new process

To apply `pinnse` to your own unit operation or process, follow these steps.

1. **Define the input and output spaces.** Identify the process inputs (`I_S`:
   feed conditions, operating parameters) and outputs (`D_S`: product
   compositions, temperatures).
2. **Generate or collect labelled data.** Run your process simulator, solve the
   ODE or algebraic system, or compile experimental data. Save the results as
   `.xlsx` or `.csv` files.
3. **Write `phys_res.py`.** Implement the governing equations as residual
   functions. The training loop calls them on collocation batches during each
   epoch.
4. **Define boundary residuals** if applicable. For ODEs, this is typically the
   initial or inlet condition.
5. **Write `main.py`.** Configure the architecture, data loaders, optimizer,
   scheduler, and loss weights, and call {meth}`Training.adam_step
   <pinnse.Training.adam_step>`.
6. **Train and evaluate.** Run `main.py` to train, then write a `check.py`
   script that loads the checkpoint and compares predictions against reference
   solutions.

:::{note}
For dynamic processes, use {class}`~pinnse.RecurrentANN` with
{class}`~pinnse.SequenceDataModule`. The residual then receives sequences of
shape `(batch, seq_len, n_features)` rather than the `(batch_size, n_inputs)`
batches in the template below. See {doc}`dynamic`.
:::

## Template: a minimal `phys_res.py`

Replace the residual computation with your governing equations.

```python
import torch
from pinnse import Denormalization


class Physics:
    def __init__(self, I_S_metrics, D_S_metrics):
        self.I_S_metrics = I_S_metrics
        self.D_S_metrics = D_S_metrics

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the physics residual.

        Parameters
        ----------
        x : torch.Tensor
            Normalized input batch (batch_size, n_inputs).
        y : torch.Tensor
            Network output in normalized space (batch_size, n_outputs).

        Returns
        -------
        torch.Tensor
            Residual tensor (batch_size, n_residuals). Training minimizes
            the mean squared residual.
        """
        # 1. Denormalize outputs (and inputs if needed) to dimensional space
        # y_dim, y_rng = Denormalization.min_max_pfr(y, self.D_S_metrics, keys=[...])

        # 2. Compute derivatives via autograd (if the equations involve dy/dx)
        # dy_dx = torch.autograd.grad(y[:, 0:1], x, grad_outputs=...,
        #                             create_graph=True)[0][:, -1:]

        # 3. Evaluate the governing equation residual
        # residual = (left-hand side) - (right-hand side)

        # 4. Return the residual (optionally scaled for numerical balance)
        # return residual
        raise NotImplementedError("Replace with your governing equations.")


class Boundary:
    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the boundary residual (e.g., inlet condition: y_out = y_in at x=0).
        """
        # residual = x[:, :n] - y[:, :n]   # Example: output equals input at boundary
        raise NotImplementedError("Replace with your boundary condition.")
```
