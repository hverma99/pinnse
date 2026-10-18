# Dynamic processes

For a dynamic process, the surrogate predicts a whole state trajectory from an
initial condition and an input schedule. `pinnse` provides a recurrent network,
{class}`~pinnse.RecurrentANN`, and a matching data module,
{class}`~pinnse.SequenceDataModule`.

## The recurrent network

`RecurrentANN` is a sequence-to-sequence network. Select the cell with
`cell="rnn"`, `"gru"` or `"lstm"`, stack cells with `n_layers`, and configure
the output head, applied at each time step, with `head_layers`. It maps inputs
of shape `(batch, seq_len, in_dim)` to outputs of shape
`(batch, seq_len, out_dim)`.

`RecurrentANN` starts from a zero hidden state, so the initial condition must be
supplied as input features, typically held constant along the sequence. Its
forward pass takes a single tensor, so it works in the same
{class}`~pinnse.Training` loop as the feedforward networks.

## Sequence data

`SequenceDataModule` is the counterpart of {class}`~pinnse.DataModule` for
trajectory data, where each sample is a whole trajectory rather than an
independent row.

| Feature | Behavior |
|---|---|
| Input format | Normalized NumPy arrays of shape `(n_trajectories, seq_len, n_features)` for both `I_S_data` and `D_S_data`. |
| `labeled_data_loader()` | Splits by whole trajectory, so no trajectory contributes to more than one of the train/validation/test partitions. `labeled_data_batch_size` counts trajectories. |
| `phys_colloc_loader()` | Generates collocation *trajectories* by Latin Hypercube Sampling. Context features (constant along a trajectory, such as the initial condition) are sampled once per trajectory. Driving features (such as a coolant temperature) are sampled as `n_segments` levels and expanded into a piecewise-constant schedule. |
| `context_cols` | Indices of the context features. If omitted, they are detected as the features that never vary within a trajectory. |

With sequence data, the physics residual receives the input and output
sequences, `x` of shape `(batch, seq_len, dim_in)` and `y` of shape
`(batch, seq_len, dim_ot)`, so it can enforce a discrete-time relation between
successive predicted states. There is no boundary-collocation loader: initial
conditions enter through the context features.

## Example

```python
import torch.nn as nn
from pinnse import RecurrentANN, SequenceDataModule

# norm_I_S: (n_trajectories, 200, 3), features [CA_0, T_0, TC_k], normalized
# norm_D_S: (n_trajectories, 200, 2), features [CA_k1, T_k1], normalized
data = SequenceDataModule(
    I_S_data=norm_I_S,
    D_S_data=norm_D_S,
    labeled_data_batch_size=64,  # trajectories per batch
    physics_coll_data_size=1000,  # collocation trajectories
    physics_coll_batch_size=64,
    n_segments=4,  # piecewise-constant coolant schedule
    test_frac=0.1,
    val_frac=0.1,
)

train_loader, val_loader, test_loader = data.labeled_data_loader()
phys_coll_loader = data.phys_colloc_loader()

model = RecurrentANN(in_dim=3, hidden_dim=64, out_dim=2, cell="gru", head_layers=[64])
```

The full workflow, including the unrolled discrete-time residual, is in
`examples/dynamic.cstr` (see {doc}`examples`).
