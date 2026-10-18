# API reference

Every public class is importable directly from the `pinnse` package:

```python
from pinnse import (
    ANN,
    SANN,
    MultiHeadANN,
    Fourier_ANN,  # Feedforward architectures
    RecurrentANN,  # Sequence architecture
    DataModule,
    SequenceDataModule,  # Data loading
    Training,
    PCGrad,  # Training loop, gradient aggregation
    Plotter,  # Visualization
    Normalization,
    Denormalization,  # Scaling
    Save,
    Analyze,
    Noise,  # Utilities
)
```

```{toctree}
:maxdepth: 2

pinns
data
train
utils
plots
```
