from ._deprecation import warn_renamed
from .data import DataModule, SequenceDataModule
from .PINNs import ANN, SANN, FourierANN, MultiHeadANN, RecurrentANN
from .PINNs import Fourier_ANN as Fourier_ANN  # deprecated alias, removed in 2.0
from .plots import Plotter
from .train import PCGrad, Training
from .utils import (
    Analyze,
    Denormalization,
    Noise,
    Normalization,
    Save,
    seed_everything,
)

__all__ = [
    "ANN",
    "SANN",
    "MultiHeadANN",
    "FourierANN",
    "RecurrentANN",
    "DataModule",
    "SequenceDataModule",
    "Training",
    "PCGrad",
    "Plotter",
    "Normalization",
    "Denormalization",
    "Save",
    "Analyze",
    "Noise",
    "seed_everything",
]


def __getattr__(name: str):
    # pinnse.DataLoader was a re-export of torch.utils.data.DataLoader.
    if name == "DataLoader":
        warn_renamed("pinnse.DataLoader", "torch.utils.data.DataLoader")
        from torch.utils.data import DataLoader

        return DataLoader
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
