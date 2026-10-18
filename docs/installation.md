# Installation

`pinnse` requires Python 3.10 or later.

## Basic install

```bash
pip install pinnse
```

This installs the core dependencies: PyTorch, NumPy, SciPy, pandas, tqdm,
scikit-learn, matplotlib, and openpyxl.

Verify the installation:

```bash
python -c "from pinnse import ANN, DataModule, Training; print('pinnse is ready')"
```

## Use a clean environment

```bash
conda create -n pinnse python=3.11 -y
conda activate pinnse
pip install pinnse
```

## Development install

To run the examples or modify the source, clone the repository and install it
in editable mode:

```bash
git clone https://github.com/hverma99/pinnse.git
cd pinnse
pip install -e ".[dev]"
```

## GPU support

Training PINNs is compute-intensive, and a CUDA-enabled GPU is strongly
recommended. `pip install pinnse` installs the default (CPU) PyTorch build. To
enable GPU acceleration, install the PyTorch build for your CUDA version, before
or after installing `pinnse`:

```bash
# Example: PyTorch with CUDA 12.6 (see https://pytorch.org/get-started for your version)
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

`pinnse` uses the GPU when one is available. All example scripts select the
device with:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

## Optional: Aspen Plus integration

Flash-separation data generation uses Aspen Plus COM automation (Windows only).
To install the optional dependency:

```bash
pip install "pinnse[aspen]"
```

:::{note}
Aspen Plus is needed only to *regenerate* the flash datasets. The supplied
`.xlsx` datasets can be used for training without Aspen.
:::

## Building the documentation

The documentation is built with Sphinx. Install the `docs` extra and build the
HTML pages:

```bash
pip install -e ".[docs]"
sphinx-build -M html docs docs/_build
```

Or, with [nox](https://nox.thea.codes), run `nox -s docs` to build, or
`nox -s serve` to build and serve the pages at <http://localhost:8000>.
