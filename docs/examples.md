# Example case studies

The [`examples/`](https://github.com/hverma99/pinnse/tree/main/examples)
directory of the repository contains complete, self-contained workflows.

:::{tip}
Begin with `examples/isopfr/efm`. It is the simplest complete example, requires
no external software, and demonstrates all core features: data loading,
normalization, physics residuals, boundary conditions, training, and
evaluation. The {doc}`quickstart` walks through it step by step.
:::

| Example | Description | Formulation |
|---|---|---|
| `flash/case1` | Flash separation with supplied labelled data. | VLE: material balance and mole-fraction sum constraints |
| `flash/case2` | Alternative flash formulation. | VLE with a different output structure |
| `isopfr/efm` | Isothermal PFR, effluent flowrate model. | ODE material balance: $dF/dV = \nu \cdot r$ |
| `isopfr/erm` | Isothermal PFR, extent of reaction model. | ODE in terms of reaction extents |
| `isopfr/cm/s0` | Isothermal PFR, conversion model (baseline). | ODE in terms of conversions |
| `isopfr/cm/s1` | Isothermal PFR, conversion model with learning-rate scheduling. | Same as `s0`, with a `StepLR` scheduler (step 5000, γ = 0.75) |
| `isopfr/efm.inverse` | Inverse PINN for parameter estimation. | Estimates activation energies from data with a trainable `nn.Parameter` |
| `isopfr/efm.noise` | Isothermal PFR (EFM) trained on labelled outputs perturbed by Gaussian measurement noise ({meth}`Noise.gaussian <pinnse.Noise.gaussian>`). Setting `phys_weight` and `bnd_weight` to `0.0` gives a purely data-driven baseline with the same data, architecture and seed. | Same as `efm`; `check.py` evaluates against the clean labels and the first-principles profile |
| `nonisopfr` | Nonisothermal PFR with coupled mass and energy balances. | Coupled ODEs: mass balance and energy balance with heat exchange |
| `dynamic.cstr` | Recurrent PINN ({class}`~pinnse.RecurrentANN`, {class}`~pinnse.SequenceDataModule`) for a nonisothermal CSTR. It predicts the state trajectory (concentration and temperature over 200 steps) from an initial condition and a piecewise-constant coolant schedule. `check.py` compares it with a per-step feedforward network, an autoregressive single-transition model, and the best accuracy any function of the instantaneous input can reach. | Unrolled discrete-time residual between successive predicted states; `SCHEME` selects Euler, backward Euler, trapezoid or RK4, and `CELL` selects RNN, GRU or LSTM |

## Files in an example

| File | Purpose |
|---|---|
| `main.py` | Configuration and training script |
| `phys_res.py` | Physics and boundary residual definitions |
| `data_gen.py` | Dataset generation script |
| `check.py` | Post-training evaluation and comparison |
| `pfr_model.py` / `cstr_model.py` | Process model helper functions (where applicable) |
| `I_S_data.xlsx` | Labelled input data |
| `D_S_data.xlsx` | Labelled output data |
| `run.sh` | Convenience shell script |
