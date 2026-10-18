# pinnse

```{image} logo.png
:alt: pinnse logo
:width: 360px
:align: center
```

`pinnse` is a modular PyTorch framework for building **physics-informed
neural-network (PINN)** surrogate models of chemical processes. It separates
the reusable machine-learning infrastructure (data handling, normalization,
network construction, training, checkpointing, and visualization) from the
process-specific ingredients that you define once for each system: operating
bounds, governing equations, and residual formulations.

```{image} framework_overview.png
:alt: pinnse framework overview
```

The workflow has three steps:

1. **Define the process physics** as residual functions (material balances,
   energy balances, equilibrium relations).
2. **Train a differentiable neural-network surrogate** against both labelled
   data and those physical constraints.
3. **Obtain a fast, physically consistent model** for simulation, design, or
   optimization.

```bash
pip install pinnse
```

New to `pinnse`? Start with {doc}`installation`, then follow the
{doc}`quickstart`, which trains a surrogate of an isothermal plug-flow reactor.

```{toctree}
:maxdepth: 2
:caption: Getting started
:hidden:

installation
quickstart
```

```{toctree}
:maxdepth: 2
:caption: User guide
:hidden:

framework
normalization
dynamic
examples
extending
```

```{toctree}
:maxdepth: 2
:caption: Reference
:hidden:

api/index
```

## Citation

If you use `pinnse`, please cite the associated paper:

> Verma, H. and Maravelias, C.T. "A Generalized Framework for Physics-Informed
> Neural Networks in Process Systems Engineering." *Computers & Chemical
> Engineering* (submitted).

## License

`pinnse` is released under the
[MIT License](https://github.com/hverma99/pinnse/blob/main/LICENSE).
