"""
Run an example script with its training cut to a few epochs, as a smoke test.

    python examples/smoke.py [--epochs N] [--lbfgs-steps M] SCRIPT [ARGS ...]

SCRIPT (an example's ``main.py`` or ``check.py``) runs unchanged as
``__main__`` from its own directory, except that every ``Training.adam_step``
stops after N epochs (default 5), with its validation and checkpoint periods
shortened to fit, and every ``Training.lbfgs_step`` after M steps (default 2).
A passing run shows that the example still works end to end, not that the
model it trains is accurate. ``nox -s examples`` runs every example this way.
"""

import argparse
import functools
import inspect
import os
import runpy
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # as the examples set it
import torch  # noqa: F401  # load before NumPy, see pyproject.toml

from pinnse import Training


def _cap(name: str, adjust) -> None:
    """Replace `Training.<name>` by a version whose arguments pass `adjust`."""
    method = getattr(Training, name)
    signature = inspect.signature(method)

    @functools.wraps(method)
    def capped(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        adjust(bound.arguments)
        return method(*bound.args, **bound.kwargs)

    setattr(Training, name, capped)


def cap_training(epochs: int, lbfgs_steps: int) -> None:
    """Limit every later `adam_step` to `epochs` and `lbfgs_step` to `lbfgs_steps`."""

    def adam(arguments: dict) -> None:
        arguments["epochs"] = min(arguments["epochs"], epochs)
        for period in ("val_every", "save_every"):
            if arguments.get(period) is not None:
                arguments[period] = min(arguments[period], arguments["epochs"])

    def lbfgs(arguments: dict) -> None:
        key = "N_LBFGS" if arguments.get("N_LBFGS") is not None else "n_steps"
        arguments[key] = min(arguments.get(key) or lbfgs_steps, lbfgs_steps)

    _cap("adam_step", adam)
    _cap("lbfgs_step", lbfgs)


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {value}")
    return value


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run a pinnse example with training cut to a few epochs."
    )
    parser.add_argument("--epochs", type=_positive_int, default=5)
    parser.add_argument("--lbfgs-steps", type=_positive_int, default=2)
    parser.add_argument("script", type=Path, help="example script to run")
    parser.add_argument("args", nargs=argparse.REMAINDER, help="its arguments")
    options = parser.parse_args(argv)

    script = options.script.resolve()
    if not script.is_file():
        parser.error(f"no such script: {options.script}")

    cap_training(options.epochs, options.lbfgs_steps)

    # Run the script as `python SCRIPT` would from its own directory, where it
    # reads its data and imports its sibling modules.
    os.chdir(script.parent)
    sys.path.insert(0, str(script.parent))
    sys.argv = [str(script), *options.args]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
