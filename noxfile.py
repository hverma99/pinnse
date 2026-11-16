"""
Nox sessions for pinnse.
"""

import glob
import os
import shutil
import time
import tomllib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import nox
from packaging.requirements import Requirement

nox.options.sessions = ["docs"]

TORCH_CPU = "https://download.pytorch.org/whl/cpu"

# Examples whose check.py evaluates the model against Aspen Plus through
# pywin32, so only their main.py runs in the examples session.
EXAMPLES_WITHOUT_CHECK = {"flash/case1", "flash/case2"}


@nox.session
def tests(session: nox.Session) -> None:
    """Run the test suite; fails if coverage drops below the configured 100%."""
    session.install("torch", "--index-url", TORCH_CPU)
    session.install("-e", ".", "pytest", "pytest-cov")
    session.run("pytest", *session.posargs)


def minimum_versions() -> dict[str, str]:
    """Map every dependency in pyproject.toml to its declared minimum version."""
    with open("pyproject.toml", "rb") as f:
        dependencies = tomllib.load(f)["project"]["dependencies"]
    minimums = {}
    for line in dependencies:
        requirement = Requirement(line)
        lower = [s.version for s in requirement.specifier if s.operator == ">="]
        if len(lower) != 1:
            raise ValueError(f"{line!r} must declare exactly one '>=' minimum")
        minimums[requirement.name] = lower[0]
    return minimums


@nox.session(python="3.11", venv_backend="uv|virtualenv")
def minimums(session: nox.Session) -> None:
    """Run the test suite with every dependency at its declared minimum version."""
    minimums = minimum_versions()
    constraints = Path(session.create_tmp()) / "minimums.txt"
    constraints.write_text("".join(f"{n}=={v}\n" for n, v in minimums.items()))

    session.install(f"torch=={minimums['torch']}", "--index-url", TORCH_CPU)
    session.install("-c", str(constraints), "-e", ".", "pytest", "pytest-cov")
    session.run(
        "python",
        "-c",
        "import sys, importlib.metadata as m; "
        "[print(n, m.version(n)) for n in sys.argv[1:]]",
        *minimums,
    )
    session.run("pytest", *session.posargs)


@nox.session
def examples(session: nox.Session) -> None:
    """
    Smoke-test the examples in parallel: run each main.py, then its check.py,
    in a copy of examples/ with training cut to a few epochs (see
    examples/smoke.py). Pass example directories to run only those, e.g.
    ``nox -s examples -- nonisopfr flash/case1``.
    """
    session.install("torch", "--index-url", TORCH_CPU)
    session.install("-e", ".")

    root = Path(session.create_tmp()).resolve() / "examples"
    shutil.rmtree(root, ignore_errors=True)
    shutil.copytree(
        "examples",
        root,
        ignore=shutil.ignore_patterns(
            "logs", "figures", "compare", "plots", "__pycache__", "*.out"
        ),
    )

    found = {p.parent.relative_to(root).as_posix() for p in root.glob("**/main.py")}
    selected = set(session.posargs) or found
    if unknown := selected - found:
        session.error(f"No example in {sorted(unknown)}; choose from {sorted(found)}")

    # The examples run in parallel, one per CPU on a single thread each, without
    # a GUI backend for matplotlib; a script's output is shown only if it fails.
    env = {"MPLBACKEND": "Agg", "OMP_NUM_THREADS": "1"}

    def smoke(name: str) -> None:
        scripts = ["main.py"]
        if name not in EXAMPLES_WITHOUT_CHECK and (root / name / "check.py").is_file():
            scripts.append("check.py")
        for script in scripts:
            start = time.perf_counter()
            session.run(
                "python",
                str(root / "smoke.py"),
                str(root / name / script),
                env=env,
                silent=True,
            )
            session.log(
                f"examples/{name}/{script} passed in "
                f"{time.perf_counter() - start:.0f} s"
            )

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as pool:
        runs = {name: pool.submit(smoke, name) for name in sorted(selected)}
    if failed := [name for name, run in runs.items() if run.exception()]:
        session.error(f"Examples failed: {', '.join(failed)}")


@nox.session
def build(session: nox.Session) -> None:
    """Build the sdist and wheel into dist/, validate them, and import the wheel."""
    session.install("build", "twine")
    shutil.rmtree("dist", ignore_errors=True)
    session.run("python", "-m", "build")
    session.run("twine", "check", "--strict", *glob.glob("dist/*"))

    session.install("torch", "--index-url", TORCH_CPU)
    session.install(*glob.glob("dist/*.whl"))
    session.chdir(session.create_tmp())
    session.run(
        "python",
        "-c",
        "import pinnse; "
        "missing = [n for n in pinnse.__all__ if not hasattr(pinnse, n)]; "
        "assert not missing, missing; "
        "print('pinnse', pinnse.__file__)",
    )


@nox.session
def docs(session: nox.Session) -> None:
    """Build the HTML documentation into docs/_build/html."""
    session.install("torch", "--index-url", TORCH_CPU)
    session.install("-e", ".[docs]")
    session.chdir("docs")
    session.run("sphinx-build", "-M", "html", ".", "_build", "-W", "--keep-going")


@nox.session
def serve(session: nox.Session) -> None:
    """Build the documentation and serve it at http://localhost:8000."""
    session.notify("docs")
    session.log("Serving docs/_build/html at http://localhost:8000 (Ctrl+C to stop)")
    session.run("python", "-m", "http.server", "8000", "-d", "docs/_build/html")
