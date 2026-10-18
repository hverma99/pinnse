"""
Nox sessions for pinnse.
"""

import glob
import shutil

import nox

nox.options.sessions = ["docs"]

TORCH_CPU = "https://download.pytorch.org/whl/cpu"


@nox.session
def tests(session: nox.Session) -> None:
    """Run the test suite; fails if coverage drops below the configured 100%."""
    session.install("torch", "--index-url", TORCH_CPU)
    session.install("-e", ".", "pytest", "pytest-cov")
    session.run("pytest", *session.posargs)


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
