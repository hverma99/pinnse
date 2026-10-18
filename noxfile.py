"""Nox sessions for pinnse. Run `nox -s docs` to build the documentation."""

import nox

nox.options.sessions = ["docs"]

# CPU-only PyTorch is enough to import pinnse for autodoc, and much smaller.
TORCH_CPU = "https://download.pytorch.org/whl/cpu"


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
