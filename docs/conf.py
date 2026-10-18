"""Sphinx configuration for the pinnse documentation."""

import importlib.metadata

project = "pinnse"
author = "Harshit Verma"
copyright = "2026, Harshit Verma"
version = release = importlib.metadata.version("pinnse")

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
exclude_patterns = ["_build", "task.md", "Thumbs.db", ".DS_Store"]

# MyST: allow ::: fences, $...$ math, and auto-generated heading anchors.
myst_enable_extensions = ["colon_fence", "dollarmath"]
myst_heading_anchors = 3

# autodoc: document members in source order, with type hints in the description.
# Some classes document their arguments in __init__, so show both docstrings.
autoclass_content = "both"
# Do not fill undocumented methods (forward, __init__) with torch's docstrings.
autodoc_inherit_docstrings = False
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

# The pinnse docstrings use NumPy style, with an "Inputs" section for arguments.
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_custom_sections = [("Inputs", "params_style")]

# Link type hints such as torch.Tensor and pandas.DataFrame to their own docs.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    "torch": ("https://docs.pytorch.org/docs/stable", None),
}

html_theme = "furo"
html_title = f"pinnse {release}"
html_logo = "logo.png"
html_static_path = []
html_theme_options = {
    "source_repository": "https://github.com/hverma99/pinnse",
    "source_branch": "main",
    "source_directory": "docs/",
}
