import os
import sys
import tomllib
from pathlib import Path

sys.path.insert(0, os.path.abspath(".."))

project = "Atomic Agents"
copyright = "2026, Kenny Vaneetvelde"
author = "Kenny Vaneetvelde"

with open(Path(__file__).resolve().parent.parent / "pyproject.toml", "rb") as f:
    version = tomllib.load(f)["project"]["version"]
release = version

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinxcontrib.mermaid",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# Intersphinx configuration
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pydantic": ("https://docs.pydantic.dev/latest/", None),
}

# MyST configuration
myst_enable_extensions = [
    "colon_fence",
    "deflist",
]

# Autodoc configuration
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}
