"""Configuration file for the Sphinx documentation builder.

For the full list of built-in configuration values, see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

import os
import sys
from typing import Any

# Add project root to path
sys.path.insert(0, os.path.abspath("../../src"))

# Import version from package (which reads from pyproject.toml)
import vla_metrics  # noqa: E402

# -- Project information -----------------------------------------------------
project = "VLA Metrics"
copyright = "2025, Ameya Wagh"
author = "Ameya Wagh"
# Version is read from pyproject.toml via vla_metrics.__version__
version = vla_metrics.__version__
release = vla_metrics.__version__

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx_autodoc_typehints",
    "myst_parser",
]

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"

# Autosummary settings
autosummary_generate = True

# Napoleon settings (for Google and NumPy style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# MyST settings (for Markdown support)
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "html_image",
]

# Suppress warnings for missing cross-references in included markdown files
suppress_warnings = ["myst.xref_missing"]

# Templates
templates_path = ["_templates"]

# Source file suffixes
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# Exclude patterns
exclude_patterns: list[str] = []

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "navigation_depth": 4,
    "collapse_navigation": False,
    "sticky_navigation": True,
    "includehidden": True,
    "titles_only": False,
}

html_static_path: list[str] = []  # No static files for now
html_logo = None
html_favicon = None

# -- Options for LaTeX output ------------------------------------------------
latex_elements: dict[str, Any] = {}
latex_documents = [
    ("index", "vla-metrics.tex", "VLA Metrics Documentation", "Ameya Wagh", "manual"),
]

# -- Options for manual page output ------------------------------------------
man_pages = [("index", "vla-metrics", "VLA Metrics Documentation", [author], 1)]

# -- Options for Texinfo output ----------------------------------------------
texinfo_documents = [
    (
        "index",
        "vla-metrics",
        "VLA Metrics Documentation",
        author,
        "vla-metrics",
        "TorchMetrics-based evaluation metrics for VLA models",
        "Miscellaneous",
    ),
]
