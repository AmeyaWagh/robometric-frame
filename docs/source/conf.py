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
import robometric_frame  # noqa: E402

# -- Project information -----------------------------------------------------
project = "FRAME: Framework for Robotic Action and Motion Evaluation"
copyright = "2025, Ameya Wagh"
author = "Ameya Wagh"
# Version is read from pyproject.toml via robometric_frame.__version__
version = robometric_frame.__version__
release = robometric_frame.__version__

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
suppress_warnings = ["myst.xref_missing", "image.not_readable"]

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

html_static_path: list[str] = []
html_logo = None
html_favicon = None


def setup(app: Any) -> None:
    """Copy logo to build output during build."""
    import shutil
    from pathlib import Path

    def copy_logo(app: Any, exception: Any) -> None:
        if exception is not None:
            return
        # Source: docs/frame-logo.png (relative to project root)
        src = Path(__file__).parent.parent / "frame-logo.png"
        # Destination: {outdir}/docs/frame-logo.png
        dst_dir = Path(app.outdir) / "docs"
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / "frame-logo.png"
        if src.exists():
            shutil.copy2(src, dst)

    app.connect("build-finished", copy_logo)


# -- Options for LaTeX output ------------------------------------------------
latex_elements: dict[str, Any] = {
    # The paper size ('letterpaper' or 'a4paper').
    "papersize": "letterpaper",
    # The font size ('10pt', '11pt' or '12pt').
    "pointsize": "10pt",
    # Additional stuff for the LaTeX preamble.
    "preamble": r"""
% Gracefully handle unsupported image formats (like SVG)
\usepackage{graphicx}
\DeclareGraphicsExtensions{.pdf,.png,.jpg,.jpeg,.PDF,.PNG,.JPG,.JPEG}
% Support for Unicode characters (like Greek letters σ, Σ, etc.)
\usepackage{newunicodechar}
% Lowercase Greek letters
\newunicodechar{α}{\ensuremath{\alpha}}
\newunicodechar{β}{\ensuremath{\beta}}
\newunicodechar{γ}{\ensuremath{\gamma}}
\newunicodechar{δ}{\ensuremath{\delta}}
\newunicodechar{ε}{\ensuremath{\varepsilon}}
\newunicodechar{θ}{\ensuremath{\theta}}
\newunicodechar{κ}{\ensuremath{\kappa}}
\newunicodechar{λ}{\ensuremath{\lambda}}
\newunicodechar{μ}{\ensuremath{\mu}}
\newunicodechar{π}{\ensuremath{\pi}}
\newunicodechar{σ}{\ensuremath{\sigma}}
\newunicodechar{ω}{\ensuremath{\omega}}
% Uppercase Greek letters
\newunicodechar{Δ}{\ensuremath{\Delta}}
\newunicodechar{Θ}{\ensuremath{\Theta}}
\newunicodechar{Σ}{\ensuremath{\Sigma}}
% Mathematical symbols
\newunicodechar{→}{\ensuremath{\rightarrow}}
\newunicodechar{×}{\ensuremath{\times}}
\newunicodechar{²}{\ensuremath{^2}}
\newunicodechar{∞}{\ensuremath{\infty}}
""",
}
latex_documents = [
    ("index", "robometric-frame.tex", "FRAME Documentation", "Ameya Wagh", "manual"),
]

# Tell Sphinx to not include SVG images in LaTeX builds
latex_show_urls = "footnote"
latex_show_pagerefs = True

# -- Options for manual page output ------------------------------------------
man_pages = [("index", "robometric-frame", "FRAME Documentation", [author], 1)]

# -- Options for Texinfo output ----------------------------------------------
texinfo_documents = [
    (
        "index",
        "robometric-frame",
        "FRAME Documentation",
        author,
        "robometric-frame",
        "TorchMetrics-based evaluation metrics for robotics policies",
        "Miscellaneous",
    ),
]
