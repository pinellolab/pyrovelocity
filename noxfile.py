"""Nox sessions."""
import os
import shutil
import sys
from pathlib import Path
from textwrap import dedent

import nox


try:
    from nox_poetry import Session
    from nox_poetry import session
except ImportError:
    message = f"""\
    Nox failed to import the 'nox-poetry' package.

    Please install it using the following command:

    {sys.executable} -m pip install nox-poetry"""
    raise SystemExit(dedent(message)) from None


package = "pyrovelocity"
python_versions = ["3.8"]
nox.needs_version = ">= 2021.6.6"
nox.options.sessions = (
    "docs-build",
)

@session(name="docs-build", python=python_versions[0])
def docs_build(session: Session) -> None:
    """Build the documentation."""
    args = session.posargs or ["docs", "docs/_build"]
    if not session.posargs and "FORCE_COLOR" in os.environ:
        args.insert(0, "--color")

    session.install(".")
    session.install("sphinx",
                    "nbsphinx",
                    "furo",
                    "myst-parser",
                    "sphinx_autodoc_typehints",
                    "sphinx_copybutton",
                    "sphinx-click")

    build_dir = Path("docs", "_build")
    if build_dir.exists():
        shutil.rmtree(build_dir)

    session.run("sphinx-build", *args)


@session(python=python_versions[0])
def docs(session: Session) -> None:
    """Build and serve the documentation with live reloading on file changes."""
    args = session.posargs or ["--open-browser", "docs", "docs/_build"]
    session.install(".")
    session.install("sphinx", "nbsphinx", "sphinx-autobuild", "sphinx-click", "furo", "myst-parser")

    build_dir = Path("docs", "_build")
    if build_dir.exists():
        shutil.rmtree(build_dir)

    session.run("sphinx-autobuild", *args)
