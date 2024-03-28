"""
Pyro-Velocity

Pyro-Velocity is a python package for probabilistic modeling of RNA velocity.

docs: https://pyrovelocity.readthedocs.io
source: https://github.com/pinellolab/pyrovelocity
PyPI package: https://pypi.org/project/pyrovelocity
Conda package: https://anaconda.org/conda-forge/pyrovelocity
discussions: https://github.com/pinellolab/pyrovelocity/discussions
issues: https://github.com/pinellolab/pyrovelocity/issues
"""

from importlib import metadata

import pyrovelocity.analyze
import pyrovelocity.cytotrace
import pyrovelocity.datasets
import pyrovelocity.io
import pyrovelocity.logging
import pyrovelocity.metrics
import pyrovelocity.models
import pyrovelocity.plots
import pyrovelocity.postprocess
import pyrovelocity.summarize
import pyrovelocity.tasks
import pyrovelocity.train
import pyrovelocity.utils


# import pyrovelocity.interfaces
# import pyrovelocity.plot
# import pyrovelocity.tests
# import pyrovelocity.workflows


try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    __version__ = "unknown"

del metadata

__all__ = [
    "analyze",
    "cytotrace",
    "datasets",
    #     "interfaces",
    "io",
    "logging",
    "metrics",
    "models",
    #     "plot",
    "plots",
    "postprocess",
    "summarize",
    "tasks",
    #     "tests",
    "train",
    "utils",
    #     "workflows",
]
