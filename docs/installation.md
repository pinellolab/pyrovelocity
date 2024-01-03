```{highlight} shell

```

# Installation

We currently only support linux amd64 with access to at least one GPU having NVIDIA drivers installed.
This is related to issues [such as this](https://github.com/google/jax/issues/7097) among the indirect dependencies.
We provide a [docker container](https://github.com/pinellolab/pyrovelocity/pkgs/container/pyrovelocity) for the same architecture. We also provide an [IaC development environment](https://github.com/pinellolab/pyrovelocity/blob/main/reproducibility/environment/README.md).

If you are planning to attempt to install Pyro-Velocity on your own linux amd64 machine, it should be performed inside a virtual environment such as provided by [virtualenv](https://virtualenv.pypa.io/en/latest/) or [conda](https://github.com/conda-forge/miniforge#mambaforge).

## Stable release

To install pyrovelocity from [PyPI](https://pypi.org/project/pyrovelocity/) run

```console
$ pip install pyrovelocity
```

## Development source

The source for pyrovelocity can be downloaded from the [GitHub repository].

You can install the latest development version directly from github

```console
$ python -m pip install "pyrovelocity @ git+https://github.com/pinellolab/pyrovelocity.git@main"
```

[github repository]: https://github.com/pinellolab/pyrovelocity
[pip]: https://pip.pypa.io
