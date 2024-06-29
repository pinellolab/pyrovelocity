# Pyro-Velocity

<div align="center" style="margin-left: auto; margin-right: auto; max-width: 540px; overflow-x: auto;">
<a href="https://docs.pyrovelocity.net">
<img
    src="https://raw.githubusercontent.com/pinellolab/pyrovelocity/beta/docs/_static/logo.png"
    alt="Pyro-Velocity logo"
    style="width: 300px; max-width: 90%; height: auto; max-height: 350px;"
    role="img">
</a>

𝒫robabilistic modeling of RNA velocity ⬱

|         |                                                                                                                                                  |
| ------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| CI/CD   | [![CID][cid-badge]][cid-link] [![codecov][codecov-badge]][codecov-link] [![pre-commit.ci status][precommit-badge]][precommit-link]               |
| Docs    | [![Cloudflare Pages][cloudflare-badge]][cloudflare-link] [![Preprint][preprint-badge]][preprint-link]                                            |
| Package | [![PyPI - Version][pypi-badge]][pypi-link] [![Conda-forge badge][conda-forge-badge]][anaconda-link] [![Docker image][docker-badge]][docker-link] |
| Meta    | [![pyro-ppl-badge]][pyro-ppl-link] [![flyte-badge]][flyte-link] [![hydra-zen-badge]][hydra-zen-link] [![scvi-tools-badge]][scvi-tools-link]      |
|         | [![bear-badge]][bear-link] [![black-badge]][black-link] [![License][license-badge]][license-link] [![Tuple][tuple-badge]][tuple-link]            |

[bear-badge]: https://raw.githubusercontent.com/beartype/beartype-assets/main/badge/bear-ified.svg
[bear-link]: https://beartype.readthedocs.io
[cid-badge]: https://github.com/pinellolab/pyrovelocity/actions/workflows/cid.yaml/badge.svg?branch=master
[cid-link]: https://github.com/pinellolab/pyrovelocity/actions/workflows/cid.yaml
[precommit-badge]: https://results.pre-commit.ci/badge/github/pinellolab/pyrovelocity/main.svg
[precommit-link]: https://results.pre-commit.ci/latest/github/pinellolab/pyrovelocity/main
[flyte-badge]: https://storage.googleapis.com/pyrovelocity/badges/flyte-carrier.svg
[flyte-link]: https://docs.flyte.org
[hydra-zen-badge]: https://storage.googleapis.com/pyrovelocity/badges/Hydra-zen.svg
[hydra-zen-link]: https://mit-ll-responsible-ai.github.io/hydra-zen/
[cloudflare-badge]: https://img.shields.io/badge/Docs-pages-gray.svg?style=flat&logo=cloudflare&color=F26722
[cloudflare-link]: https://docs.pyrovelocity.net
[preprint-badge]: https://img.shields.io/badge/doi-10.1101/2022.09.12.507691v2-B31B1B
[preprint-link]: https://doi.org/10.1101/2022.09.12.507691
[pypi-badge]: https://img.shields.io/pypi/v/pyrovelocity.svg?logo=pypi&label=PyPI&color=F26722&logoColor=F26722
[pypi-link]: https://pypi.org/project/pyrovelocity/
[conda-forge-badge]: https://img.shields.io/conda/vn/conda-forge/pyrovelocity.svg?logo=conda-forge&label=conda-forge&color=F26722
[anaconda-link]: https://anaconda.org/conda-forge/pyrovelocity
[docker-badge]: https://img.shields.io/badge/docker-image-blue?logo=docker
[docker-link]: https://github.com/pinellolab/pyrovelocity/pkgs/container/pyrovelocity
[codecov-badge]: https://codecov.io/gh/pinellolab/pyrovelocity/branch/main/graph/badge.svg
[codecov-link]: https://codecov.io/gh/pinellolab/pyrovelocity
[black-badge]: https://img.shields.io/badge/code%20style-black-000000.svg
[black-link]: https://github.com/psf/black
[license-badge]: https://img.shields.io/badge/license-AGPL%203-gray.svg?color=F26722
[license-link]: https://spdx.org/licenses/
[pyro-ppl-badge]: https://storage.googleapis.com/pyrovelocity/badges/Pyro-PPL.svg
[pyro-ppl-link]: https://pyro.ai
[scvi-tools-badge]: https://storage.googleapis.com/pyrovelocity/badges/scvi-tools.svg
[scvi-tools-link]: https://scvi-tools.org
[tuple-badge]: https://img.shields.io/badge/Tuple%20❤️%20OSS-5A67D8?logo=tuple
[tuple-link]: https://tuple.app/github-badge

</div>

---

[Pyro-Velocity](https://docs.pyrovelocity.net) is a library for probabilistic inference in minimal models approximating gene expression dynamics from, possibly multimodal, single-cell sequencing data.
It provides posterior estimates of gene expression parameters, predictive estimates of gene expression states, and local estimates of cell state transition probabilities.
It can be used as a component in frameworks that attempt to retain the ability to propagate uncertainty in predicting: distributions over cell fates from subpopulations of cell states, response to cell state perturbations, or candidate genes or gene modules that correlate with determination of specific cell fates.

---

## Documentation 📒

Please see the [Documentation](https://docs.pyrovelocity.net).

## Changelog 🔀

Changes for each release are listed in the [Changelog](https://docs.pyrovelocity.net/about/changelog).

## Contributing ✨

Please review the [Contributing Guide](https://docs.pyrovelocity.net/about/contributing) for instructions on setting up a development environment and submitting pull requests.

## Community 🏘

If you would like to apply [Pyro-Velocity](https://docs.pyrovelocity.net) in your research, have an idea for a new feature, have a problem using the library, or just want to chat, please feel free to [start a discussion](https://github.com/pinellolab/pyrovelocity/discussions).

If you have a feature request or issue using Pyro-Velocity that may require making changes to the contents of this repository, please [file an issue](https://github.com/pinellolab/pyrovelocity/issues) containing

- a [GitHub permananent link](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/creating-a-permanent-link-to-a-code-snippet) to the location in the repository you think is causing a problem or will require changes, and
- provide a [minimal reproducible example](https://en.wikipedia.org/wiki/Minimal_reproducible_example) of the problem or proposed improvement.

We are always interested in discussions and issues that can help to improve the [Documentation](https://docs.pyrovelocity.net).

## License ⚖️

[AGPL](https://github.com/pinellolab/pyrovelocity/blob/main/LICENSE)
