# Pyro-Velocity

<div align="center">

<img src="https://raw.githubusercontent.com/pinellolab/pyrovelocity/main/docs/_static/logo.png" alt="Pyro-Velocity logo" width="300" role="img">

|         |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| ------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| CI/CD   | [![CID](https://github.com/pinellolab/pyrovelocity/actions/workflows/cid.yaml/badge.svg?branch=master)](https://github.com/pinellolab/pyrovelocity/actions/workflows/cid.yaml) [![CML](https://github.com/pinellolab/pyrovelocity/actions/workflows/cml.yml/badge.svg)](https://github.com/pinellolab/pyrovelocity/actions/workflows/cml.yml) [![pre-commit.ci status](https://results.pre-commit.ci/badge/github/pinellolab/pyrovelocity/main.svg)](https://results.pre-commit.ci/latest/github/pinellolab/pyrovelocity/master) |
| Docs    | [![GitHub Pages](https://img.shields.io/github/deployments/pinellolab/pyrovelocity/github-pages?logo=github&label=Docs)](https://pinellolab.github.io/pyrovelocity) [![Preprint](https://img.shields.io/badge/doi-10.1101/2022.09.12.507691v2-B31B1B)](https://doi.org/10.1101/2022.09.12.507691)                                                                                                                                                                                                                                |
| Package | [![PyPI - Version](https://img.shields.io/pypi/v/pyrovelocity.svg?logo=pypi&label=PyPI&logoColor=gold)](https://pypi.org/project/pyrovelocity/) [![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pyrovelocity.svg?logo=python&label=Python&logoColor=gold)](https://pypi.org/project/pyrovelocity/) [![Docker iamge](https://img.shields.io/badge/docker-image-blue?logo=docker)](https://github.com/pinellolab/pyrovelocity/pkgs/container/pyrovelocity)                                                        |
| Meta    | [![codecov](https://codecov.io/gh/pinellolab/pyrovelocity/branch/main/graph/badge.svg)](https://codecov.io/gh/pinellolab/pyrovelocity) [![code style - Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![License - MIT](https://img.shields.io/badge/license-AGPL%203-purple)](https://spdx.org/licenses/) [![Tuple](https://img.shields.io/badge/Tuple%20❤️%20OSS-5A67D8?logo=tuple)](https://tuple.app/github-badge)                                                        |

</div>

---

`Pyro-Velocity` is a Bayesian, generative, and multivariate RNA velocity
model to estimate _uncertainty_ in predictions of future cell states from
minimal models approximating splicing dynamics.
This approach models _raw sequencing counts_ with _synchronized cell time_ across
all expressed genes to provide quantifiable information on
cell fate choice and developmental trajectory dynamics.

## Features

- Probabilistic modeling of RNA velocity
- Direct modeling of raw spliced and unspliced read count
- Multiple uncertainty diagnostics analysis and visualizations
- Synchronized cell time estimation across genes
- Multivariate denoised gene expression and velocity prediction

<!-- ![Velocity workflow comparison](docs/source/readme_figure1.png) -->
<img src="https://raw.githubusercontent.com/pinellolab/pyrovelocity/master/docs/source/readme_figure1.png" alt="Velocity workflow comparison">

## Installation

### Quick start

See `make | grep codespace`, run `make codespace_create code`, or

[![codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/?hide_repo_select=true&ref=491-workflows&repo=533596575&skip_quickstart=true&machine=premiumLinux&devcontainer_path=.devcontainer%2Fdevcontainer.json)

This provides a zero install demonstration of the supported development environment. See `make -n setup_dev` to preview what is required to setup a local copy of this environment.

### Development

We currently support installation and usage in a linux 64-bit development environment with access to a GPU.
An [IaC](https://en.wikipedia.org/wiki/Infrastructure_as_code) setup that works with [GCP](https://registry.terraform.io/providers/hashicorp/google/latest/docs) is documented in [reproducibility/environment/README.md](reproducibility/environment/README.md).
Before proceeding to setup a minimal development environment, please fork this repository and clone a copy of your fork to your development machine. Unless otherwise mentioned, all commands assume your current working directory is the root of your local copy of your fork of this repository.

Please install mambaforge according to the instructions provided in [conda-forge/miniforge](https://github.com/conda-forge/miniforge#install).

You can then create a development environment with

```bash
mamba env create [--prefix /path_to_conda_environment] -f conda/environment-gpu.yml
```

This step takes about 10 minutes depending on network speed.

Make sure you are able to successfully activate the installed environment

```bash
conda activate pyrovelocity-gpu
which python
```

by checking that the output of which python following activation refers to the python binary in the correct conda environment.

You can then install a development copy with

```
pip install --no-deps -e .
```

If this is successful, you will be able to

```python
import pyrovelocity
```

from a python interpreter.

### User

Please see the [documentation](https://pyrovelocity.readthedocs.io/en/latest/installation.html).

## Quick start

After the installation, let\'s look at your dataset to see how
Pyro-Velocity can help understand cell dynamics.

Starting from raw sequencing FASTQ files, obtained for example with
SMART-seq, 10X genomics, inDrop or other similar single-cell assays, you
can preprocess the data to generate spliced and unspliced gene count
tables in h5ad file (or loom file using cellranger+velocyto or the
kallisto pipeline.

Starting from these count tables we show below a minimal step-by-step
workflow to illustrate the main features of Pyro-Velocity in a Jupyter
Notebook:

Step 0. Even though _pyrovelocity_ is a cluster-free method to evaluate uncertainty
of cell fate, it will dependend on 2 dimensional embedding results for evaluation of
uncertainty and generate visualization, we would suggest run your new datasets using
[scanpy tutorial](https://scanpy-tutorials.readthedocs.io/en/latest/pbmc3k.html).

Step 1. Load your data, load your data(e.g. _local_file.h5ad_) with
scvelo by using:

```python
import scvelo as scv
adata = scv.read("local_file.h5ad")
```

Step 2. Minimally preprocess your _adata_ object:

```python
adata.layers['raw_spliced']   = adata.layers['spliced']
adata.layers['raw_unspliced'] = adata.layers['unspliced']
adata.obs['u_lib_size_raw'] = adata.layers['raw_unspliced'].toarray().sum(-1)
adata.obs['s_lib_size_raw'] = adata.layers['raw_spliced'].toarray().sum(-1)
scv.pp.filter_and_normalize(adata, min_shared_counts=30, n_top_genes=2000)
scv.pp.moments(adata, n_pcs=30, n_neighbors=30)
```

Step 3. Train the Pyro-Velocity model:

```python
from pyrovelocity.api import train_model
from pyrovelocity.plot import vector_field_uncertainty
num_epochs = 1000 # large data
# num_epochs = 4000 # small data
# Model 1
adata_model_pos = train_model(adata,
                               max_epochs=num_epochs, svi_train=True, log_every=100,
                               patient_init=45,
                               batch_size=4000, use_gpu=0, cell_state='state_info',
                               include_prior=True,
                               offset=False,
                               library_size=True,
                               patient_improve=1e-3,
                               model_type='auto',
                               guide_type='auto_t0_constraint',
                               train_size=1.0)
# Or Model 2
adata_model_pos = train_model(adata,
                               max_epochs=num_epochs, svi_train=True, log_every=100,
                               patient_init=45,
                               batch_size=4000, use_gpu=0, cell_state='state_info',
                               include_prior=True,
                               offset=True,
                               library_size=True,
                               patient_improve=1e-3,
                               model_type='auto',
                               guide_type='auto',
                               train_size=1.0)

# adata_model_pos is a returned list in which 0th element is the trained model,
# the 1st element is the posterior samples of all random variables

trained_model = adata_model_pos[0]
posterior_samples = adata_model_pos[1]
v_map_all, embeds_radian, fdri = vector_field_uncertainty(
    adata,
    posterior_samples=posterior_samples,
    basis="umap"
)

save_res = True
if save_res:
    trained_model.save('saved_model', overwrite=True)
    result_dict = {"adata_model_pos": posterior_samples,
                   "v_map_all": v_map_all,
                   "embeds_radian": embeds_radian, "fdri": fdri} #, "embed_mean": embed_mean}
    import pickle
    with open("posterior_samples.pkl", "wb") as f:
         pickle.dump(result_dict, f)
```

Step 4: Generate Pyro-Velocity's vector field and shared time plots
with uncertainty estimation.

```python
from pyrovelocity.plot import plot_state_uncertainty
from pyrovelocity.plot import plot_posterior_time, plot_gene_ranking,\
      vector_field_uncertainty, plot_vector_field_uncertain,\
      plot_mean_vector_field, project_grid_points,rainbowplot,denoised_umap,\
      us_rainbowplot, plot_arrow_examples, set_colorbar

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


embedding = 'umap' # change to umap or tsne based on your embedding method

# This generates the posterior samples of all vector fields
# and statistical testing results from Rayleigh test
v_map_all, embeds_radian, fdri = vector_field_uncertainty(adata, posterior_samples,
                                                          basis=embedding, denoised=False, n_jobs=30)
fig, ax = plt.subplots()
# This returns the posterior mean of the vector field
embed_mean = plot_mean_vector_field(posterior_samples, adata, ax=ax, n_jobs=30, basis=embedding)
# This plot single-cell level vector field uncertainty
# and averaged cell vector field uncertainty on the grid points
# based on angular standard deviation
fig, ax = plt.subplots(1, 2)
fig.set_size_inches(11.5, 5)
plot_vector_field_uncertain(adata, embed_mean, embeds_radian,
                            ax=ax,
                            fig=fig, cbar=False, basis=embedding, scale=None)

# This generates shared time uncertainty plot with contour lines
fig, ax = plt.subplots(1, 3)
fig.set_size_inches(12, 2.8)
adata.obs['shared_time_uncertain'] = posterior_samples['cell_time'].std(0).flatten()
ax_cb = scv.pl.scatter(adata, c='shared_time_uncertain', ax=ax[0], show=False, cmap='inferno', fontsize=7, s=20, colorbar=True, basis=embedding)
select = adata.obs['shared_time_uncertain'] > np.quantile(adata.obs['shared_time_uncertain'], 0.9)
sns.kdeplot(adata.obsm[f'X_{embedding}'][:, 0][select],
            adata.obsm[f'X_{embedding}'][:, 1][select],
            ax=ax[0], levels=3, fill=False)

# This generates vector field uncertainty based on Rayleigh test.
adata.obs.loc[:, 'vector_field_rayleigh_test'] = fdri
im = ax[1].scatter(adata.obsm[f'X_{embedding}'][:, 0],
                   adata.obsm[f'X_{embedding}'][:, 1], s=3, alpha=0.9,
                   c=adata.obs['vector_field_rayleigh_test'], cmap='inferno_r',
                   linewidth=0)
set_colorbar(im, ax[1], labelsize=5, fig=fig, position='right')
select = adata.obs['vector_field_rayleigh_test'] > np.quantile(adata.obs['vector_field_rayleigh_test'], 0.95)
sns.kdeplot(adata.obsm[f'X_{embedding}'][:, 0][select],
            adata.obsm[f'X_{embedding}'][:, 1][select], ax=ax[1], levels=3, fill=False)
ax[1].axis('off')
ax[1].set_title("vector field\nrayleigh test\nfdr<0.05: %s%%" % (round((fdri < 0.05).sum()/fdri.shape[0], 2)*100), fontsize=7)
```

Step 5: Prioritize putative cell fate marker genes based on negative
mean absolute errors and pearson correlation between denoised spliced
expression and posterior mean shared time, and then visualize the top
one with rainbow plots

```python
fig = plt.figure(figsize=(7.07, 4.5))
subfig = fig.subfigures(1, 2, wspace=0.0, hspace=0, width_ratios=[1.6, 4])
ax = fig.subplots(1)
# This generates the selected cell fate markers and output in DataFrame
volcano_data, _ = plot_gene_ranking([posterior_samples], [adata], ax=ax,
                                     time_correlation_with='st', assemble=True)
# This generates the rainbow plots for the selected markers.
_ = rainbowplot(volcano_data, adata, posterior_samples,
                subfig[1], data=['st', 'ut'], num_genes=4)
```

## Illustrative examples of Pyro-Velocity analyses on different single-cell datasets

### Pyro-Velocity applied to a PBMC dataset

See the [data referred to here](https://scvelo.readthedocs.io/en/stable/perspectives/Perspectives/).

This is a scRNA-seq dataset of fully mature peripheral blood mononuclear
cells (PBMC) generated using the 10X genomics kit and containing 65,877
cells with 11 fully differentiated immune cell types. This dataset
doesn\'t contain stem and progenitor cells or other signature of and
undergoing dynamical differentiation, thus no consistent velocity flow
should be detected.

Below we show the main output generated by Pyro-Velocity Model 1
analysis. Pyro-Velocity failed to detect high-confidence trajectories in
the mature blood cell states, consistent with what is known about the
biology underlying these cells.

**Vector field with uncertainty**

<!-- ![PBMC vector field uncertainty](docs/source/readme_figure2.png) -->
<img src="https://raw.githubusercontent.com/pinellolab/pyrovelocity/main/docs/source/readme_figure2.png" alt="PBMC vector field uncertainty">

These 6 plots from left to right show: 1. cell types, 2. stream plot of
Pyro-velocity vector field based on the posterior mean of 30 posterior
samples, 3. single cell vector field examples showing all 30 posterior
samples as vectors for 3 arbitrarily selected cells; 4. single cell
vector field with uncertainty based on angular standard deviation across
30 posterior samples, 5. averaged vector field uncertainty from 4. 6.
Rayleigh test of posterior samples vector field, the title shows the
expected false discovery rate using a 5% threshold.

The full example can be reproduced using the
[PBMC](https://github.com/pinellolab/pyrovelocity/blob/master/docs/source/notebooks/pbmc.ipynb)
Jupyter notebook.

### Pyro-Velocity applied to a pancreas development dataset

See the [data referred to here](https://scvelo.readthedocs.io/en/stable/VelocityBasics/).

Here we apply Pyro-Velocity to a single cell RNA-seq dataset of mouse
pancreas in the E15.5 embryo developmental stage. This dataset was
generated using the 10X genomics kit and contains 3,696 cells with 8
cell types including progenitor cells, intermediate and terminal cell
states.

Below we show the main output generated by Pyro-Velocity Model 1
analysis. Pyro-Velocity was able to define well-known developmental cell
hierarchies identifying cell trajectories originating from ductal
progenitor cells and culminated in the production of mature Alpha, Beta,
Delta, and Epsilon cells.

**Vector field with uncertainty**

<!-- ![Pancreas vector field uncertainty](docs/source/readme_figure3.png) -->
<img src="https://raw.githubusercontent.com/pinellolab/pyrovelocity/main/docs/source/readme_figure3.png" alt="Pancreas vector field uncertainty">

These 6 plots from left to right are showing the same analyses presented
as in the example above.

**Shared time with uncertainty**

<!-- ![Pancreas shared time uncertainty](docs/source/readme_figure4.png) -->
<img src="https://raw.githubusercontent.com/pinellolab/pyrovelocity/master/docs/source/readme_figure4.png" alt="Pancreas vector field uncertainty">

The left figure shows the average of 30 posterior samples for the cell
shared time, the title of the figure shows the Spearman\'s correlation
with the Cytotrace score, an orthogonal state-of-the-art method used to
predict cell differentiation based on the number of expressed genes per
cell (Gulati et. al, Science 2020). The right figure shows the standard
deviation across posterior samples of shared time.

**Gene selection and visualization**

To uncover potential cell fate determinant markers genes of the mouse
pancreas, we first select the top 300 genes with the best velocity model
fit (we use negative mean absolute error ), then we rank the filtered
genes using Pearson\'s correlation between denoised spliced expression
and the posterior mean of the recovered shared time across cells.

<!-- ![Pancreas Volcano plot for gene selection](docs/source/readme_figure6.png) -->
<img src="https://raw.githubusercontent.com/pinellolab/pyrovelocity/main/docs/source/readme_figure6.png" alt="Pancreas Volcano plot for gene selection">

For the selected genes, it is possible to explore in depth their
dynamic, using phase portraits, rainbow plots, and UMAP rendering of
denoised splicing gene expression across cells.

<!-- ![Pancreas vector field uncertainty](docs/source/readme_figure7.png) -->
<img src="https://raw.githubusercontent.com/pinellolab/pyrovelocity/main/docs/source/readme_figure7.png" alt="Pancreas vector field uncertainty">

The full example can be reproduced using the
[Pancreas](https://github.com/pinellolab/pyrovelocity/blob/master/docs/source/notebooks/pancreas.ipynb)
jupyter notebook.

### Pyro-Velocity applied to the LARRY dataset

See the [data referred to here](https://figshare.com/articles/dataset/larry_invitro_adata_sub_raw_h5ad/20780344).

This last example, present the analysis of a recent scRNA-seq dataset
profiling mouse hematopoises at high resolution thanks to lineage
relationship information captured by the Lineage And RNA RecoverY
(LARRY) system. LARRY leverages unique lentiviral barcodes that enables
to clonally trace cell fates over time (Weinrab et al. Cell 20).

Below we show the main output generated by Pyro-Velocity analysis.

**Vector field with uncertainty**

<!-- ![LARRY vector field uncertainty](docs/source/readme_figure8.png) -->
<img src="https://raw.githubusercontent.com/pinellolab/pyrovelocity/main/docs/source/readme_figure8.png" alt="LARRY vector field uncertainty">

These 5 plots from left to right shows: 1) Cell types, 2) Clone
progression vector field by using centroid of cells belonging to the
same barcode for generating directed connection between consecutive
physical times, 3) single cell vector field with uncertainty based on
angular standard deviation across 30 posterior samples, 4. averaged
vector field uncertainty from 3. 5. Rayleigh test of posterior samples
vector field, the title shows the false discovery rate using threshold
5%.

**Shared time with uncertainty**

To quantitatively assess the quality of the the receovered shared time
we also considered the agreement of our method with Cospar, a
state-of-the-art method specifically designed for predicting fate
potency based on LARRY data.

<!-- ![Pancreas shared time uncertainty](docs/source/readme_figure9.png) -->
<img src="https://raw.githubusercontent.com/pinellolab/pyrovelocity/main/docs/source/readme_figure9.png" alt="Pancreas shared time uncertainty">

The leftmost figure shows the Cospar fate potency score, the middle
figure shows the average of 30 posterior samples from Pyro-Velocity
shared time per cell, the title of the figure shows the Spearman's
correlation between cell latent shared time and fate potency scores
derived from Cospar, the right figure shows the standard deviation
across posterior samples of shared time.

The full example can be reproduced using the
[LARRY](https://github.com/pinellolab/pyrovelocity/blob/main/docs/source/notebooks/larry.ipynb)
jupyter notebook.

# Troubleshooting

If you are having an issue using pyrovelocity, please feel free to [start a discussion](https://github.com/pinellolab/pyrovelocity/discussions)
or [file an issue](https://github.com/pinellolab/pyrovelocity/issues) containing a [MRE](https://en.wikipedia.org/wiki/Minimal_reproducible_example).

Also see [contributing](./docs/contributing.md).
