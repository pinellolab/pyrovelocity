import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scvelo as scv
import seaborn as sns

from pyrovelocity.plot import plot_arrow_examples
from pyrovelocity.plot import plot_gene_ranking
from pyrovelocity.plot import plot_posterior_time
from pyrovelocity.plot import plot_vector_field_uncertain
from pyrovelocity.plot import rainbowplot

"""Loads preprocessed figure 2 data and produces figure 2.

Inputs:
  data:
    "fig2_pancreas_processed.h5ad"
    "fig2_pbmc_processed.h5ad"
  models:
    "fig2_pancreas_data.pkl"
    "fig2_pbmc_data.pkl"

Outputs:
  figures:
    "Fig2_pancreas_raw_gene_selection_model1.tif"
    "Fig2_pancreas_raw_gene_selection_model1.svg"
"""


##################
# load checkpoints
##################

with open("fig2_pbmc_data.pkl", "rb") as f:
    result_dict = pickle.load(f)

adata_model_pos_pbmc = result_dict["adata_model_pos"]
v_map_all_pbmc = result_dict["v_map_all"]
embeds_radian_pbmc = result_dict["embeds_radian"]
fdri_pbmc = result_dict["fdri"]
embed_mean_pbmc = result_dict["embed_mean"]

with open("fig2_pancreas_data.pkl", "rb") as f:
    result_dict = pickle.load(f)

adata_model_pos = result_dict["adata_model_pos"]
v_map_all = result_dict["v_map_all"]
embeds_radian = result_dict["embeds_radian"]
fdri = result_dict["fdri"]
embed_mean = result_dict["embed_mean"]

adata = scv.read("fig2_pancreas_processed.h5ad")
adata_pbmc = scv.read("fig2_pbmc_processed.h5ad")


#################
# generate figure
#################

fig = plt.figure(figsize=(7.07, 6.5))
dot_size = 3
font_size = 7
subfig = fig.subfigures(3, 1, wspace=0.0, hspace=0, height_ratios=[1.2, 1.2, 2.6])

ress = pd.DataFrame(
    {
        "cell_type": adata_pbmc.obs["celltype"].values,
        "X1": adata_pbmc.obsm["X_tsne"][:, 0],
        "X2": adata_pbmc.obsm["X_tsne"][:, 1],
    }
)
pbmcfig_A0 = subfig[0].subfigures(1, 2, wspace=0.0, hspace=0, width_ratios=[4, 2])
ax = pbmcfig_A0[0].subplots(1, 4)
sns.scatterplot(
    x="X1",
    y="X2",
    data=ress,
    alpha=0.9,
    s=dot_size,
    linewidth=0,
    edgecolor="none",
    hue="cell_type",
    ax=ax[0],
    legend="brief",
)
ax[0].axis("off")
ax[0].set_title("Cell types\n", fontsize=font_size)
ax[0].legend(
    bbox_to_anchor=[4.0, -0.01],
    ncol=3,
    prop={"size": font_size},
    fontsize=font_size,
    frameon=False,
)
ax[0].text(
    -0.1,
    1.15,
    "a",
    transform=ax[0].transAxes,
    fontsize=7,
    fontweight="bold",
    va="top",
    ha="right",
)

kwargs = dict(
    color="gray",
    density=0.8,
    add_margin=0.1,
    s=dot_size,
    show=False,
    alpha=0.2,
    min_mass=3.5,
    frameon=False,
)
scv.pl.velocity_embedding_stream(
    adata_pbmc, basis="tsne", fontsize=font_size, ax=ax[1], title="", **kwargs
)
ax[1].set_title("Scvelo\n", fontsize=7)

scv.pl.velocity_embedding_stream(
    adata_pbmc,
    fontsize=font_size,
    basis="tsne",
    title="",
    ax=ax[2],
    vkey="velocity_pyro",
    **kwargs
)
ax[2].set_title("Pyro-Velocity\n", fontsize=7)


plot_arrow_examples(
    adata_pbmc,
    np.transpose(v_map_all_pbmc, (1, 2, 0)),
    embeds_radian_pbmc,
    ax=ax[3],
    n_sample=30,
    fig=pbmcfig_A0,
    basis="tsne",
    scale=None,
    alpha=0.3,
    index=100,
    scale2=None,
    num_certain=0,
    num_total=4,
)
ax[3].set_title("Single cell\nvector field", fontsize=7)


plot_vector_field_uncertain(
    adata_pbmc,
    embed_mean_pbmc,
    embeds_radian_pbmc,
    fig=pbmcfig_A0[1],
    cbar=True,
    basis="tsne",
    scale=0.018,
    arrow_size=5,
)
pbmcfig_A0[0].subplots_adjust(
    hspace=0.2, wspace=0.1, left=0.01, right=0.99, top=0.99, bottom=0.45
)
pbmcfig_A0[1].subplots_adjust(
    hspace=0.2, wspace=0.1, left=0.01, right=0.99, top=0.99, bottom=0.45
)
pbmcfig_A0[0].text(-0.06, 0.58, "PBMC", size=7, rotation="vertical", va="center")

ress = pd.DataFrame(
    {
        "cell_type": adata.obs["clusters"].values,
        "X1": adata.obsm["X_umap"][:, 0],
        "X2": adata.obsm["X_umap"][:, 1],
    }
)
subfig_A0 = subfig[1].subfigures(1, 2, wspace=0.0, hspace=0, width_ratios=[4, 2])

ax = subfig_A0[0].subplots(1, 4)
sns.scatterplot(
    x="X1",
    y="X2",
    data=ress,
    alpha=0.9,
    s=dot_size,
    linewidth=0,
    edgecolor="none",
    hue="cell_type",
    ax=ax[0],
    legend="brief",
)
ax[0].axis("off")
ax[0].set_title("Cell types\n", fontsize=font_size)
ax[0].legend(
    bbox_to_anchor=[2.9, -0.01],
    ncol=4,
    prop={"size": font_size},
    fontsize=font_size,
    frameon=False,
)
ax[0].text(
    -0.1,
    1.18,
    "b",
    transform=ax[0].transAxes,
    fontsize=7,
    fontweight="bold",
    va="top",
    ha="right",
)
kwargs = dict(
    color="gray",
    density=0.8,
    add_margin=0.1,
    s=dot_size,
    show=False,
    alpha=0.2,
    min_mass=3.5,
    frameon=False,
)
scv.pl.velocity_embedding_stream(
    adata, fontsize=font_size, ax=ax[1], title="", **kwargs
)
ax[1].set_title("Scvelo\n", fontsize=7)

scv.pl.velocity_embedding_stream(
    adata,
    fontsize=font_size,
    basis="umap",
    title="",
    ax=ax[2],
    vkey="velocity_pyro",
    **kwargs
)
ax[2].set_title("Pyro-Velocity\n", fontsize=7)


plot_arrow_examples(
    adata,
    np.transpose(v_map_all, (1, 2, 0)),
    embeds_radian,
    ax=ax[3],
    n_sample=30,
    fig=fig,
    basis="umap",
    scale=0.005,
    alpha=0.2,
    index=5,
    scale2=0.03,
    num_certain=6,
    num_total=6,
)
ax[3].set_title("Single cell\nvector field", fontsize=7)


plot_vector_field_uncertain(
    adata,
    embed_mean,
    embeds_radian,
    fig=subfig_A0[1],
    cbar=True,
    basis="umap",
    scale=0.03,
    # scale=1.0,
    arrow_size=5,
)
subfig_A0[0].subplots_adjust(
    hspace=0.2, wspace=0.1, left=0.01, right=0.99, top=0.80, bottom=0.33
)
subfig_A0[1].subplots_adjust(
    hspace=0.2, wspace=0.1, left=0.01, right=0.99, top=0.80, bottom=0.33
)
subfig_A0[0].text(-0.06, 0.58, "Pancreas", size=7, rotation="vertical", va="center")

subfig_B = subfig[2].subfigures(1, 2, wspace=0.0, hspace=0, width_ratios=[1.6, 4])
ax = subfig_B[0].subplots(2, 1)
plot_posterior_time(adata_model_pos, adata, ax=ax[0], fig=subfig_B[0], addition=False)
subfig_B[0].subplots_adjust(
    hspace=0.3, wspace=0.1, left=0.01, right=0.8, top=0.92, bottom=0.17
)


volcano_data2, _ = plot_gene_ranking(
    [adata_model_pos], [adata], ax=ax[1], time_correlation_with="st", assemble=True
)
ax[0].text(
    -0.22,
    1.15,
    "c",
    transform=ax[0].transAxes,
    fontsize=7,
    fontweight="bold",
    va="top",
    ha="right",
)
ax[1].text(
    -0.1,
    1.15,
    "d",
    transform=ax[1].transAxes,
    fontsize=7,
    fontweight="bold",
    va="top",
    ha="right",
)


_ = rainbowplot(
    volcano_data2, adata, adata_model_pos, subfig_B[1], data=["st", "ut"], num_genes=4
)


fig.savefig(
    "Fig2_pancreas_raw_gene_selection_model1.tif",
    facecolor=fig.get_facecolor(),
    bbox_inches="tight",
    edgecolor="none",
    dpi=300,
)

fig.savefig(
    "Fig2_pancreas_raw_gene_selection_model1.svg",
    facecolor=fig.get_facecolor(),
    bbox_inches="tight",
    edgecolor="none",
    dpi=300,
)
