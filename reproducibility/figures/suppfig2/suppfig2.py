import os

import matplotlib.pyplot as plt
import pandas as pd
import scvelo as scv
import seaborn as sns
from pyrovelocity.cytotrace import compute_similarity2
from pyrovelocity.plot import (
    plot_gene_ranking,
    plot_posterior_time,
    plot_state_uncertainty,
    rainbowplot,
)


def param_set(top_genes=2000, n_neighbors=30):
    fs = f"pancreas_paramset_{top_genes}_{n_neighbors}.h5ad"
    if os.path.exists(fs):
        adata = scv.read(fs)
    else:
        adata = scv.datasets.pancreas()
        scv.pp.filter_and_normalize(
            adata, min_shared_counts=20, n_top_genes=top_genes
        )
        scv.pp.neighbors(adata, n_pcs=30, n_neighbors=n_neighbors)
        scv.pp.moments(adata, n_pcs=30, n_neighbors=n_neighbors)
        scv.tl.recover_dynamics(adata, n_jobs=20)
        scv.tl.velocity(adata, mode="dynamical")
        scv.tl.velocity_graph(adata)
        scv.tl.velocity_embedding(adata)
        adata.write(fs)
    return adata


import pickle

import numpy as np

with open("../fig2/model1/fig2_pancreas_data.pkl", "rb") as f:
    result_dict = pickle.load(f)
adata_model_pos = result_dict["adata_model_pos"]
v_map_all = result_dict["v_map_all"]
embeds_radian = result_dict["embeds_radian"]
fdri = result_dict["fdri"]
embed_mean = result_dict["embed_mean"]

with open("../fig2/model1/fig2_pbmc_data.pkl", "rb") as f:
    result_dict = pickle.load(f)
adata_model_pos_pbmc = result_dict["adata_model_pos"]
v_map_all_pbmc = result_dict["v_map_all"]
embeds_radian_pbmc = result_dict["embeds_radian"]
fdri_pbmc = result_dict["fdri"]
embed_mean_pbmc = result_dict["embed_mean"]
adata_pbmc = scv.read("../fig2/model1/pbmc_processed.h5ad")


s = 3
fig = plt.figure(figsize=(12, 8))
subfig = fig.subfigures(3, 1, wspace=0.0, hspace=0, height_ratios=[1, 1, 2])
ax = subfig[0].subplots(1, 8)
ax[0].text(
    -0.22,
    1.15,
    "a",
    transform=ax[0].transAxes,
    fontsize=7,
    fontweight="bold",
    va="top",
    ha="right",
)
adata = param_set()
ress = pd.DataFrame(
    {
        "cell_type": adata.obs["clusters"].values,
        "X1": adata.obsm["X_umap"][:, 0],
        "X2": adata.obsm["X_umap"][:, 1],
    }
)
sns.scatterplot(
    x="X1",
    y="X2",
    data=ress,
    alpha=0.9,
    s=s,
    linewidth=0,
    edgecolor="none",
    hue="cell_type",
    ax=ax[0],
    legend="brief",
)
ax[0].legend(
    bbox_to_anchor=[2.9, -0.01],
    ncol=4,
    prop={"size": 7},
    fontsize=7,
    frameon=False,
)
ax[0].axis("off")
scv.pl.velocity_embedding_grid(
    adata,
    color="clusters",
    ax=ax[0],
    show=False,
    legend_loc="none",
    scale=0.3,
    density=0.6,
    arrow_size=3,
    s=s,
    alpha=0,
)
ax[0].set_title("Default: 30 neighbors 2000 variable genes")
ax[0].set_title("")

adata = param_set(n_neighbors=3)
scv.tl.velocity_embedding(adata)
adata_sub = adata[adata.obs.clusters == "Epsilon"]
scv.pl.velocity_embedding_grid(
    adata_sub,
    color="clusters",
    ax=ax[1],
    show=False,
    legend_loc="none",
    scale=0.3,
    density=0.4,
    arrow_size=3,
    s=s,
)
ax[1].set_title("3 neighbors\n2000 variable genes", fontsize=7)

# ax = fig.add_subplot(gs[2])
adata = param_set(n_neighbors=300)
scv.tl.velocity_embedding(adata)
adata_sub = adata[adata.obs.clusters == "Epsilon"]
scv.pl.velocity_embedding_grid(
    adata_sub,
    color="clusters",
    ax=ax[2],
    show=False,
    legend_loc="none",
    scale=0.3,
    density=0.4,
    arrow_size=3,
    s=s,
)
ax[2].set_title("300 neighbors\n2000 variable genes", fontsize=7)

# ax = fig.add_subplot(gs[3])
adata = param_set(top_genes=50)
scv.tl.velocity_embedding(adata)
adata_sub = adata[adata.obs.clusters == "Epsilon"]
scv.pl.velocity_embedding_grid(
    adata_sub,
    color="clusters",
    ax=ax[3],
    show=False,
    legend_loc="none",
    scale=0.3,
    density=0.4,
    arrow_size=3,
    s=s,
)
ax[3].set_title("30 neighbors\n50 variable genes", fontsize=7)

# ax = fig.add_subplot(gs[4])
adata = param_set(top_genes=3000)
scv.tl.velocity_embedding(adata)
adata_sub = adata[adata.obs.clusters == "Epsilon"]
scv.pl.velocity_embedding_grid(
    adata_sub,
    color="clusters",
    ax=ax[4],
    show=False,
    legend_loc="none",
    scale=0.3,
    density=0.4,
    arrow_size=3,
    s=s,
)
ax[4].set_title("30 neighbors\n3000 variable genes", fontsize=7)

bin = 30
adata_pbmc.obs["shared_time_uncertain"] = (
    adata_model_pos_pbmc["cell_time"].std(0).flatten()
)
scv.pl.scatter(
    adata_pbmc,
    c="shared_time_uncertain",
    ax=ax[5],
    colorbar=True,
    basis="tsne",
    show=False,
    cmap="inferno",
    fontsize=7,
)
ax[5].text(
    -0.18,
    1.15,
    "b",
    transform=ax[5].transAxes,
    fontsize=7,
    fontweight="bold",
    va="top",
    ha="right",
)
select = adata_pbmc.obs["shared_time_uncertain"] > np.quantile(
    adata_pbmc.obs["shared_time_uncertain"], 0.9
)
sns.kdeplot(
    adata_pbmc.obsm["X_tsne"][:, 0][select],
    adata_pbmc.obsm["X_tsne"][:, 1][select],
    ax=ax[5],
    levels=3,
    fill=False,
)
adata_pbmc.obs["vector_field_rayleigh_test"] = fdri_pbmc
scv.pl.scatter(
    adata_pbmc,
    c="vector_field_rayleigh_test",
    basis="tsne",
    ax=ax[6],
    show=False,
    cmap="inferno_r",
    fontsize=7,
    colorbar=True,
)
select = adata_pbmc.obs["vector_field_rayleigh_test"] > np.quantile(
    adata_pbmc.obs["vector_field_rayleigh_test"], 0.9
)
sns.kdeplot(
    adata_pbmc.obsm["X_tsne"][:, 0][select],
    adata_pbmc.obsm["X_tsne"][:, 1][select],
    ax=ax[6],
    levels=3,
    fill=False,
)
ax[6].set_title(
    "Vector field rayleigh test <5%% FDR:%.2f%%"
    % (round((fdri_pbmc < 0.05).sum() / fdri_pbmc.shape[0], 3) * 100),
    fontsize=5,
)

plot_state_uncertainty(
    adata_model_pos_pbmc,
    adata_pbmc,
    basis="tsne",
    kde=True,
    data="raw",
    top_percentile=0.9,
    ax=ax[7],
)
ax[2].set_title("state uncertainty", fontsize=7)

from pyrovelocity.data import load_data

adata = load_data(top_n=2000, min_shared_counts=30)


scv.tl.latent_time(adata)
df_genes_cors = compute_similarity2(
    adata.layers["spliced"].toarray(),
    adata.obs.latent_time.values.reshape(-1, 1),
)
scvelo_top = pd.DataFrame(
    {
        "cor": df_genes_cors[0],
        "likelihood": adata.var["fit_likelihood"],
        "genes": adata.var_names,
    }
)

_, ax = plt.subplots()
volcano_data2, _ = plot_gene_ranking(
    [adata_model_pos], [adata], ax=ax, time_correlation_with="st", assemble=True
)

adata.obs["shared_time"] = adata_model_pos["cell_time"].mean(0).flatten()
ax = subfig[1].subplots(1, 7)
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
adata.obs["1-cytotrace"] = 1 - adata.obs["cytotrace"]
ax_cb = scv.pl.scatter(
    adata,
    c="1-cytotrace",
    ax=ax[0],
    show=False,
    cmap="inferno",
    fontsize=7,
    colorbar=True,
)
ax_cb = scv.pl.scatter(
    adata,
    c="latent_time",
    ax=ax[1],
    show=False,
    cmap="inferno",
    fontsize=7,
    colorbar=True,
)
ax_cb = scv.pl.scatter(
    adata,
    c="shared_time",
    ax=ax[2],
    show=False,
    cmap="inferno",
    fontsize=7,
    colorbar=True,
)

import matplotlib.pyplot as plt
from matplotlib_venn import venn2

set1 = set(
    volcano_data2.sort_values("mean_mae", ascending=False)
    .head(300)
    .sort_values("time_correlation", ascending=False)
    .head(50)
    .index
)
# set2 = set(volcano_data1.sort_values("mean_mae", ascending=False).head(300).sort_values("time_correlation", ascending=False).head(50).index)
set3 = set(
    scvelo_top.sort_values("likelihood", ascending=False)
    .head(300)
    .sort_values("cor", ascending=False)
    .head(50)
    .index
)

out = venn2([set1, set3], ["Denoised", "Scvelo\ntop 50"], ax=ax[3])
for text in out.set_labels:
    text.set_fontsize(7)
for text in out.subset_labels:
    text.set_fontsize(5)
ax[3].set_title("cell fate marker overlap", fontsize=7)

pos = adata_model_pos
bin = 30
adata.obs["shared_time_uncertain"] = (
    adata_model_pos["cell_time"].std(0).flatten()
)
ax_cb = scv.pl.scatter(
    adata,
    c="shared_time_uncertain",
    ax=ax[4],
    show=False,
    cmap="inferno",
    fontsize=7,
    colorbar=True,
)
select = adata.obs["shared_time_uncertain"] > np.quantile(
    adata.obs["shared_time_uncertain"], 0.9
)
sns.kdeplot(
    adata.obsm["X_umap"][:, 0][select],
    adata.obsm["X_umap"][:, 1][select],
    ax=ax[4],
    levels=3,
    fill=False,
)


adata.obs["vector_field_rayleigh_test"] = fdri
ax_cb = scv.pl.scatter(
    adata,
    c="vector_field_rayleigh_test",
    ax=ax[5],
    show=False,
    cmap="inferno_r",
    fontsize=7,
    colorbar=True,
)
select = adata.obs["vector_field_rayleigh_test"] > np.quantile(
    adata.obs["vector_field_rayleigh_test"], 0.9
)
sns.kdeplot(
    adata.obsm["X_umap"][:, 0][select],
    adata.obsm["X_umap"][:, 1][select],
    ax=ax[5],
    levels=3,
    fill=False,
)
ax[5].set_title(
    "Rayleigh test <5%% FDR:%s%%"
    % (round((fdri < 0.05).sum() / fdri.shape[0], 2) * 100),
    fontsize=5,
)

_ = plot_state_uncertainty(
    pos, adata, kde=True, data="raw", top_percentile=0.9, ax=ax[6]
)
ax[6].set_title("state uncertainty", fontsize=7)

subfig[0].subplots_adjust(
    hspace=0.3, wspace=0.18, left=0.0, right=0.92, top=0.92, bottom=0.25
)
subfig[1].subplots_adjust(
    hspace=0.3, wspace=0.15, left=0.0, right=0.92, top=0.92, bottom=0.25
)
supfigs = subfig[2].subfigures(
    1, 2, wspace=0.0, hspace=0, width_ratios=[1.6, 4]
)
ax = supfigs[0].subplots(2, 1)
plot_posterior_time(
    adata_model_pos, adata, ax=ax[0], fig=supfigs[0], addition=False
)
supfigs[0].subplots_adjust(
    hspace=0.3, wspace=0.1, left=0.01, right=0.75, top=0.92, bottom=0.15
)
volcano_data2, _ = plot_gene_ranking(
    [adata_model_pos],
    [adata],
    ax=ax[1],
    time_correlation_with="st",
    assemble=True,
    negative=True,
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
    volcano_data2,
    adata,
    adata_model_pos,
    supfigs[1],
    data=["st", "ut"],
    num_genes=4,
    negative=True,
)
fig.savefig(
    "SuppFigure2.pdf",
    facecolor=fig.get_facecolor(),
    bbox_inches="tight",
    edgecolor="none",
    dpi=300,
)
