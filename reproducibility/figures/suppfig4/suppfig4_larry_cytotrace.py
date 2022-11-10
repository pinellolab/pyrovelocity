import os
import pickle

import cospar as cs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scvelo as scv
import seaborn as sns
from scipy.stats import spearmanr

from pyrovelocity.data import load_larry
from pyrovelocity.data import load_unipotent_larry
from pyrovelocity.plot import plot_gene_ranking
from pyrovelocity.plot import plot_posterior_time
from pyrovelocity.plot import plot_state_uncertainty
from pyrovelocity.plot import rainbowplot
from pyrovelocity.plot import set_colorbar


cs.logging.print_version()
cs.settings.verbosity = 2
cs.settings.data_path = "../fig3/LARRY_data"  # A relative path to save data.
cs.settings.figure_path = "../fig3/LARRY_figure"  # A relative path to save figures.
cs.settings.set_figure_params(
    format="png", figsize=[4, 3.5], dpi=75, fontsize=14, pointsize=2
)

adata_input = scv.read("../fig3/larry_invitro_adata_with_scvelo_dynamicalvelocity.h5ad")
adata = load_larry()
adata_cospar = scv.read(
    "../fig3/LARRY_data/LARRY_MultiTimeClone_Later_FullSpace0_t*2.0*4.0*6_adata_with_transition_map.h5ad"
)
adata_cytotrace = scv.read(
    "../fig3/larry_invitro_adata_sub_raw_withcytotrace.h5ad"
)  # skip_regress=False

adata_uni_mono = load_unipotent_larry()
adata_uni_mono = adata_uni_mono[adata_uni_mono.obs.state_info != "Centroid", :].copy()

adata_uni_neu = load_unipotent_larry("neu")
adata_uni_neu = adata_uni_neu[adata_uni_neu.obs.state_info != "Centroid", :].copy()
adata_uni_bifurcation = adata_uni_mono.concatenate(adata_uni_neu)

cs.pl.fate_potency(
    adata_cospar,
    used_Tmap="transition_map",
    map_backward=True,
    method="norm-sum",
    color_bar=True,
    fate_count=True,
)

with open("../fig3/fig3_allcells_data_model2.pkl", "rb") as pk:
    result_dict = pickle.load(pk)
adata_model_pos_all = result_dict["adata_model_pos"]
v_map_all_all = result_dict["v_map_all"]
embeds_radian_all = result_dict["embeds_radian"]
fdri_all = result_dict["fdri"]
embed_mean_all = result_dict["embed_mean"]
adata_input_all = scv.read("../fig3/fig3_larry_allcells_top2000_model2.h5ad")
adata_input_all.obs.cytotrace = adata_cytotrace.obs.cytotrace

gold = adata_cospar[adata_input_all.obs_names.str.replace("-0", ""), :].obs.fate_potency
gold_select = ~np.isnan(gold)

all_metrics = np.zeros((2, 2))
for i, gold_standard in enumerate([-gold[gold_select]]):
    for j, pred in enumerate(
        [
            adata_input_all.obs.latent_time.values[gold_select],
            adata_model_pos_all["cell_time"].mean(0).flatten()[gold_select],
        ]
    ):
        all_metrics[i, j] = spearmanr(gold_standard, pred)[0]
for _, gold_standard in enumerate([1 - adata_input_all.obs.cytotrace]):
    for j, pred in enumerate(
        [
            adata_input_all.obs.latent_time.values,
            adata_model_pos_all["cell_time"].mean(0).flatten(),
        ]
    ):
        all_metrics[1, j] = spearmanr(gold_standard, pred)[0]
all_metrics = pd.DataFrame(
    all_metrics,
    index=["Fate potency", "Cytotrace"],
    columns=["scVelo", "Pyro-Velocity"],
)

gold_uni_mono = adata_cospar[
    adata_uni_mono.obs_names.str.replace("-0", ""), :
].obs.fate_potency
gold_select_uni_mono = ~np.isnan(gold_uni_mono)
if not os.path.exists("larry_mono_top2000.h5ad"):
    scv.pp.filter_and_normalize(adata_uni_mono, n_top_genes=2000, min_shared_counts=20)
    scv.pp.moments(adata_uni_mono)
    scv.tl.recover_dynamics(adata_uni_mono, n_jobs=10)
    scv.tl.velocity(adata_uni_mono, mode="dynamical")
    scv.tl.velocity_graph(adata_uni_mono)
    scv.tl.velocity_embedding(adata_uni_mono, basis="emb")
    scv.tl.latent_time(adata_uni_mono)
else:
    adata_uni_mono = scv.read("larry_mono_top2000.h5ad")
with open("../fig3/fig3_mono_data_model1.pkl", "rb") as pk:
    result_dict = pickle.load(pk)
adata_model_pos_mono = result_dict["adata_model_pos"]
v_map_all_mono = result_dict["v_map_all"]
embeds_radian_mono = result_dict["embeds_radian"]
fdri_mono = result_dict["fdri"]
embed_mean_mono = result_dict["embed_mean"]
all_unimono_metrics = np.zeros((2, 2))
for i, gold_standard in enumerate([-gold_uni_mono[gold_select_uni_mono]]):
    for j, pred in enumerate(
        [
            adata_uni_mono.obs.latent_time.values[gold_select_uni_mono],
            adata_model_pos_mono["cell_time"].mean(0).flatten()[gold_select_uni_mono],
        ]
    ):
        all_unimono_metrics[i, j] = spearmanr(gold_standard, pred)[0]
for _, gold_standard in enumerate([1 - adata_uni_mono.obs.cytotrace]):
    for j, pred in enumerate(
        [
            adata_uni_mono.obs.latent_time.values,
            adata_model_pos_mono["cell_time"].mean(0).flatten(),
        ]
    ):
        all_unimono_metrics[1, j] = spearmanr(gold_standard, pred)[0]
all_unimono_metrics = pd.DataFrame(
    all_unimono_metrics,
    index=["Fate potency", "Cytotrace"],
    columns=["scVelo", "Pyro-Velocity"],
)

gold_uni_neu = adata_cospar[
    adata_uni_neu.obs_names.str.replace("-0", ""), :
].obs.fate_potency
gold_select_uni_neu = ~np.isnan(gold_uni_neu)

if not os.path.exists("larry_neu_top2000.h5ad"):
    scv.pp.filter_and_normalize(adata_uni_neu, n_top_genes=2000, min_shared_counts=20)
    scv.pp.moments(adata_uni_neu)
    scv.tl.recover_dynamics(adata_uni_neu, n_jobs=10)
    scv.tl.velocity(adata_uni_neu, mode="dynamical")
    scv.tl.velocity_graph(adata_uni_neu)
    scv.tl.velocity_embedding(adata_uni_neu, basis="emb")
    scv.tl.latent_time(adata_uni_neu)
else:
    adata_uni_neu = scv.read("larry_neu_top2000.h5ad")

with open("../fig3/fig3_neu_data_model1.pkl", "rb") as pk:
    result_dict = pickle.load(pk)
adata_model_pos_neu = result_dict["adata_model_pos"]
v_map_all_neu = result_dict["v_map_all"]
embeds_radian_neu = result_dict["embeds_radian"]
fdri_neu = result_dict["fdri"]
embed_mean_neu = result_dict["embed_mean"]

all_unineu_metrics = np.zeros((2, 2))
for i, gold_standard in enumerate([-gold_uni_neu[gold_select_uni_neu]]):
    for j, pred in enumerate(
        [
            adata_uni_neu.obs.latent_time.values[gold_select_uni_neu],
            adata_model_pos_neu["cell_time"].mean(0).flatten()[gold_select_uni_neu],
        ]
    ):
        all_unineu_metrics[i, j] = spearmanr(gold_standard, pred)[0]
for _, gold_standard in enumerate([1 - adata_uni_neu.obs.cytotrace]):
    for j, pred in enumerate(
        [
            adata_uni_neu.obs.latent_time.values,
            adata_model_pos_neu["cell_time"].mean(0).flatten(),
        ]
    ):
        all_unineu_metrics[1, j] = spearmanr(gold_standard, pred)[0]
all_unineu_metrics = pd.DataFrame(
    all_unineu_metrics,
    index=["Fate potency", "Cytotrace"],
    columns=["scVelo", "Pyro-Velocity"],
)


gold_uni_bifurcation = adata_cospar[
    adata_uni_bifurcation.obs_names.str.replace(r"-\d-\d", ""), :
].obs.fate_potency
gold_select_uni_bifurcation = ~np.isnan(gold_uni_bifurcation)

if not os.path.exists("larry_bifurcation_top2000.h5ad"):
    scv.pp.filter_and_normalize(
        adata_uni_bifurcation, n_top_genes=2000, min_shared_counts=20
    )
    scv.pp.moments(adata_uni_bifurcation)
    scv.tl.recover_dynamics(adata_uni_bifurcation, n_jobs=10)
    scv.tl.velocity(adata_uni_bifurcation, mode="dynamical")
    scv.tl.velocity_graph(adata_uni_bifurcation)
    scv.tl.velocity_embedding(adata_uni_bifurcation, basis="emb")
    scv.tl.latent_time(adata_uni_bifurcation)
else:
    adata_uni_bifurcation = scv.read("larry_bifurcation_top2000.h5ad")

with open("../fig3/fig3_uni_bifurcation_data_model2.pkl", "rb") as pk:
    result_dict = pickle.load(pk)
adata_model_pos_bifurcation = result_dict["adata_model_pos"]
v_map_all_bifurcation = result_dict["v_map_all"]
embeds_radian_bifurcation = result_dict["embeds_radian"]
fdri_bifurcation = result_dict["fdri"]
embed_mean_bifurcation = result_dict["embed_mean"]

all_unibifurcation_metrics = np.zeros((2, 2))
for i, gold_standard in enumerate([-gold_uni_bifurcation[gold_select_uni_bifurcation]]):
    for j, pred in enumerate(
        [
            adata_uni_bifurcation.obs.latent_time.values[gold_select_uni_bifurcation],
            adata_model_pos_bifurcation["cell_time"]
            .mean(0)
            .flatten()[gold_select_uni_bifurcation],
        ]
    ):
        all_unibifurcation_metrics[i, j] = spearmanr(gold_standard, pred)[0]
for _, gold_standard in enumerate([1 - adata_uni_bifurcation.obs.cytotrace]):
    for j, pred in enumerate(
        [
            adata_uni_bifurcation.obs.latent_time.values,
            adata_model_pos_bifurcation["cell_time"].mean(0).flatten(),
        ]
    ):
        all_unibifurcation_metrics[1, j] = spearmanr(gold_standard, pred)[0]
all_unibifurcation_metrics = pd.DataFrame(
    all_unibifurcation_metrics,
    index=["Fate potency", "Cytotrace"],
    columns=["scVelo", "Pyro-Velocity"],
)


dotsize = 3
scale = 0.35
arrow = 3.5

fig = plt.figure(figsize=(9.6, 10.5))
subfig = fig.subfigures(2, 1, wspace=0.0, hspace=0, height_ratios=[1, 1.3])
ax = subfig[0].subplots(2, 4)
adata_cytotrace.obs.loc[:, "1-Cytotrace"] = 1 - adata_cytotrace.obs.cytotrace
scv.pl.scatter(
    adata_cytotrace,
    basis="emb",
    fontsize=7,
    color="1-Cytotrace",
    cmap="inferno_r",
    show=False,
    ax=ax[0][0],
    s=dotsize,
)
ax[0][0].set_title(
    "Cytotrace\ncorrelation with fate potency: %.2f"
    % spearmanr(1 - adata_input_all.obs.cytotrace[gold_select], -gold[gold_select])[0],
    fontsize=7,
)
ax[0][0].text(
    -0.22,
    1.15,
    "a",
    transform=ax[0][0].transAxes,
    fontsize=7,
    fontweight="bold",
    va="top",
    ha="right",
)
g = sns.heatmap(
    all_metrics,
    annot=True,
    fmt=".3f",
    ax=ax[1][0],
    cbar=False,
    annot_kws={"fontsize": 7},
)
ax[1][0].set_xticklabels(
    ax[1][0].get_xmajorticklabels(), fontsize=7, rotation=0, ha="right"
)
ax[1][0].set_yticklabels(ax[1][0].get_ymajorticklabels(), fontsize=7)
ax[1][0].set_title("Multi-fate all cells", fontsize=7)
g = sns.heatmap(
    all_unibifurcation_metrics,
    annot=True,
    fmt=".3f",
    ax=ax[1][1],
    cbar=False,
    annot_kws={"fontsize": 7},
)
ax[1][1].set_xticklabels(
    ax[1][1].get_xmajorticklabels(), fontsize=7, rotation=0, ha="right"
)
ax[1][1].set_yticklabels(ax[1][1].get_ymajorticklabels(), fontsize=7)
ax[1][1].set_title("bi-fate cells", fontsize=7)
g = sns.heatmap(
    all_unimono_metrics,
    annot=True,
    fmt=".3f",
    ax=ax[1][2],
    cbar=False,
    annot_kws={"fontsize": 7},
)
ax[1][2].set_xticklabels(
    ax[1][2].get_xmajorticklabels(), fontsize=7, rotation=0, ha="right"
)
ax[1][2].set_yticklabels(ax[1][2].get_ymajorticklabels(), fontsize=7)
ax[1][2].set_title("uni-fate monocyte cells", fontsize=7)
g = sns.heatmap(
    all_unineu_metrics,
    annot=True,
    fmt=".3f",
    ax=ax[1][3],
    cbar=False,
    annot_kws={"fontsize": 7},
)
ax[1][3].set_xticklabels(
    ax[1][3].get_xmajorticklabels(), fontsize=7, rotation=0, ha="right"
)
ax[1][3].set_yticklabels(ax[1][3].get_ymajorticklabels(), fontsize=7)
ax[1][3].set_title("uni-fate neutrophil cells", fontsize=7)
ax[1][0].text(
    -0.22,
    1.15,
    "c",
    transform=ax[1][0].transAxes,
    fontsize=7,
    fontweight="bold",
    va="top",
    ha="right",
)


adata_input_all.obs["shared_time_uncertain"] = (
    adata_model_pos_all["cell_time"].std(0).flatten()
)
ax_cb = scv.pl.scatter(
    adata_input_all,
    c="shared_time_uncertain",
    ax=ax[0][1],
    show=False,
    cmap="inferno",
    fontsize=7,
    s=20,
    colorbar=True,
    basis="emb",
)
select = adata_input_all.obs["shared_time_uncertain"] > np.quantile(
    adata_input_all.obs["shared_time_uncertain"], 0.9
)
sns.kdeplot(
    adata_input_all.obsm["X_emb"][:, 0][select],
    adata_input_all.obsm["X_emb"][:, 1][select],
    ax=ax[0][1],
    levels=3,
    fill=False,
)
adata_input_all.obs.loc[:, "vector_field_rayleigh_test"] = fdri_all
basis = "emb"
im = ax[0][2].scatter(
    adata.obsm[f"X_{basis}"][:, 0],
    adata.obsm[f"X_{basis}"][:, 1],
    s=3,
    alpha=0.9,
    c=adata_input_all.obs["vector_field_rayleigh_test"],
    cmap="inferno_r",
    linewidth=0,
)
ax[0][1].text(
    -0.1,
    1.15,
    "b",
    transform=ax[0][1].transAxes,
    fontsize=7,
    fontweight="bold",
    va="top",
    ha="right",
)
set_colorbar(im, ax[0][2], labelsize=5, fig=fig, position="right")
select = adata_input_all.obs["vector_field_rayleigh_test"] > np.quantile(
    adata_input_all.obs["vector_field_rayleigh_test"], 0.95
)
sns.kdeplot(
    adata_input_all.obsm["X_emb"][:, 0][select],
    adata_input_all.obsm["X_emb"][:, 1][select],
    ax=ax[0][2],
    levels=3,
    fill=False,
)
ax[0][2].axis("off")
ax[0][2].set_title(
    "vector field\nrayleigh test\nfdr<0.05: %s%%"
    % (round((fdri_all < 0.05).sum() / fdri_all.shape[0], 2) * 100),
    fontsize=7,
)
_ = plot_state_uncertainty(
    adata_model_pos_all,
    adata_input_all,
    kde=True,
    data="raw",
    top_percentile=0.9,
    ax=ax[0][3],
    basis="emb",
)
# _ = plot_state_uncertainty(adata_model_pos_all, adata_input_all, kde=True, data='denoised', top_percentile=0.95, ax=ax[4], basis='emb')
ax[0][-1].set_title("state uncertainty", fontsize=7)
subfig[0].subplots_adjust(
    hspace=0.2, wspace=0.48, left=0.01, right=0.92, top=0.93, bottom=0.1
)
subfig_B = subfig[1].subfigures(1, 2, wspace=0.0, hspace=0, width_ratios=[1.8, 4])
ax = subfig_B[0].subplots(2, 1)
plot_posterior_time(
    adata_model_pos_all,
    adata_input_all,
    ax=ax[0],
    fig=subfig_B[0],
    addition=False,
    basis="emb",
)
volcano_data2, _ = plot_gene_ranking(
    [adata_model_pos_all],
    [adata_input_all],
    ax=ax[1],
    time_correlation_with="st",
    assemble=True,
    adjust_text=True,
)
ax[0].text(
    -0.22,
    1.15,
    "d",
    transform=ax[0].transAxes,
    fontsize=7,
    fontweight="bold",
    va="top",
    ha="right",
)
ax[1].text(
    -0.1,
    1.15,
    "e",
    transform=ax[1].transAxes,
    fontsize=7,
    fontweight="bold",
    va="top",
    ha="right",
)
subfig_B[0].subplots_adjust(
    hspace=0.3, wspace=0.1, left=0.01, right=0.8, top=0.92, bottom=0.15
)
_ = rainbowplot(
    volcano_data2,
    adata_input_all,
    adata_model_pos_all,
    subfig_B[1],
    data=["st", "ut"],
    num_genes=4,
    add_line=True,
    basis="emb",
    cell_state="state_info",
    scvelo_colors=True,
)
fig.savefig(
    "SuppFigure4.pdf",
    facecolor=fig.get_facecolor(),
    bbox_inches="tight",
    edgecolor="none",
    dpi=300,
)
