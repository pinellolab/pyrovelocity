import os
import pickle

import cospar as cs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scvelo as scv
import seaborn as sns
from pyrovelocity.data import load_larry
from pyrovelocity.plot import (
    align_trajectory_diff,
    get_clone_trajectory,
    plot_posterior_time,
    plot_vector_field_uncertain,
)
from scipy.spatial import distance
from scipy.stats import spearmanr
from scvelo.plotting.velocity_embedding_grid import default_arrow

"""Loads preprocessed figure 3 data and produces figure 3.

Inputs:
  data:
    "data/larry.h5ad" via load_larry()
    "larry_invitro_adata_with_scvelo_dynamicalvelocity.h5ad"
    "LARRY_data/LARRY_MultiTimeClone_Later_FullSpace0_t*2.0*4.0*6_adata_with_transition_map.h5ad"
    "larry_invitro_adata_sub_raw_withcytotrace.h5ad"
    "fig3_mono_processed_model1.h5ad"
    "fig3_neu_processed_model1.h5ad"
    "fig3_larry_uni_bifurcation_top2000_model2.h5ad"
    "fig3_larry_allcells_top2000_model2.h5ad"
  models:
    "fig3_mono_data_model1.pkl"
    "fig3_neu_data_model1.pkl"
    "fig3_uni_bifurcation_data_model2.pkl"
    "fig3_allcells_data_model2.pkl"

Outputs:
  data:
    "global_gold_standard2.h5ad"
  figures:
    "Figure3.pdf"
"""


##################
# load checkpoints
##################

cs.logging.print_version()
cs.settings.verbosity = 2
cs.settings.data_path = "LARRY_data"  # A relative path to save data.
cs.settings.figure_path = "LARRY_figure"  # A relative path to save figures.
cs.settings.set_figure_params(
    format="png", figsize=[4, 3.5], dpi=75, fontsize=14, pointsize=2
)

adata = load_larry()
adata_input_vel = scv.read(
    "larry_invitro_adata_with_scvelo_dynamicalvelocity.h5ad"
)
scv.tl.velocity_embedding(adata_input_vel, basis="emb")

adata_cospar = scv.read(
    "LARRY_data/LARRY_MultiTimeClone_Later_FullSpace0_t*2.0*4.0*6_adata_with_transition_map.h5ad"
)
adata_cytotrace = scv.read(
    "larry_invitro_adata_sub_raw_withcytotrace.h5ad"
)  # skip=False

cs.pl.fate_potency(
    adata_cospar,
    used_Tmap="transition_map",
    map_backward=True,
    method="norm-sum",
    color_bar=True,
    fate_count=True,
)

with open("fig3_mono_data_model1.pkl", "rb") as pk:
    result_dict = pickle.load(pk)
adata_model_pos_mono = result_dict["adata_model_pos"]
v_map_all_mono = result_dict["v_map_all"]
embeds_radian_mono = result_dict["embeds_radian"]
fdri_mono = result_dict["fdri"]
embed_mean_mono = result_dict["embed_mean"]
adata_input_mono = scv.read("fig3_mono_processed_model1.h5ad")

with open("fig3_neu_data_model1.pkl", "rb") as pk:
    result_dict = pickle.load(pk)
adata_model_pos_neu = result_dict["adata_model_pos"]
v_map_all_neu = result_dict["v_map_all"]
embeds_radian_neu = result_dict["embeds_radian"]
fdri_neu = result_dict["fdri"]
embed_mean_neu = result_dict["embed_mean"]
adata_input_neu = scv.read("fig3_neu_processed_model1.h5ad")

with open("fig3_uni_bifurcation_data_model2.pkl", "rb") as pk:
    result_dict = pickle.load(pk)

adata_model_pos = result_dict["adata_model_pos"]
v_map_all = result_dict["v_map_all"]
embeds_radian = result_dict["embeds_radian"]
fdri = result_dict["fdri"]
embed_mean = result_dict["embed_mean"]
adata_input = scv.read("fig3_larry_uni_bifurcation_top2000_model2.h5ad")

with open("fig3_allcells_data_model2.pkl", "rb") as pk:
    result_dict = pickle.load(pk)
adata_model_pos_all = result_dict["adata_model_pos"]
v_map_all_all = result_dict["v_map_all"]
embeds_radian_all = result_dict["embeds_radian"]
fdri_all = result_dict["fdri"]
embed_mean_all = result_dict["embed_mean"]
adata_input_all = scv.read("fig3_larry_allcells_top2000_model2.h5ad")


adata_input_neu_clone = get_clone_trajectory(adata_input_neu)
adata_input_mono_clone = get_clone_trajectory(adata_input_mono)
adata_input_uni_clone = adata_input_neu_clone.concatenate(
    adata_input_mono_clone
)

if os.path.exists("global_gold_standard2.h5ad"):
    adata_input_all_clone = scv.read("global_gold_standard2.h5ad")
else:
    adata_reduced_gene_for_clone_vec = adata[:, adata_input.var_names].copy()
    print(adata_reduced_gene_for_clone_vec.shape)
    adata_input_all_clone = get_clone_trajectory(
        adata_reduced_gene_for_clone_vec
    )
    adata_input_all_clone.write("global_gold_standard2.h5ad")

adata_input_all_clone.obsm["clone_vector_emb"][
    np.isnan(adata_input_all_clone.obsm["clone_vector_emb"])
] = 0
adata_input_mono_clone.obsm["clone_vector_emb"][
    np.isnan(adata_input_mono_clone.obsm["clone_vector_emb"])
] = 0
adata_input_neu_clone.obsm["clone_vector_emb"][
    np.isnan(adata_input_neu_clone.obsm["clone_vector_emb"])
] = 0
adata_input_uni_clone.obsm["clone_vector_emb"][
    np.isnan(adata_input_uni_clone.obsm["clone_vector_emb"])
] = 0


cutoff = 10
density = 0.35
diff_mono = align_trajectory_diff(
    [adata_input_mono_clone, adata_input_mono, adata_input_mono],
    [
        adata_input_mono_clone.obsm["clone_vector_emb"],
        adata_input_mono.obsm["velocity_emb"],
        embed_mean_mono,
    ],
    embed="emb",
    density=density,
)
scvelo_mono_cos = pd.DataFrame(diff_mono).apply(
    lambda x: 1 - distance.cosine(x[2:4], x[4:6]), axis=1
)
pyro_mono_cos = pd.DataFrame(diff_mono).apply(
    lambda x: 1 - distance.cosine(x[2:4], x[6:8]), axis=1
)
scvelo_mono_cos_mean = scvelo_mono_cos.mean()
pyro_mono_cos_mean = pyro_mono_cos.mean()

diff_neu = align_trajectory_diff(
    [adata_input_neu_clone, adata_input_neu, adata_input_neu],
    [
        adata_input_neu_clone.obsm["clone_vector_emb"],
        adata_input_neu.obsm["velocity_emb"],
        embed_mean_neu,
    ],
    embed="emb",
    density=density,
)
scvelo_neu_cos = pd.DataFrame(diff_neu).apply(
    lambda x: 1 - distance.cosine(x[2:4], x[4:6]), axis=1
)
pyro_neu_cos = pd.DataFrame(diff_neu).apply(
    lambda x: 1 - distance.cosine(x[2:4], x[6:8]), axis=1
)
scvelo_neu_cos_mean = scvelo_neu_cos.mean()
pyro_neu_cos_mean = pyro_neu_cos.mean()

diff_bifur = align_trajectory_diff(
    [adata_input_uni_clone, adata_input, adata_input],
    [
        adata_input_uni_clone.obsm["clone_vector_emb"],
        adata_input.obsm["velocity_emb"],
        embed_mean,
    ],
    embed="emb",
    density=density,
)
scvelo_bifur_cos = pd.DataFrame(diff_bifur).apply(
    lambda x: 1 - distance.cosine(x[2:4], x[4:6]), axis=1
)
pyro_bifur_cos = pd.DataFrame(diff_bifur).apply(
    lambda x: 1 - distance.cosine(x[2:4], x[6:8]), axis=1
)
scvelo_bifur_cos_mean = scvelo_bifur_cos.mean()
pyro_bifur_cos_mean = pyro_bifur_cos.mean()

exclude_day6 = False
if exclude_day6:
    diff_all = align_trajectory_diff(
        [
            adata_input_all_clone[adata_input_all_clone.obs.time_info != 6, :],
            adata_input_all[adata_input_all.obs.time_info != 6, :],
            adata_input_all[adata_input_all.obs.time_info != 6, :],
        ],
        [
            adata_input_all_clone.obsm["clone_vector_emb"][
                adata_input_all_clone.obs.time_info != 6, :
            ],
            adata_input_all.obsm["velocity_emb"][
                adata_input_all.obs.time_info != 6
            ],
            embed_mean_all[adata_input_all.obs.time_info != 6],
        ],
        embed="emb",
        density=density,
    )
else:
    diff_all = align_trajectory_diff(
        [adata_input_all_clone, adata_input, adata_input_all],
        [
            adata_input_all_clone.obsm["clone_vector_emb"],
            adata_input_vel.obsm["velocity_emb"],
            embed_mean_all,
        ],
        embed="emb",
        density=density,
    )
scvelo_all_cos = pd.DataFrame(diff_all).apply(
    lambda x: 1 - distance.cosine(x[2:4], x[4:6]), axis=1
)
pyro_all_cos = pd.DataFrame(diff_all).apply(
    lambda x: 1 - distance.cosine(x[2:4], x[6:8]), axis=1
)
scvelo_all_cos_mean = scvelo_all_cos.mean()
pyro_all_cos_mean = pyro_all_cos.mean()


##################
# generate figures
##################

hl, hw, hal = default_arrow(3)
quiver_kwargs = {"angles": "xy", "scale_units": "xy"}
quiver_kwargs.update({"width": 0.001, "headlength": hl / 2})
quiver_kwargs.update(
    {"headwidth": hw / 2, "headaxislength": hal / 2, "alpha": 0.6}
)
quiver_kwargs.update({"linewidth": 1, "zorder": 3})
plt.figure()
plt.scatter(diff_all[:, 0], diff_all[:, 1], s=5, color="k")
plt.quiver(
    diff_all[:, 0],
    diff_all[:, 1],
    diff_all[:, 2],
    diff_all[:, 3],
    **quiver_kwargs,
)
plt.savefig("test.pdf")

clean_cosine = np.array(
    [
        scvelo_mono_cos_mean,
        pyro_mono_cos_mean,
        scvelo_neu_cos_mean,
        pyro_neu_cos_mean,
        scvelo_bifur_cos_mean,
        pyro_bifur_cos_mean,
        scvelo_all_cos_mean,
        pyro_all_cos_mean,
    ]
).reshape(4, 2)

scv.pl.scatter(
    adata_input_all,
    basis="emb",
    fontsize=7,
    legend_loc="on data",
    legend_fontsize=7,
    color="state_info",
    show=False,
)

dotsize = 3
scale = 0.35
arrow = 3.5
fig, ax = plt.subplots(4, 8)
fig.set_size_inches(14, 8)
ax00 = ax[0]
ax0 = ax[1]
ax1 = ax[2]
ax2 = ax[3]
# scv.pl.scatter(adata_input_mono, basis='emb', fontsize=7,
#                legend_loc='on data', legend_fontsize=7,
#                color='state_info', cmap='RdBu_r', show=False, ax=ax00[0])
res = pd.DataFrame(
    {
        "X": adata_input_mono.obsm["X_emb"][:, 0],
        "Y": adata_input_mono.obsm["X_emb"][:, 1],
        "celltype": adata_input_mono.obs.state_info,
    }
)
sns.scatterplot(
    data=res,
    x="X",
    y="Y",
    hue="celltype",
    palette=dict(
        zip(
            adata_input_all.obs.state_info.cat.categories,
            adata_input_all.uns["state_info_colors"],
        )
    ),
    ax=ax00[0],
    s=dotsize,
    alpha=0.90,
    linewidth=0,
    legend=False,
)
ax00[0].set_title("Cell types", fontsize=7)
ax00[0].set_ylabel("Unipotent Monocyte lineage", fontsize=7)
scv.pl.velocity_embedding_grid(
    adata_input_mono_clone,
    # scale=None,
    scale=scale,
    autoscale=True,
    show=False,
    s=dotsize,
    density=density,
    arrow_size=arrow,
    linewidth=1,
    vkey="clone_vector",
    basis="emb",
    ax=ax00[1],
    title="Clonal progression",
    color="gray",
    arrow_color="black",
    fontsize=7,
)
# scvelo
scv.pl.velocity_embedding_grid(
    adata_input_mono,
    show=False,
    s=dotsize,
    density=density,
    # scale=None,
    scale=scale,
    autoscale=True,
    arrow_size=arrow,
    linewidth=1,
    basis="emb",
    ax=ax00[2],
    title="Scvelo",
    fontsize=7,
    color="gray",
    arrow_color="black",
)
ax00[2].set_title(
    "scVelo cosine similarity: %.2f" % clean_cosine[0, 0], fontsize=7
)
plot_vector_field_uncertain(
    adata_input_mono,
    embed_mean_mono,
    embeds_radian_mono,
    ax=ax00[3:5],
    cbar=False,
    fig=fig,
    basis="emb",
    # scale=None,
    scale=scale,
    arrow_size=arrow,
    p_mass_min=1,
    autoscale=True,
    cbar_pos=[0.46, 0.28, 0.1, 0.05],
    density=density,
    only_grid=False,
)
ax00[4].set_title(
    "Pyro-Velocity cosine similarity: %.2f" % clean_cosine[0, 1], fontsize=7
)
scv.pl.scatter(
    adata_cospar[adata_input_mono.obs_names.str.replace("-0", ""), :],
    basis="emb",
    fontsize=7,
    color="fate_potency",
    cmap="inferno_r",
    show=False,
    ax=ax00[5],
    s=dotsize,
)
ax00[5].set_title("Clonal fate potency", fontsize=7)
gold = adata_cospar[
    adata_input_mono.obs_names.str.replace("-0", ""), :
].obs.fate_potency
select = ~np.isnan(gold)
scv.pl.scatter(
    adata_input_mono,
    c="latent_time",
    basis="emb",
    s=dotsize,
    cmap="inferno",
    ax=ax00[6],
    show=False,
    fontsize=7,
)
ax00[6].set_title(
    "Scvelo latent time\ncorrelation: %.2f"
    % spearmanr(-gold[select], adata_input_mono.obs.latent_time.values[select])[
        0
    ],
    fontsize=7,
)
plot_posterior_time(
    adata_model_pos_mono,
    adata_input_mono,
    ax=ax00[7],
    basis="emb",
    fig=fig,
    addition=False,
    position="right",
)
ax00[7].set_title(
    "Pyro-Velocity shared time\ncorrelation: %.2f"
    % spearmanr(
        -gold[select],
        adata_model_pos_mono["cell_time"].mean(0).flatten()[select],
    )[0],
    fontsize=7,
)

res = pd.DataFrame(
    {
        "X": adata_input_neu.obsm["X_emb"][:, 0],
        "Y": adata_input_neu.obsm["X_emb"][:, 1],
        "celltype": adata_input_neu.obs.state_info,
    }
)
sns.scatterplot(
    data=res,
    x="X",
    y="Y",
    hue="celltype",
    palette=dict(
        zip(
            adata_input_all.obs.state_info.cat.categories,
            adata_input_all.uns["state_info_colors"],
        )
    ),
    s=dotsize,
    alpha=0.90,
    linewidth=0,
    legend=False,
    ax=ax0[0],
)
ax0[0].set_title("Cell types", fontsize=7)
ax0[0].set_ylabel("Unipotent Neutrophil", fontsize=7)
scv.pl.velocity_embedding_grid(
    adata_input_neu_clone,
    # scale=None,
    scale=scale,
    show=False,
    autoscale=True,
    s=dotsize,
    density=density,
    arrow_size=arrow,
    linewidth=1,
    vkey="clone_vector",
    basis="emb",
    ax=ax0[1],
    title="Clonal progression",
    color="gray",
    arrow_color="black",
    fontsize=7,
)

scv.pl.velocity_embedding_grid(
    adata_input_neu,
    show=False,
    s=dotsize,
    density=density,
    # scale=None,
    scale=scale,
    arrow_size=arrow,
    linewidth=1,
    basis="emb",
    ax=ax0[2],
    title="Scvelo",
    fontsize=7,
    autoscale=True,
    color="gray",
    arrow_color="black",
)
ax0[2].set_title(
    "scVelo cosine similarity: %.2f" % clean_cosine[1, 0], fontsize=7
)
plot_vector_field_uncertain(
    adata_input_neu,
    embed_mean_neu,
    embeds_radian_neu,
    ax=ax0[3:5],
    cbar=False,
    fig=fig,
    basis="emb",
    # scale=None,
    scale=scale,
    arrow_size=arrow,
    p_mass_min=1,
    density=density,
    cbar_pos=[0.46, 0.28, 0.1, 0.05],
    only_grid=False,
    autoscale=True,
)
ax0[4].set_title(
    "Pyro-Velocity cosine similarity: %.2f" % clean_cosine[1, 1], fontsize=7
)
scv.pl.scatter(
    adata_cospar[adata_input_neu.obs_names.str.replace("-0", ""), :],
    basis="emb",
    fontsize=7,
    color="fate_potency",
    cmap="inferno_r",
    show=False,
    ax=ax0[5],
    s=dotsize,
)
ax0[5].set_title("Clonal fate potency", fontsize=7)
gold = adata_cospar[
    adata_input_neu.obs_names.str.replace("-0", ""), :
].obs.fate_potency
select = ~np.isnan(gold)
scv.pl.scatter(
    adata_input_neu,
    c="latent_time",
    basis="emb",
    s=dotsize,
    cmap="inferno",
    ax=ax0[6],
    show=False,
    fontsize=7,
)
ax0[6].set_title(
    "Scvelo latent time\ncorrelation: %.2f"
    % spearmanr(-gold[select], adata_input_neu.obs.latent_time.values[select])[
        0
    ],
    fontsize=7,
)
plot_posterior_time(
    adata_model_pos_neu,
    adata_input_neu,
    ax=ax0[7],
    basis="emb",
    fig=fig,
    addition=False,
    position="right",
)
ax0[7].set_title(
    "Pyro-Velocity shared time\ncorrelation: %.2f"
    % spearmanr(
        -gold[select],
        adata_model_pos_neu["cell_time"].mean(0).flatten()[select],
    )[0],
    fontsize=7,
)

# scv.pl.scatter(adata_input, basis='emb', fontsize=7,
#                legend_loc='on data', legend_fontsize=7,
#                color='state_info', cmap='RdBu_r', show=False, ax=ax1[0])
res = pd.DataFrame(
    {
        "X": adata_input.obsm["X_emb"][:, 0],
        "Y": adata_input.obsm["X_emb"][:, 1],
        "celltype": adata_input.obs.state_info,
    }
)
sns.scatterplot(
    data=res,
    x="X",
    y="Y",
    hue="celltype",
    palette=dict(
        zip(
            adata_input_all.obs.state_info.cat.categories,
            adata_input_all.uns["state_info_colors"],
        )
    ),
    s=dotsize,
    alpha=0.9,
    linewidth=0,
    legend=False,
    ax=ax1[0],
)
ax1[0].set_title("Cell types", fontsize=7)

scv.pl.velocity_embedding_grid(
    adata_input_uni_clone,
    # scale=None,
    scale=scale,
    show=False,
    s=dotsize,
    density=density,
    arrow_size=arrow,
    linewidth=1,
    vkey="clone_vector",
    basis="emb",
    ax=ax1[1],
    title="Clonal progression",
    color="gray",
    arrow_color="black",
    fontsize=7,
    autoscale=True,
)

scv.pl.velocity_embedding_grid(
    adata_input,
    show=False,
    s=dotsize,
    density=density,
    # scale=None,
    scale=scale,
    arrow_size=arrow,
    linewidth=1,
    basis="emb",
    ax=ax1[2],
    title="Scvelo",
    fontsize=7,
    color="gray",
    arrow_color="black",
    autoscale=True,
)
ax1[2].set_title(
    "scVelo cosine similarity: %.2f" % clean_cosine[2, 0], fontsize=7
)
plot_vector_field_uncertain(
    adata_input,
    embed_mean,
    embeds_radian,
    ax=ax1[3:5],
    cbar=False,
    fig=fig,
    basis="emb",
    # scale=None,
    scale=scale,
    arrow_size=arrow,
    p_mass_min=1,
    density=density,
    cbar_pos=[0.46, 0.28, 0.1, 0.05],
    only_grid=False,
    autoscale=True,
)
ax1[4].set_title(
    "Pyro-Velocity cosine similarity: %.2f" % clean_cosine[2, 1], fontsize=7
)
gold = adata_cospar[
    adata_input.obs_names.str.replace(r"-\d-\d", ""), :
].obs.fate_potency
select = ~np.isnan(gold)
scv.pl.scatter(
    adata_cospar[adata_input.obs_names.str.replace(r"-\d-\d", ""), :],
    basis="emb",
    fontsize=7,
    color="fate_potency",
    cmap="inferno_r",
    show=False,
    ax=ax1[5],
    s=dotsize,
)
ax1[5].set_title("Clonal fate potency", fontsize=7)
scv.pl.scatter(
    adata_input,
    c="latent_time",
    basis="emb",
    s=dotsize,
    cmap="inferno",
    ax=ax1[6],
    show=False,
    fontsize=7,
)
ax1[6].set_title(
    "Scvelo latent time\ncorrelation: %.2f"
    % spearmanr(-gold[select], adata_input.obs.latent_time.values[select])[0],
    fontsize=7,
)
plot_posterior_time(
    adata_model_pos,
    adata_input,
    ax=ax1[7],
    basis="emb",
    fig=fig,
    addition=False,
    position="right",
)
ax1[7].set_title(
    "Pyro-Velocity shared time\ncorrelation: %.2f"
    % spearmanr(
        -gold[select], adata_model_pos["cell_time"].mean(0).flatten()[select]
    )[0],
    fontsize=7,
)

# scv.pl.scatter(adata_input_all, basis='emb', fontsize=7,
#                legend_loc='on data', legend_fontsize=7,
#                color='state_info', show=False)
res = pd.DataFrame(
    {
        "X": adata_input_all.obsm["X_emb"][:, 0],
        "Y": adata_input_all.obsm["X_emb"][:, 1],
        "celltype": adata_input_all.obs.state_info,
    }
)
sns.scatterplot(
    data=res,
    x="X",
    y="Y",
    hue="celltype",
    palette=dict(
        zip(
            adata_input_all.obs.state_info.cat.categories,
            adata_input_all.uns["state_info_colors"],
        )
    ),
    s=dotsize,
    alpha=0.9,
    linewidth=0,
    legend="brief",
    ax=ax2[0],
)
ax2[0].set_title("Cell types", fontsize=7)
scv.pl.velocity_embedding_grid(
    adata_input_all_clone,
    # scale=0.25,
    scale=scale,
    autoscale=True,
    show=False,
    s=dotsize,
    density=density,
    arrow_size=arrow,
    linewidth=1,
    vkey="clone_vector",
    basis="emb",
    ax=ax2[1],
    title="Clonal progression",
    color="gray",
    arrow_color="black",
    fontsize=7,
)

scv.pl.velocity_embedding_grid(
    # adata_input_all,
    adata_input_vel,
    show=False,
    s=dotsize,
    density=density,
    # scale=None,
    scale=scale,
    arrow_size=arrow,
    linewidth=1,
    basis="emb",
    ax=ax2[2],
    title="Scvelo",
    fontsize=7,
    color="gray",
    arrow_color="black",
    autoscale=True,
)
ax2[2].set_title(
    "scVelo cosine similarity: %.2f" % clean_cosine[3, 0], fontsize=7
)
plot_vector_field_uncertain(
    adata_input_all,
    embed_mean_all,
    embeds_radian_all,
    ax=ax2[3:5],
    cbar=True,
    fig=fig,
    # basis='emb', scale=0.0003, p_mass_min=0.1,
    # basis='emb', scale=0.001, p_mass_min=20,
    # basis='emb', scale=None, p_mass_min=1,
    basis="emb",
    scale=scale,
    p_mass_min=1,
    density=density,
    arrow_size=arrow,
    cbar_pos=[0.46, 0.28, 0.1, 0.012],
    only_grid=False,
    autoscale=True,
)
ax2[4].set_title(
    "Pyro-Velocity cosine similarity: %.2f" % clean_cosine[3, 1], fontsize=7
)
scv.pl.scatter(
    adata_cospar,
    basis="emb",
    fontsize=7,
    color="fate_potency",
    cmap="inferno_r",
    show=False,
    ax=ax2[5],
    s=dotsize,
)
ax2[5].set_title("Clonal fate potency", fontsize=7)
gold = adata_cospar[
    adata_input_all.obs_names.str.replace("-0", ""), :
].obs.fate_potency
select = ~np.isnan(gold)
scv.pl.scatter(
    adata_input_all,
    c="latent_time",
    basis="emb",
    s=dotsize,
    cmap="inferno",
    ax=ax2[6],
    show=False,
    fontsize=7,
)
ax2[6].set_title(
    "Scvelo latent time\ncorrelation: %.2f"
    % spearmanr(-gold[select], adata_input_all.obs.latent_time.values[select])[
        0
    ],
    fontsize=7,
)
adata_input_all.obs.cytotrace = adata_cytotrace.obs.cytotrace
plot_posterior_time(
    adata_model_pos_all,
    adata_input_all,
    ax=ax2[7],
    basis="emb",
    fig=fig,
    addition=False,
    position="right",
)
ax2[7].set_title(
    "Pyro-Velocity shared time\ncorrelation: %.2f"
    % spearmanr(
        -gold[select],
        adata_model_pos_all["cell_time"].mean(0).flatten()[select],
    )[0],
    fontsize=7,
)

for a, label, title in zip(
    [ax00[0], ax0[0], ax1[0], ax2[0]],
    ["a", "b", "c", "d"],
    [
        "Monocyte lineage",
        "Neutrophil lineage",
        "Bifurcation lineages",
        "All lineages",
    ],
):
    a.text(
        -0.1,
        1.15,
        label,
        transform=a.transAxes,
        fontsize=7,
        fontweight="bold",
        va="top",
        ha="right",
    )
    a.text(
        -0.1,
        0.42,
        title,
        transform=a.transAxes,
        fontsize=7,
        fontweight="bold",
        rotation="vertical",
        va="center",
    )
    a.axis("off")
ax2[0].legend(
    bbox_to_anchor=[2.3, -0.03],
    ncol=4,
    prop={"size": 7},
    fontsize=7,
    frameon=False,
)
fig.subplots_adjust(
    hspace=0.3, wspace=0.13, left=0.01, right=0.99, top=0.95, bottom=0.3
)

fig.savefig(
    "Figure3.pdf",
    facecolor=fig.get_facecolor(),
    bbox_inches="tight",
    edgecolor="none",
    dpi=300,
)
