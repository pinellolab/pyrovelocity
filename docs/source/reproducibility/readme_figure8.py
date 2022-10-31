import pickle

import cospar as cs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scvelo as scv
import seaborn as sns
from scipy.spatial import distance
from scipy.stats import spearmanr
from scvelo.plotting.velocity_embedding_grid import default_arrow

from pyrovelocity.plot import plot_posterior_time
from pyrovelocity.plot import plot_vector_field_uncertain
from pyrovelocity.plot import set_colorbar


cs.logging.print_version()
cs.settings.verbosity = 2
cs.settings.data_path = "LARRY_data"  # A relative path to save data.
cs.settings.figure_path = "LARRY_figure"  # A relative path to save figures.
cs.settings.set_figure_params(
    format="png", figsize=[4, 3.5], dpi=75, fontsize=14, pointsize=2
)

adata_input = scv.read("larry_invitro_adata_with_scvelo_dynamicalvelocity.h5ad")
adata = scv.read("larry_invitro_adata_sub_raw.h5ad")
adata_cospar = scv.read(
    "LARRY_MultiTimeClone_Later_FullSpace0_t*2.0*4.0*6_adata_with_transition_map.h5ad"
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


# with open("fig3_allcells_data.pkl", "rb") as pk:
with open("fig3_allcells_data_model2.pkl", "rb") as pk:
    result_dict = pickle.load(pk)
adata_model_pos_all = result_dict["adata_model_pos"]
v_map_all_all = result_dict["v_map_all"]
embeds_radian_all = result_dict["embeds_radian"]
fdri_all = result_dict["fdri"]
embed_mean_all = result_dict["embed_mean"]
# adata_input_all = scv.read("larry_allcells_top2000.h5ad")
adata_input_all = scv.read("fig3_larry_allcells_top2000_model2.h5ad")


# adata_input_all_clone = scv.read("/PHShome/qq06/pyrovelocity/figures/global_gold_standard2.h5ad")
adata_input_all_clone = scv.read("global_gold_standard2.h5ad")
adata_input_all_clone.obsm["clone_vector_emb"][
    np.isnan(adata_input_all_clone.obsm["clone_vector_emb"])
] = 0

# Calculate mean cosine similarity
from pyrovelocity.plot import align_trajectory_diff


cutoff = 10
density = 0.35
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
            adata_input_all.obsm["velocity_emb"][adata_input_all.obs.time_info != 6],
            embed_mean_all[adata_input_all.obs.time_info != 6],
        ],
        embed="emb",
        density=density,
    )
else:
    diff_all = align_trajectory_diff(
        [adata_input_all_clone, adata_input_all, adata_input_all],
        [
            adata_input_all_clone.obsm["clone_vector_emb"],
            adata_input_all.obsm["velocity_emb"],
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


hl, hw, hal = default_arrow(3)
quiver_kwargs = {"angles": "xy", "scale_units": "xy"}
quiver_kwargs.update({"width": 0.001, "headlength": hl / 2})
quiver_kwargs.update({"headwidth": hw / 2, "headaxislength": hal / 2, "alpha": 0.6})
quiver_kwargs.update({"linewidth": 1, "zorder": 3})
plt.figure()
plt.scatter(diff_all[:, 0], diff_all[:, 1], s=5, color="k")
plt.quiver(
    diff_all[:, 0], diff_all[:, 1], diff_all[:, 2], diff_all[:, 3], **quiver_kwargs
)
plt.savefig("test.pdf")

clean_cosine = np.array([scvelo_all_cos_mean, pyro_all_cos_mean])

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
fig, ax = plt.subplots(1, 5)
fig.set_size_inches(14, 2.5)
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
    ax=ax[0],
)
ax[0].set_title("Cell types", fontsize=7)
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
    ax=ax[1],
    title="Clonal progression",
    color="gray",
    arrow_color="black",
    fontsize=7,
)
plot_vector_field_uncertain(
    adata_input_all,
    embed_mean_all,
    embeds_radian_all,
    ax=ax[2:4],
    cbar=True,
    fig=fig,
    basis="emb",
    scale=scale,
    p_mass_min=1,
    density=density,
    arrow_size=arrow,
    cbar_pos=[0.46, 0.28, 0.1, 0.012],
    only_grid=False,
    autoscale=True,
)
ax[0].legend(
    bbox_to_anchor=[2.3, -0.03], ncol=4, prop={"size": 7}, fontsize=7, frameon=False
)
ax[0].axis("off")
adata_input_all.obs.loc[:, "vector_field_rayleigh_test"] = fdri_all
basis = "emb"
im = ax[4].scatter(
    adata_input_all.obsm[f"X_{basis}"][:, 0],
    adata_input_all.obsm[f"X_{basis}"][:, 1],
    s=3,
    alpha=0.9,
    c=adata_input_all.obs["vector_field_rayleigh_test"],
    cmap="inferno_r",
    linewidth=0,
)
set_colorbar(im, ax[4], labelsize=5, fig=fig, position="right")
select = adata_input_all.obs["vector_field_rayleigh_test"] > np.quantile(
    adata_input_all.obs["vector_field_rayleigh_test"], 0.95
)
sns.kdeplot(
    adata_input_all.obsm["X_emb"][:, 0][select],
    adata_input_all.obsm["X_emb"][:, 1][select],
    ax=ax[-1],
    levels=3,
    fill=False,
)
ax[0].axis("off")
ax[-1].axis("off")
ax[-1].set_title(
    "Vector field\nRayleigh test\nfdr<0.05: %s%%"
    % (round((fdri_all < 0.05).sum() / fdri_all.shape[0], 2) * 100),
    fontsize=7,
)
fig.subplots_adjust(
    hspace=0.3, wspace=0.13, left=0.01, right=0.99, top=0.95, bottom=0.3
)
fig.savefig(
    "readme_figure8.png",
    facecolor=fig.get_facecolor(),
    bbox_inches="tight",
    edgecolor="none",
    dpi=300,
)

fig, ax = plt.subplots(1, 3)
fig.set_size_inches(7, 2)
scv.pl.scatter(
    adata_cospar,
    basis="emb",
    fontsize=7,
    color="fate_potency",
    cmap="inferno_r",
    show=False,
    ax=ax[0],
    s=dotsize,
)
ax[0].set_title("Cospar clonal fate potency", fontsize=7)
gold = adata_cospar[adata_input_all.obs_names.str.replace("-0", ""), :].obs.fate_potency
select = ~np.isnan(gold)
adata_input_all.obs.cytotrace = adata_cytotrace.obs.cytotrace
plot_posterior_time(
    adata_model_pos_all,
    adata_input_all,
    ax=ax[1],
    basis="emb",
    fig=fig,
    addition=False,
    position="right",
)
ax[1].set_title(
    "Pyro-Velocity shared time\ncorrelation with Cospar: %.2f"
    % spearmanr(
        -gold[select], adata_model_pos_all["cell_time"].mean(0).flatten()[select]
    )[0],
    fontsize=7,
)
adata_input_all.obs["shared_time_uncertain"] = (
    adata_model_pos_all["cell_time"].std(0).flatten()
)
ax_cb = scv.pl.scatter(
    adata_input_all,
    c="shared_time_uncertain",
    ax=ax[2],
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
    ax=ax[2],
    levels=3,
    fill=False,
)
fig.savefig(
    "readme_figure9.png",
    facecolor=fig.get_facecolor(),
    bbox_inches="tight",
    edgecolor="none",
    dpi=300,
)
