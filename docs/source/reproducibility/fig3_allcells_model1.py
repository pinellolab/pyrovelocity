import os

import cospar as cs
import matplotlib.pyplot as plt
import numpy as np
import scvelo as scv
import seaborn as sns
from dynamical_velocity2 import PyroVelocity
from dynamical_velocity2.data import load_data
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scvelo.datasets import simulation


cs.logging.print_version()
cs.settings.verbosity = 2
cs.settings.data_path = "LARRY_data"  # A relative path to save data. If not existed before, create a new one.
cs.settings.figure_path = "LARRY_figure"  # A relative path to save figures. If not existed before, create a new one.
cs.settings.set_figure_params(
    format="png", figsize=[4, 3.5], dpi=75, fontsize=14, pointsize=2
)
import matplotlib.pyplot as plt
import pandas as pd
import scvelo as scv
import seaborn as sns
from dynamical_velocity2 import PyroVelocity
from dynamical_velocity2.api import train_model
from dynamical_velocity2.data import load_data
from dynamical_velocity2.plot import denoised_umap
from dynamical_velocity2.plot import plot_arrow_examples
from dynamical_velocity2.plot import plot_gene_ranking
from dynamical_velocity2.plot import plot_mean_vector_field
from dynamical_velocity2.plot import plot_posterior_time
from dynamical_velocity2.plot import plot_vector_field_uncertain
from dynamical_velocity2.plot import project_grid_points
from dynamical_velocity2.plot import rainbowplot
from dynamical_velocity2.plot import us_rainbowplot
from dynamical_velocity2.plot import vector_field_uncertainty
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scvelo.datasets import simulation


adata_input = scv.read("larry_invitro_adata_with_scvelo_dynamicalvelocity.h5ad")
adata = scv.read("larry_invitro_adata_sub_raw.h5ad")

adata_cospar = scv.read(
    "LARRY_MultiTimeClone_Later_FullSpace0_t*2.0*4.0*6_adata_with_transition_map.h5ad"
)
adata_cytotrace = scv.read("larry_invitro_adata_sub_raw_withcytotrace.h5ad")
adata_vel = scv.read("larry_invitro_adata_with_scvelo_dynamicalvelocity.h5ad")

cs.pl.fate_potency(
    adata_cospar,
    used_Tmap="transition_map",
    map_backward=True,
    method="norm-sum",
    color_bar=True,
    fate_count=True,
)


adata_input.layers["raw_spliced"] = adata[:, adata_input.var_names].layers["spliced"]
adata_input.layers["raw_unspliced"] = adata[:, adata_input.var_names].layers[
    "unspliced"
]
adata_input.obs["u_lib_size_raw"] = adata_input.layers["unspliced"].toarray().sum(-1)
adata_input.obs["s_lib_size_raw"] = adata_input.layers["spliced"].toarray().sum(-1)

adata_model_pos_split = train_model(
    adata_input,
    max_epochs=1000,
    svi_train=True,
    lr=0.01,
    patient_init=45,
    batch_size=4000,
    use_gpu=1,
    log_every=100,
    patient_improve=1e-3,
    model_type="auto",
    guide_type="auto_t0_constraint",
    train_size=1.0,
    offset=False,
    library_size=True,
    include_prior=True,
)

pos = adata_model_pos_split[1]
scale = 1
# pos_ut = pos['ut'].mean(axis=0)
# pos_st = pos['st'].mean(axis=0)
# pos_u = pos['u'].mean(axis=0)
# pos_s = pos['s'].mean(axis=0)
# pos_v = pos['beta'].mean(0)[0]* pos_ut / scale - pos['gamma'].mean(0)[0] * pos_st
pos_time = pos["cell_time"].mean(0)

gold_standard = adata_cospar.obs["fate_potency"].values
(select,) = np.where(~np.isnan(gold_standard))
print(spearmanr(pos_time[select], gold_standard[select]))

# velocity_samples = pos['beta_k'] * pos['ut'] / scale - pos['gamma_k'] * pos['st']


def check_shared_time(adata_model_pos, adata):
    gold_standard = adata_cospar.obs["fate_potency"].values
    (select,) = np.where(~np.isnan(gold_standard))
    print(spearmanr(pos_time[select], gold_standard[select]))

    adata.obs["cell_time"] = adata_model_pos[1]["cell_time"].squeeze().mean(0)
    adata.obs["1-Cytotrace"] = 1 - adata_cytotrace.obs["cytotrace"]
    fig, ax = plt.subplots(1, 4)
    fig.set_size_inches(15, 3)
    scv.tl.latent_time(adata_vel)
    scv.pl.scatter(
        adata_vel,
        color="latent_time",
        show=False,
        ax=ax[0],
        title="scvelo %.2f"
        % spearmanr(1 - adata_cytotrace.obs.cytotrace, adata.obs.latent_time)[0],
        cmap="RdBu_r",
        basis="emb",
    )
    scv.pl.scatter(
        adata,
        color="cell_time",
        show=False,
        basis="emb",
        ax=ax[1],
        title="pyro %.2f"
        % spearmanr(1 - adata_cytotrace.obs.cytotrace, adata.obs.cell_time)[0],
    )
    scv.pl.scatter(adata, color="1-Cytotrace", show=False, ax=ax[2], basis="emb")
    scv.pl.scatter(
        adata_cospar, color="fate_potency", show=False, ax=ax[3], basis="emb"
    )
    print(spearmanr(adata.obs.cell_time, adata_vel.obs.latent_time))
    fig.savefig(
        "fig3_all_test_sub_model1.pdf",
        facecolor=fig.get_facecolor(),
        bbox_inches="tight",
        edgecolor="none",
        dpi=300,
    )


check_shared_time(adata_model_pos_split, adata_input)

fig, ax = plt.subplots()
volcano_data, _ = plot_gene_ranking(
    [adata_model_pos_split[1]], [adata_input], ax=ax, time_correlation_with="st"
)
fig.savefig(
    "fig3_all_test_volcano_sub.pdf",
    facecolor=fig.get_facecolor(),
    bbox_inches="tight",
    edgecolor="none",
    dpi=300,
)
fig = us_rainbowplot(
    volcano_data.sort_values("mean_mae", ascending=False)
    .head(50)
    .sort_values("time_correlation", ascending=False)
    .head(3)
    .index,
    adata_input,
    adata_model_pos_split[1],
    data=["st", "ut"],
    cell_state="state_info",
)
# fig = us_rainbowplot(['Grin2b', 'Map1b', 'Ppp3ca'],
fig.savefig(
    "fig3_all_test_rainbow_sub_model1.pdf",
    facecolor=fig.get_facecolor(),
    bbox_inches="tight",
    edgecolor="none",
    dpi=300,
)

fig, ax = plt.subplots()
embed_mean = plot_mean_vector_field(
    adata_model_pos_split[1], adata_input, ax=ax, basis="emb"
)
fig.savefig(
    "fig3_test_vecfield_sub_model1.pdf",
    facecolor=fig.get_facecolor(),
    bbox_inches="tight",
    edgecolor="none",
    dpi=300,
)

v_map_all, embeds_radian, fdri = vector_field_uncertainty(
    adata_input, adata_model_pos_split[1], basis="emb", denoised=False, n_jobs=1
)

fig, ax = plt.subplots()
embed_mean = plot_mean_vector_field(
    adata_model_pos_split[1], adata_input, ax=ax, basis="emb"
)

adata_input.write("fig3_larry_allcells_top2000_model1.h5ad")
adata_model_pos_split[0].save("Fig3_allcells_model1", overwrite=True)

result_dict = {
    "adata_model_pos": adata_model_pos_split[1],
    "v_map_all": v_map_all,
    "embeds_radian": embeds_radian,
    "fdri": fdri,
    "embed_mean": embed_mean,
}
import pickle


with open("fig3_allcells_data_model1.pkl", "wb") as f:
    pickle.dump(result_dict, f)
