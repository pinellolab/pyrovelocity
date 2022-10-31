import os

import cospar as cs
import matplotlib.pyplot as plt
import numpy as np
import scvelo as scv
from scipy.stats import spearmanr

from pyrovelocity.api import train_model
from pyrovelocity.plot import plot_mean_vector_field
from pyrovelocity.plot import vector_field_uncertainty


cs.logging.print_version()
cs.settings.verbosity = 2
cs.settings.data_path = "LARRY_data"  # A relative path to save data.
cs.settings.figure_path = "LARRY_figure"  # A relative path to save figures.
cs.settings.set_figure_params(
    format="png", figsize=[4, 3.5], dpi=75, fontsize=14, pointsize=2
)


adata = scv.read("mono_unipotent_cells.h5ad")
adata_input = adata[adata.obs.state_info != "Centroid", :].copy()

adata2 = scv.read("neu_unipotent_cells.h5ad")
adata_input2 = adata2[adata2.obs.state_info != "Centroid", :].copy()

adata = adata.concatenate(adata2)
adata_input = adata_input.concatenate(adata_input2)

if not os.path.exists("larry_uni_bifurcation_top2000.h5ad"):
    scv.pp.filter_and_normalize(adata_input, n_top_genes=2000, min_shared_counts=20)
    scv.pp.moments(adata_input)
    scv.tl.recover_dynamics(adata_input, n_jobs=10)
    scv.tl.velocity(adata_input, mode="dynamical")
    scv.tl.velocity_graph(adata_input)
    scv.tl.velocity_embedding(adata_input, basis="emb")
    scv.tl.latent_time(adata_input)
else:
    adata_input = scv.read("larry_uni_bifurcation_top2000.h5ad")

adata_input.layers["raw_spliced"] = adata[
    adata_input.obs_names, adata_input.var_names
].layers["spliced"]
adata_input.layers["raw_unspliced"] = adata[
    adata_input.obs_names, adata_input.var_names
].layers["unspliced"]

adata_input.obs["u_lib_size_raw"] = adata_input.layers["unspliced"].toarray().sum(-1)
adata_input.obs["s_lib_size_raw"] = adata_input.layers["spliced"].toarray().sum(-1)
# adata_model_pos_split = train_model(adata_input, max_epochs=2000, svi_train=False, log_every=100,

adata_model_pos_split = train_model(
    adata_input,
    max_epochs=4000,
    svi_train=False,
    log_every=100,
    patient_init=45,
    batch_size=-1,
    use_gpu=0,
    cell_state="state_info",
    include_prior=True,
    offset=True,
    library_size=True,
    model_type="auto",
    patient_improve=1e-4,
    guide_type="auto",
    train_size=1.0,
)


pos = adata_model_pos_split[1]
scale = 1
# pos_ut = pos['ut'].mean(axis=0)
# pos_st = pos['st'].mean(axis=0)
# pos_u = pos['u'].mean(axis=0)
# pos_s = pos['s'].mean(axis=0)
# pos_v = pos['beta'].mean(0)[0]* pos_ut / scale - pos['gamma'].mean(0)[0] * pos_st
# velocity_samples = pos['beta'] * pos['ut'] / scale - pos['gamma'] * pos['st']
pos_time = pos["cell_time"].mean(0)

adata_cospar = scv.read(
    "LARRY_MultiTimeClone_Later_FullSpace0_t*2.0*4.0*6_adata_with_transition_map.h5ad"
)

adata_cytotrace = scv.read("larry_invitro_adata_sub_raw_withcytotrace.h5ad")
cs.pl.fate_potency(
    adata_cospar,
    used_Tmap="transition_map",
    map_backward=True,
    method="norm-sum",
    color_bar=True,
    fate_count=True,
)

adata_cospar_sub = adata_cospar[
    adata_input.obs_names.str.replace(r"-\d-\d", ""), :
].copy()

adata_cytotrace_sub = adata_cytotrace[
    adata_input.obs_names.str.replace(r"-\d-\d", ""), :
].copy()


def check_shared_time(adata_model_pos, adata):
    # adata_cospar_sub = adata_cospar[(adata_cospar.obs['fate_bias_Neutrophil_Monocyte'] != 0.5), :].copy()
    gold_standard = adata_cospar_sub.obs["fate_potency"].values
    (select,) = np.where(~np.isnan(gold_standard))
    print("--------------")
    print(gold_standard.shape)
    print(select.shape)
    print(spearmanr(pos_time.squeeze()[select], gold_standard[select]))

    test_batch_size = 2000
    adata.obs["cell_time"] = adata_model_pos[1]["cell_time"].squeeze().mean(0)
    # adata.obs['lineage'] = adata_model_pos[1]['kinetics_lineage'].squeeze().mean(0)
    # adata.obs['lineage_prob'] = adata_model_pos[1]['kinetics_prob'].mean(0).argmax(-1).squeeze()

    adata.obs["1-Cytotrace"] = 1 - adata_cytotrace_sub.obs["cytotrace"]
    fig, ax = plt.subplots(1, 6)
    fig.set_size_inches(22, 3)
    scv.pl.scatter(
        adata,
        color="latent_time",
        show=False,
        ax=ax[0],
        title="scvelo %.2f"
        % spearmanr(1 - adata_cytotrace_sub.obs.cytotrace, adata.obs.latent_time)[0],
        cmap="RdBu_r",
        basis="emb",
    )
    # scv.pl.scatter(adata, color='cell_time', show=False, basis='emb',
    #                ax=ax[1], title='pyro %.2f' % spearmanr((1-adata_cytotrace_sub.obs.cytotrace, adata.obs.cell_time))[0])
    # scv.pl.scatter(adata, color='1-Cytotrace', show=False, ax=ax[2], basis='emb')
    scv.pl.scatter(
        adata_cospar_sub, color="fate_potency", show=False, ax=ax[3], basis="emb"
    )
    # scv.pl.scatter(adata, color='lineage', show=False, ax=ax[4], basis='emb')
    # scv.pl.scatter(adata, color='lineage_prob', show=False, ax=ax[5], basis='emb')
    fig.savefig(
        "fig3_bifurcation_test_sub.pdf",
        facecolor=fig.get_facecolor(),
        bbox_inches="tight",
        edgecolor="none",
        dpi=300,
    )


# velocity_samples = pos['beta'] * pos['ut'] / scale - pos['gamma'] * pos['st']

check_shared_time(adata_model_pos_split, adata_input)

gold_standard = adata_cospar_sub.obs["fate_potency"].values
(select,) = np.where(~np.isnan(gold_standard))
print(spearmanr(pos_time[select], gold_standard[select]))

v_map_all, embeds_radian, fdri = vector_field_uncertainty(
    adata_input, adata_model_pos_split[1], basis="emb", denoised=False, n_jobs=1
)

fig, ax = plt.subplots()
embed_mean = plot_mean_vector_field(
    adata_model_pos_split[1], adata_input, ax=ax, basis="emb"
)
fig.savefig("fig3_uni_bifurcation_vectorfield_model2.pdf")

