import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import scvelo as scv

from pyrovelocity.api import train_model
from pyrovelocity.data import load_larry
from pyrovelocity.plot import plot_gene_ranking
from pyrovelocity.plot import plot_mean_vector_field
from pyrovelocity.plot import us_rainbowplot
from pyrovelocity.plot import vector_field_uncertainty


"""Loads and plots data for all cell types and trains and saves model2 model.

Inputs:
  "data/larry.h5ad" via load_larry()

Outputs:
  data:
    "larry_invitro_adata_with_scvelo_dynamicalvelocity.h5ad"
    "fig3_larry_allcells_top2000_model2.h5ad"
    "fig3_allcells_data_model2.pkl"
  models:
    Fig3_allcells_model2/
    ├── attr.pkl
    ├── model_params.pt
    ├── param_store_test.pt
    └── var_names.csv
  figures:
    "fig3_all_test_volcano_sub_model2.pdf"
    "fig3_all_test_rainbow_sub_model2.pdf"
    "fig3_test_vecfield_sub_model2.pdf"
"""


###############
# load data
###############

adata = load_larry()

if not os.path.exists("larry_invitro_adata_with_scvelo_dynamicalvelocity.h5ad"):
    adata_input = adata.copy()
    scv.pp.filter_and_normalize(adata_input, min_shared_counts=30, n_top_genes=2000)
    scv.pp.moments(adata_input, n_pcs=30, n_neighbors=30)
    scv.tl.recover_dynamics(adata_input, n_jobs=30)
    scv.tl.velocity(adata_input, mode="dynamical")
    scv.tl.velocity_graph(adata_input, n_jobs=30)
    scv.tl.latent_time(adata_input)
    adata_input.write("larry_invitro_adata_with_scvelo_dynamicalvelocity.h5ad")
else:
    adata_input = scv.read("larry_invitro_adata_with_scvelo_dynamicalvelocity.h5ad")

adata_input.layers["raw_spliced"] = adata[:, adata_input.var_names].layers["spliced"]
adata_input.layers["raw_unspliced"] = adata[:, adata_input.var_names].layers[
    "unspliced"
]

adata_input.obs["u_lib_size_raw"] = adata_input.layers["unspliced"].toarray().sum(-1)
adata_input.obs["s_lib_size_raw"] = adata_input.layers["spliced"].toarray().sum(-1)


#############
# train model
#############

adata_model_pos_split = train_model(
    adata_input,
    max_epochs=1000,
    svi_train=True,
    log_every=100,
    patient_init=45,
    batch_size=4000,
    use_gpu=1,
    cell_state="state_info",
    include_prior=True,
    offset=True,
    library_size=True,
    patient_improve=1e-3,
    model_type="auto",
    guide_type="auto",
    train_size=1.0,
)

v_map_all, embeds_radian, fdri = vector_field_uncertainty(
    adata_input, adata_model_pos_split[1], basis="emb", denoised=False, n_jobs=1
)


##################
# generate figures
##################

fig, ax = plt.subplots()
volcano_data, _ = plot_gene_ranking(
    [adata_model_pos_split[1]], [adata_input], ax=ax, time_correlation_with="st"
)
fig.savefig(
    "fig3_all_test_volcano_sub_model2.pdf",
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
fig.savefig(
    "fig3_all_test_rainbow_sub_model2.pdf",
    facecolor=fig.get_facecolor(),
    bbox_inches="tight",
    edgecolor="none",
    dpi=300,
)

fig, ax = plt.subplots()
embed_mean = plot_mean_vector_field(
    adata_model_pos_split[1],
    adata_input,
    ax=ax,
    basis="emb",
    n_jobs=1
    # n_jobs > 1 potentially raises joblib.externals.loky.process_executor.BrokenProcessPool: A task has failed to un-serialize. Please ensure that the arguments of the function are all picklable.
    # solution: https://stackoverflow.com/questions/56154654/a-task-failed-to-un-serialize
    #           https://joblib.readthedocs.io/en/latest/parallel.html#thread-based-parallelism-vs-process-based-parallelism
)
fig.savefig(
    "fig3_test_vecfield_sub_model2.pdf",
    facecolor=fig.get_facecolor(),
    bbox_inches="tight",
    edgecolor="none",
    dpi=300,
)


##################
# save checkpoints
##################


adata_input.write("fig3_larry_allcells_top2000_model2.h5ad")
adata_model_pos_split[0].save("Fig3_allcells_model2", overwrite=True)

result_dict = {
    "adata_model_pos": adata_model_pos_split[1],
    "v_map_all": v_map_all,
    "embeds_radian": embeds_radian,
    "fdri": fdri,
    "embed_mean": embed_mean,
}

with open("fig3_allcells_data_model2.pkl", "wb") as f:
    pickle.dump(result_dict, f)
