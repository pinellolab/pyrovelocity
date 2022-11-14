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
    "fig3_larry_allcells_top2000_model1.h5ad"
    "fig3_allcells_data_model1.pkl"
  models:
    Fig3_allcells_model1/
    ├── attr.pkl
    ├── model_params.pt
    ├── param_store_test.pt
    └── var_names.csv
  figures:
    "fig3_all_test_volcano_sub_model1.pdf"
    "fig3_all_test_rainbow_sub_model1.pdf"
    "fig3_test_vecfield_sub_model1.pdf"
"""


###############
# load data
###############

adata = scv.read("../../fig3/data/larry.h5ad")
adata_input = scv.read(
    "../../fig3/larry_invitro_adata_with_scvelo_dynamicalvelocity.h5ad"
)

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
    patient_improve=1e-3,
    model_type="auto",
    guide_type="auto_t0_constraint",
    train_size=1.0,
    offset=False,
    library_size=True,
    include_prior=True,
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
    "fig3_all_test_volcano_sub_model1.pdf",
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
    "fig3_all_test_rainbow_sub_model1.pdf",
    facecolor=fig.get_facecolor(),
    bbox_inches="tight",
    edgecolor="none",
    dpi=300,
)

fig, ax = plt.subplots()
embed_mean = plot_mean_vector_field(
    adata_model_pos_split[1], adata_input, ax=ax, basis="emb", n_jobs=1
)
fig.savefig(
    "fig3_test_vecfield_sub_model1.pdf",
    facecolor=fig.get_facecolor(),
    bbox_inches="tight",
    edgecolor="none",
    dpi=300,
)


##################
# save checkpoints
##################


adata_input.write("fig3_larry_allcells_top2000_model1.h5ad")
adata_model_pos_split[0].save("Fig3_allcells_model1", overwrite=True)

result_dict = {
    "adata_model_pos": adata_model_pos_split[1],
    "v_map_all": v_map_all,
    "embeds_radian": embeds_radian,
    "fdri": fdri,
    "embed_mean": embed_mean,
}

with open("fig3_allcells_data_model1.pkl", "wb") as f:
    pickle.dump(result_dict, f)
