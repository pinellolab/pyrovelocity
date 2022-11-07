import pickle
import scvelo as scv
import os
import matplotlib.pyplot as plt

from pyrovelocity.api import train_model
from pyrovelocity.plot import plot_mean_vector_field
from pyrovelocity.plot import vector_field_uncertainty
from pyrovelocity.data import load_larry, load_unipotent_larry

if os.path.exists("mono_unipotent_cells.h5ad"):
    adata = scv.read("mono_unipotent_cells.h5ad")
else:
    adata = load_unipotent_larry()

adata_input = adata[adata.obs.state_info != "Centroid", :].copy()
scv.pp.filter_and_normalize(adata_input, n_top_genes=2000, min_shared_counts=20)
scv.pp.moments(adata_input)
scv.tl.recover_dynamics(adata_input, n_jobs=10)
scv.tl.velocity(adata_input, mode="dynamical")
scv.tl.velocity_graph(adata_input)
scv.tl.velocity_embedding(adata_input, basis="emb")
scv.tl.latent_time(adata_input)

adata_input.layers["raw_spliced"] = adata[
    adata_input.obs_names, adata_input.var_names
].layers["spliced"]
adata_input.layers["raw_unspliced"] = adata[
    adata_input.obs_names, adata_input.var_names
].layers["unspliced"]

adata_input.obs["u_lib_size_raw"] = adata_input.layers["unspliced"].toarray().sum(-1)
adata_input.obs["s_lib_size_raw"] = adata_input.layers["spliced"].toarray().sum(-1)

adata_model_pos_split = train_model(
    adata_input,
    max_epochs=4000,
    svi_train=False,
    log_every=100,
    patient_init=45,
    batch_size=-1,
    use_gpu=1,
    cell_state="state_info",
    include_prior=True,
    offset=False,
    library_size=True,
    patient_improve=1e-4,
    guide_type="auto_t0_constraint",
    train_size=1.0,
)

v_map_all, embeds_radian, fdri = vector_field_uncertainty(
    adata_input, adata_model_pos_split[1], basis="emb"
)

fig, ax = plt.subplots()
embed_mean = plot_mean_vector_field(
    adata_model_pos_split[1], adata_input, ax=ax, basis="emb"
)

adata_input.write("fig3_mono_processed_model1.h5ad")

adata_model_pos_split[0].save("Fig3_mono_model1", overwrite=True)
result_dict = {
    "adata_model_pos": adata_model_pos_split[1],
    "v_map_all": v_map_all,
    "embeds_radian": embeds_radian,
    "fdri": fdri,
    "embed_mean": embed_mean,
}
with open("fig3_mono_data_model1.pkl", "wb") as f:
    pickle.dump(result_dict, f)
