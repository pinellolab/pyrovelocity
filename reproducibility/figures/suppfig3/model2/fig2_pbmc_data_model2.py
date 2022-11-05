import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
import scvelo as scv

from pyrovelocity.api import train_model
from pyrovelocity.plot import plot_mean_vector_field
from pyrovelocity.plot import vector_field_uncertainty


if os.path.exists("pbmc_processed.h5ad"):
    adata = scv.read("pbmc_processed.h5ad")
else:
    adata = scv.datasets.pbmc68k()
    adata.obsm["X_tsne"][:, 0] *= -1
    scv.pp.remove_duplicate_cells(adata)
    scv.pp.filter_and_normalize(adata, min_shared_counts=30, n_top_genes=2000)
    scv.pp.moments(adata)
    scv.tl.velocity(adata, mode='stochastic')
    scv.tl.recover_dynamics(adata, n_jobs=-1)
    top_genes = adata.var['fit_likelihood'].sort_values(ascending=False).index
    adata_sub =  adata[:, top_genes[:3]].copy()
    scv.tl.velocity_graph(adata_sub, n_jobs=-1)
    scv.tl.velocity_embedding(adata_sub)

    adata_all = scv.datasets.pbmc68k()
    adata_sub.layers['raw_spliced'] = adata_all[:, adata_sub.var_names].layers['spliced']
    adata_sub.layers['raw_unspliced'] = adata_all[:, adata_sub.var_names].layers['unspliced']
    adata_sub.obs['u_lib_size_raw'] = np.array(adata_sub.layers['raw_unspliced'].sum(axis=-1), dtype=np.float32).flatten()
    adata_sub.obs['s_lib_size_raw'] = np.array(adata_sub.layers['raw_spliced'].sum(axis=-1), dtype=np.float32).flatten()
    adata = adata_sub.copy()
    adata.write("pbmc_processed.h5ad")

adata_model_pos = train_model(
    adata,
    max_epochs=4000,
    svi_train=False,
    # log_every=2000, # old
    log_every=100,  # old
    patient_init=45,
    batch_size=-1,
    use_gpu=0,
    cell_state="celltype",
    patient_improve=1e-4,
    guide_type="auto",
    train_size=1.0,
    offset=True,
    library_size=True,
    include_prior=True,
)

v_map_all, embeds_radian, fdri = vector_field_uncertainty(
    adata, adata_model_pos[1], n_jobs=20
)

fig, ax = plt.subplots()
embed_mean = plot_mean_vector_field(adata_model_pos[1], adata, ax=ax, basis="tsne", n_jobs=20)

adata.write("fig2_pbmc_processed_model2.h5ad")
adata_model_pos[0].save("Fig2_pbmc_model2", overwrite=True)

result_dict = {
    "adata_model_pos": adata_model_pos[1],
    "v_map_all": v_map_all,
    "embeds_radian": embeds_radian,
    "fdri": fdri,
    "embed_mean": embed_mean,
}

with open("fig2_pbmc_data_model2.pkl", "wb") as f:
    pickle.dump(result_dict, f)
