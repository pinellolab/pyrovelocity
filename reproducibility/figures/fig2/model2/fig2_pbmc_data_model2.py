import matplotlib.pyplot as plt
import scvelo as scv

from pyrovelocity.api import train_model
from pyrovelocity.plot import plot_mean_vector_field
from pyrovelocity.plot import vector_field_uncertainty


kwargs = dict(
    linewidth=1.5,
    density=0.8,
    color="celltype",
    frameon=False,
    add_margin=0.1,
    alpha=0.1,
    min_mass=3.5,
    add_outline=True,
    outline_width=(0.02, 0.02),
)


adata = scv.read("pbmc_processed.h5ad")

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
    adata, adata_model_pos[1], n_jobs=1
)

fig, ax = plt.subplots()
embed_mean = plot_mean_vector_field(adata_model_pos[1], adata, ax=ax, basis="tsne")

adata.write("fig2_pbmc_processed_model2.h5ad")
adata_model_pos[0].save("Fig2_pbmc_model2", overwrite=True)

result_dict = {
    "adata_model_pos": adata_model_pos[1],
    "v_map_all": v_map_all,
    "embeds_radian": embeds_radian,
    "fdri": fdri,
    "embed_mean": embed_mean,
}
import pickle


with open("fig2_pbmc_data_model2.pkl", "wb") as f:
    pickle.dump(result_dict, f)
