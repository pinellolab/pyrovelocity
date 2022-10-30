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
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

##!mkdir -p LARRY_figure
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


adata = scv.read("mono_unipotent_cells.h5ad")
adata_input = adata[adata.obs.state_info != "Centroid", :].copy()

if not os.path.exists("larry_mono_top2000.h5ad"):
    scv.pp.filter_and_normalize(adata_input, n_top_genes=2000, min_shared_counts=20)
    scv.pp.moments(adata_input)
    ###    scv.pp.filter_and_normalize(adata_input, min_shared_counts=30, n_top_genes=2000)
    ###    scv.pp.moments(adata_input, n_pcs=30, n_neighbors=30)
    scv.tl.recover_dynamics(adata_input, n_jobs=10)
    scv.tl.velocity(adata_input, mode="dynamical")
    scv.tl.velocity_graph(adata_input)
    scv.tl.velocity_embedding(adata_input, basis="emb")
    scv.tl.latent_time(adata_input)
else:
    adata_input = scv.read("larry_mono_top2000.h5ad")

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
import pickle


with open("fig3_mono_data_model1.pkl", "wb") as f:
    pickle.dump(result_dict, f)
