import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scvelo as scv
import seaborn as sns
import torch
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
from dynamical_velocity2.utils import mae
from dynamical_velocity2.utils import mae_evaluate
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split


adata = load_data(top_n=2000, min_shared_counts=30)
adata_model_pos = train_model(
    adata,
    max_epochs=1,
    svi_train=False,
    log_every=1000,
    patient_init=45,
    batch_size=-1,
    use_gpu=0,
    include_prior=True,
    offset=True,
    library_size=True,
    patient_improve=1e-4,
    guide_type="auto",
    train_size=1.0,
)
import pyro

# Model 2
import torch


demo_data = torch.ones(adata.shape[0], adata.shape[1]).to("cuda:0")
pyrovelocity_graph = pyro.render_model(
    adata_model_pos[0].module._model,
    model_args=(
        demo_data,
        demo_data,
        demo_data,
        demo_data,
        demo_data,
        demo_data,
        demo_data,
        demo_data,
    ),
    render_params=True,
    render_distributions=True,
    filename="suppfig1_graph_model2.pdf",
)
# pyrovelocity_graph.unflatten(stagger=2)


adata = load_data(top_n=2000, min_shared_counts=30)
adata_model_pos = train_model(
    adata,
    max_epochs=1,
    svi_train=False,
    log_every=1000,
    patient_init=45,
    batch_size=-1,
    use_gpu=0,
    include_prior=True,
    offset=False,
    library_size=True,
    patient_improve=1e-4,
    guide_type="auto_t0_constraint",
    train_size=1.0,
)

import pyro

# Model 1
import torch


demo_data = torch.ones(adata.shape[0], adata.shape[1]).to("cuda:0")
pyrovelocity_graph = pyro.render_model(
    adata_model_pos[0].module._model,
    model_args=(
        demo_data,
        demo_data,
        demo_data,
        demo_data,
        demo_data,
        demo_data,
        demo_data,
        demo_data,
    ),
    render_params=True,
    render_distributions=True,
    filename="suppfig1_graph_model1.pdf",
)
# pyrovelocity_graph.unflatten(stagger=2)
