import scvelo as scv
import numpy as np
import torch
import matplotlib.pyplot as plt
from pyrovelocity.data import load_data
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr, pearsonr
from pyrovelocity.api import train_model
import seaborn as sns
import pandas as pd
from pyrovelocity.plot import plot_posterior_time, plot_gene_ranking,\
      vector_field_uncertainty, plot_vector_field_uncertain,\
      plot_mean_vector_field, project_grid_points,rainbowplot,denoised_umap,\
      us_rainbowplot, plot_arrow_examples
import matplotlib
from pyrovelocity.utils import mae, mae_evaluate

adata = load_data(top_n=2000, min_shared_counts=30)
adata_model_pos = train_model(adata, max_epochs=1, svi_train=False, log_every=1000,
                              patient_init=45, batch_size=-1, use_gpu=0,
                              include_prior=True, offset=True, library_size=True,
                              patient_improve=1e-4, guide_type='auto', train_size=1.0)
# Model 2
import torch
import pyro
demo_data = torch.ones(adata.shape[0], adata.shape[1]).to('cuda:0')
pyrovelocity_graph = pyro.render_model(adata_model_pos[0].module._model,
                                       model_args=(demo_data,
                                                   demo_data,
                                                   demo_data,
                                                   demo_data,
                                                   demo_data,
                                                   demo_data,
                                                   demo_data, demo_data),
                                       render_params=True,
                                       render_distributions=True, filename="suppfig1_graph_model2.pdf")
# pyrovelocity_graph.unflatten(stagger=2)


adata = load_data(top_n=2000, min_shared_counts=30)
adata_model_pos = train_model(adata, max_epochs=1, svi_train=False, log_every=1000,
                              patient_init=45, batch_size=-1, use_gpu=0,
                              include_prior=True, offset=False, library_size=True,
                              patient_improve=1e-4, guide_type='auto_t0_constraint', train_size=1.0)

# Model 1
import torch
import pyro
demo_data = torch.ones(adata.shape[0], adata.shape[1]).to('cuda:0')
pyrovelocity_graph = pyro.render_model(adata_model_pos[0].module._model,
                                       model_args=(demo_data,
                                                   demo_data,
                                                   demo_data,
                                                   demo_data,
                                                   demo_data,
                                                   demo_data,
                                                   demo_data, demo_data),
                                       render_params=True,
                                       render_distributions=True, filename="suppfig1_graph_model1.pdf")
# pyrovelocity_graph.unflatten(stagger=2)
