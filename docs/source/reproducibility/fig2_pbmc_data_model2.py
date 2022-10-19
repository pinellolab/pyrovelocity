import pyro
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scvelo as scv
from scvelo.datasets import simulation
import numpy as np
import matplotlib.pyplot as plt
from dynamical_velocity2 import PyroVelocity
from dynamical_velocity2.data import load_data
from scipy.stats import pearsonr, spearmanr
import seaborn as sns
from dynamical_velocity2.data import load_data
from dynamical_velocity2._velocity_guide import AuxCellVelocityGuide
from dynamical_velocity2._velocity_model import AuxCellVelocityModel
from pyro.infer.autoguide import AutoDelta, AutoNormal, init_to_mean, AutoLowRankMultivariateNormal, AutoDiagonalNormal, AutoDiscreteParallel
from pyro.infer.autoguide.guides import AutoGuideList
from pyro import poutine
from pyro.distributions import LogNormal, Normal, Poisson, Bernoulli, Delta, NegativeBinomial, HalfNormal
import pyro
from torch.nn.functional import softplus, softmax, relu
from dynamical_velocity2.utils import mRNA, debug, tau_inv, ode_mRNA
from collections import defaultdict
import functools
from pyro.optim.clipped_adam import ClippedAdam
from dynamical_velocity2.cytotrace import cytotrace_sparse
import torch

from scvi.nn import DecoderSCVI, Encoder, Decoder,FCLayers
from dynamical_velocity2._trainer import VelocityClippedAdam
from pyro.infer import SVI, Trace_ELBO, Predictive, JitTrace_ELBO, TraceEnum_ELBO, TraceMeanField_ELBO, JitTraceMeanField_ELBO
from pyro.distributions.constraints import positive
from pyro.nn import PyroParam, PyroSample, PyroModule
from typing import Optional, Union, Tuple, Literal, Iterable
import torch
from dynamical_velocity2._velocity_model import TimeEncoder
from torch import nn
import scvelo as scv
from scvelo.datasets import simulation
import numpy as np
import matplotlib.pyplot as plt
from dynamical_velocity2 import PyroVelocity
from dynamical_velocity2.data import load_data
from scipy.stats import pearsonr, spearmanr
import seaborn as sns
kwargs = dict(linewidth=1.5, density=.8, color='celltype', frameon=False, add_margin=.1, alpha=.1, min_mass=3.5, add_outline=True, outline_width=(.02, .02))
from scipy.stats import spearmanr, pearsonr
from dynamical_velocity2.api import train_model
import seaborn as sns
import pandas as pd
from dynamical_velocity2.plot import plot_posterior_time, plot_gene_ranking,\
      vector_field_uncertainty, plot_vector_field_uncertain,\
      plot_mean_vector_field, project_grid_points,rainbowplot,plot_arrow_examples
import matplotlib

adata = scv.read("pbmc_processed.h5ad")

adata_model_pos = train_model(adata, max_epochs=4000, svi_train=False,
                             #log_every=2000, # old
                              log_every=100, # old
                              patient_init=45, batch_size=-1, use_gpu=0, cell_state='celltype',
                              patient_improve=1e-4, guide_type='auto', train_size=1.0,
                              offset=True,
                              library_size=True,
                              include_prior=True)

v_map_all, embeds_radian, fdri = vector_field_uncertainty(adata, adata_model_pos[1], n_jobs=1)

fig, ax = plt.subplots()
embed_mean = plot_mean_vector_field(adata_model_pos[1], adata, ax=ax, basis='tsne')

adata.write("fig2_pbmc_processed_model2.h5ad")
adata_model_pos[0].save('Fig2_pbmc_model2', overwrite=True)

result_dict = {"adata_model_pos": adata_model_pos[1], "v_map_all": v_map_all, "embeds_radian": embeds_radian, "fdri": fdri, "embed_mean": embed_mean}
import pickle

with open("fig2_pbmc_data_model2.pkl", "wb") as f:
    pickle.dump(result_dict, f)

