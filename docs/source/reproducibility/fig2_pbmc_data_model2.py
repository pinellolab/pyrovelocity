import functools
from collections import defaultdict
from typing import Iterable
from typing import Literal
from typing import Optional
from typing import Tuple
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pyro
import scvelo as scv
import seaborn as sns
import torch
from dynamical_velocity2 import PyroVelocity
from dynamical_velocity2._trainer import VelocityClippedAdam
from dynamical_velocity2._velocity_guide import AuxCellVelocityGuide
from dynamical_velocity2._velocity_model import AuxCellVelocityModel
from dynamical_velocity2._velocity_model import TimeEncoder
from dynamical_velocity2.cytotrace import cytotrace_sparse
from dynamical_velocity2.data import load_data
from dynamical_velocity2.utils import debug
from dynamical_velocity2.utils import mRNA
from dynamical_velocity2.utils import ode_mRNA
from dynamical_velocity2.utils import tau_inv
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pyro import poutine
from pyro.distributions import Bernoulli
from pyro.distributions import Delta
from pyro.distributions import HalfNormal
from pyro.distributions import LogNormal
from pyro.distributions import NegativeBinomial
from pyro.distributions import Normal
from pyro.distributions import Poisson
from pyro.distributions.constraints import positive
from pyro.infer import SVI
from pyro.infer import JitTrace_ELBO
from pyro.infer import JitTraceMeanField_ELBO
from pyro.infer import Predictive
from pyro.infer import Trace_ELBO
from pyro.infer import TraceEnum_ELBO
from pyro.infer import TraceMeanField_ELBO
from pyro.infer.autoguide import AutoDelta
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.infer.autoguide import AutoDiscreteParallel
from pyro.infer.autoguide import AutoLowRankMultivariateNormal
from pyro.infer.autoguide import AutoNormal
from pyro.infer.autoguide import init_to_mean
from pyro.infer.autoguide.guides import AutoGuideList
from pyro.nn import PyroModule
from pyro.nn import PyroParam
from pyro.nn import PyroSample
from pyro.optim.clipped_adam import ClippedAdam
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scvelo.datasets import simulation
from scvi.nn import Decoder
from scvi.nn import DecoderSCVI
from scvi.nn import Encoder
from scvi.nn import FCLayers
from torch import nn
from torch.nn.functional import relu
from torch.nn.functional import softmax
from torch.nn.functional import softplus


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
import matplotlib
import pandas as pd
import seaborn as sns
from dynamical_velocity2.api import train_model
from dynamical_velocity2.plot import plot_arrow_examples
from dynamical_velocity2.plot import plot_gene_ranking
from dynamical_velocity2.plot import plot_mean_vector_field
from dynamical_velocity2.plot import plot_posterior_time
from dynamical_velocity2.plot import plot_vector_field_uncertain
from dynamical_velocity2.plot import project_grid_points
from dynamical_velocity2.plot import rainbowplot
from dynamical_velocity2.plot import vector_field_uncertainty
from scipy.stats import pearsonr
from scipy.stats import spearmanr


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
