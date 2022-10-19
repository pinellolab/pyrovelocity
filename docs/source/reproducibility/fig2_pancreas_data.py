#from mpl_toolkits.axes_grid1 import make_axes_locatable
#import scvelo as scv
#import numpy as np
#import torch
#import matplotlib.pyplot as plt
#from dynamical_velocity2.data import load_data
#from sklearn.model_selection import train_test_split
#from scipy.stats import spearmanr, pearsonr
#from dynamical_velocity2.api import train_model
#import seaborn as sns
#import pandas as pd
#from dynamical_velocity2.plot import plot_posterior_time, plot_gene_ranking,\
#      vector_field_uncertainty, plot_vector_field_uncertain,\
#      plot_mean_vector_field, project_grid_points,rainbowplot,denoised_umap,\
#      us_rainbowplot, plot_arrow_examples
#import matplotlib
#from dynamical_velocity2.utils import mae, mae_evaluate
#
#adata = load_data(top_n=2000, min_shared_counts=30)
#
#adata_model_pos = train_model(adata, max_epochs=4000, svi_train=False, log_every=1000,
#                              patient_init=45, batch_size=-1, use_gpu=0,
#                              include_prior=True, offset=False, library_size=True,
#                              patient_improve=1e-4, guide_type='auto_t0_constraint', train_size=1.0)
#
#v_map_all, embeds_radian, fdri = vector_field_uncertainty(adata, adata_model_pos[1], basis='umap')
#
#fig, ax = plt.subplots()
#embed_mean = plot_mean_vector_field(adata_model_pos[1], adata, ax=ax)
#
#adata.write("fig2_pancreas_processed.h5ad")
#
#adata_model_pos[0].save('Fig2_pancreas_model', overwrite=True)
#result_dict = {"adata_model_pos": adata_model_pos[1], "v_map_all": v_map_all, "embeds_radian": embeds_radian, "fdri": fdri, "embed_mean": embed_mean}
#import pickle
#
#with open("fig2_pancreas_data.pkl", "wb") as f:
#    pickle.dump(result_dict, f)

from mpl_toolkits.axes_grid1 import make_axes_locatable
import scvelo as scv
import numpy as np
import torch
import matplotlib.pyplot as plt
from dynamical_velocity2.data import load_data
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr, pearsonr
from dynamical_velocity2.api import train_model
import seaborn as sns
import pandas as pd
from dynamical_velocity2.plot import plot_posterior_time, plot_gene_ranking,\
      vector_field_uncertainty, plot_vector_field_uncertain,\
      plot_mean_vector_field, project_grid_points,rainbowplot,denoised_umap,\
      us_rainbowplot, plot_arrow_examples
import matplotlib
from dynamical_velocity2.utils import mae, mae_evaluate

adata = load_data(top_n=2000, min_shared_counts=30)

adata_model_pos = train_model(adata, max_epochs=4000, svi_train=False, log_every=1000,
                              patient_init=45, batch_size=-1, use_gpu=0,
                              include_prior=True, offset=False, library_size=True,
                              patient_improve=1e-4, guide_type='auto_t0_constraint', train_size=1.0)

def check_shared_time(adata_model_pos, adata):
    adata.obs['cell_time'] = adata_model_pos[1]['cell_time'].squeeze().mean(0)
    adata.obs['1-Cytotrace'] = 1-adata.obs['cytotrace']
    fig, ax = plt.subplots(1, 3)
    fig.set_size_inches(15, 3)
    scv.pl.scatter(adata, color='latent_time', show=False,
                   ax=ax[0], title='scvelo %.2f' % spearmanr(1-adata.obs.cytotrace, adata.obs.latent_time)[0],
                   cmap='RdBu_r', basis='umap')
    scv.pl.scatter(adata, color='cell_time', show=False, basis='umap',
                   ax=ax[1], title='pyro %.2f' % spearmanr(1-adata.obs.cytotrace, adata.obs.cell_time)[0])
    scv.pl.scatter(adata, color='1-Cytotrace', show=False, ax=ax[2])
    print(spearmanr(adata.obs.cytotrace, adata.obs.cell_time))
    print(spearmanr(adata.obs.cell_time, adata.obs.latent_time))
    fig.savefig("fig2_test_sub.pdf", facecolor=fig.get_facecolor(), bbox_inches='tight', edgecolor='none', dpi=300)

check_shared_time(adata_model_pos, adata)

fig, ax = plt.subplots()
volcano_data, _ = plot_gene_ranking([adata_model_pos[1]], [adata], ax=ax, time_correlation_with='st')
fig.savefig("fig2_test_volcano_sub.pdf", facecolor=fig.get_facecolor(), bbox_inches='tight', edgecolor='none', dpi=300)

print(volcano_data.sort_values("mean_mae", ascending=False).head(300).sort_values("time_correlation", ascending=False).head(8))
fig = us_rainbowplot(volcano_data.sort_values("mean_mae", ascending=False).head(300).sort_values("time_correlation", ascending=False).head(5).index,
                     adata, adata_model_pos[1], data=['st', 'ut'])
fig.savefig("fig2_test_rainbow_sub.pdf", facecolor=fig.get_facecolor(), bbox_inches='tight', edgecolor='none', dpi=300)

v_map_all, embeds_radian, fdri = vector_field_uncertainty(adata, adata_model_pos[1], basis='umap')

fig, ax = plt.subplots()
embed_mean = plot_mean_vector_field(adata_model_pos[1], adata, ax=ax)
fig.savefig("fig2_test_vecfield_sub.pdf", facecolor=fig.get_facecolor(), bbox_inches='tight', edgecolor='none', dpi=300)

adata.write("fig2_pancreas_processed.h5ad")

adata_model_pos[0].save('Fig2_pancreas_model', overwrite=True)
result_dict = {"adata_model_pos": adata_model_pos[1], "v_map_all": v_map_all, "embeds_radian": embeds_radian, "fdri": fdri, "embed_mean": embed_mean}
import pickle

with open("fig2_pancreas_data.pkl", "wb") as f:
    pickle.dump(result_dict, f)

