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
from scipy.stats import spearmanr, pearsonr
from dynamical_velocity2.api import train_model
import seaborn as sns
import pandas as pd
import pickle
from dynamical_velocity2.plot import plot_posterior_time, plot_gene_ranking,\
      vector_field_uncertainty, plot_vector_field_uncertain,\
      plot_mean_vector_field, project_grid_points,rainbowplot
import matplotlib

with open("fig2_pancreas_data.pkl", "rb") as f:
    result_dict = pickle.load(f)

adata_model_pos = result_dict['adata_model_pos']
v_map_all = result_dict['v_map_all']
embeds_radian = result_dict['embeds_radian']
fdri = result_dict['fdri']
embed_mean = result_dict['embed_mean']

adata = scv.read("fig2_pancreas_processed.h5ad")

from dynamical_velocity2.cytotrace import compute_similarity2
df_genes_cors = compute_similarity2(adata.layers['spliced'].toarray(), adata.obs.latent_time.values.reshape(-1, 1))

scvelo_top = pd.DataFrame({"cor": df_genes_cors[0], 
                           "likelihood": adata.var['fit_likelihood'],
                           "genes": adata.var_names})
_, ax = plt.subplots()
volcano_data2, _ = plot_gene_ranking([adata_model_pos], [adata], ax=ax,
                                     time_correlation_with='st', assemble=True)

with pd.ExcelWriter('SuppTable1.xlsx') as writer:
    volcano_data2.sort_values("mean_mae", ascending=False).head(300).sort_values("time_correlation", ascending=False).head(50).to_excel(writer, sheet_name='pyro_velocity_top_positive_correlation_genes')
    scvelo_top.sort_values("likelihood", ascending=False).head(300).sort_values("cor", ascending=False).head(50).to_excel(writer, sheet_name='scvelo_top_positive_correlation_genes')
    volcano_data2.sort_values("mean_mae", ascending=False).head(300).sort_values("time_correlation", ascending=True).head(50).to_excel(writer, sheet_name='pyro_velocity_top_negative_correlation_genes')
    scvelo_top.sort_values("likelihood", ascending=False).head(300).sort_values("cor", ascending=True).head(50).to_excel(writer, sheet_name='scvelo_top_negative_correlation_genes')


with open("fig2_pancreas_data_model2.pkl", "rb") as f:
    result_dict = pickle.load(f)

adata_model_pos = result_dict['adata_model_pos']
v_map_all = result_dict['v_map_all']
embeds_radian = result_dict['embeds_radian']
fdri = result_dict['fdri']
embed_mean = result_dict['embed_mean']

adata = scv.read("fig2_pancreas_processed_model2.h5ad")

from dynamical_velocity2.cytotrace import compute_similarity2
df_genes_cors = compute_similarity2(adata.layers['spliced'].toarray(), adata.obs.latent_time.values.reshape(-1, 1))

scvelo_top = pd.DataFrame({"cor": df_genes_cors[0], 
                           "likelihood": adata.var['fit_likelihood'],
                           "genes": adata.var_names})
_, ax = plt.subplots()
volcano_data2, _ = plot_gene_ranking([adata_model_pos], [adata], ax=ax,
                                     time_correlation_with='st', assemble=True)

with pd.ExcelWriter('SuppTable2.xlsx') as writer:
    volcano_data2.sort_values("mean_mae", ascending=False).head(300).sort_values("time_correlation", ascending=False).head(50).to_excel(writer, sheet_name='pyro_velocity_top_positive_correlation_genes')
    scvelo_top.sort_values("likelihood", ascending=False).head(300).sort_values("cor", ascending=False).head(50).to_excel(writer, sheet_name='scvelo_top_positive_correlation_genes')
    volcano_data2.sort_values("mean_mae", ascending=False).head(300).sort_values("time_correlation", ascending=True).head(50).to_excel(writer, sheet_name='pyro_velocity_top_negative_correlation_genes')
    scvelo_top.sort_values("likelihood", ascending=False).head(300).sort_values("cor", ascending=True).head(50).to_excel(writer, sheet_name='scvelo_top_negative_correlation_genes')


