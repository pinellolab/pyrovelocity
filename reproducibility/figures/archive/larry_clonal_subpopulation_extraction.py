# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Gold standard for global

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import scvelo as scv
from scvelo.datasets import simulation
import numpy as np
import matplotlib.pyplot as plt
from dynamical_velocity2 import PyroVelocity
from dynamical_velocity2.data import load_data
from scipy.stats import pearsonr, spearmanr
import seaborn as sns
from dynamical_velocity2.data import load_data
import cospar as cs
import numpy as np
import scvelo as scv
cs.logging.print_version()
cs.settings.verbosity=2
cs.settings.data_path='LARRY_data' # A relative path to save data. If not existed before, create a new one.
cs.settings.figure_path='LARRY_figure' # A relative path to save figures. If not existed before, create a new one.
cs.settings.set_figure_params(format='png',figsize=[4,3.5],dpi=75,fontsize=14,pointsize=2)
# !mkdir -p LARRY_figure
import scvelo as scv
from scvelo.datasets import simulation
import numpy as np
import matplotlib.pyplot as plt
from dynamical_velocity2 import PyroVelocity
from dynamical_velocity2.data import load_data
from scipy.stats import pearsonr, spearmanr
import seaborn as sns
from dynamical_velocity2.data import load_data

# %%
from scipy.sparse import issparse

# %%
#adata_processed = scv.read("../notebooks/larry_invitro_adata_sub_withvelocitycospar.h5ad")
adata_processed = scv.read("../notebooks/larry_invitro_adata_with_scvelo_dynamicalvelocity.h5ad")

# %% [markdown]
# # Check processed scvelo adata

# %%
scv.tl.velocity_embedding(adata_processed, vkey='velocity', basis='emb')
scv.pl.velocity_embedding_grid(adata_processed, figsize=(15, 15), scale=0.3, basis='emb', color='state_info')

# %%
adata_processed.obs.state_info.cat.categories

# %%
adata_processed.var.velocity_genes.sum(), adata_processed.shape

# %% [markdown]
# # Run old Normal likelihood model 

# %%
adata = scv.read("../notebooks/larry_invitro_adata_sub_raw.h5ad")
adata_processed.layers['raw_spliced'] = adata[:, adata_processed.var_names].layers['spliced']
adata_processed.layers['raw_unspliced'] = adata[:, adata_processed.var_names].layers['unspliced']

# %%
adata.shape, adata_processed.shape

# %%
adata_processed.obs.time_info.unique()

# %%
adata_cospar = scv.read("../notebooks/LARRY_data/LARRY_MultiTimeClone_Later_FullSpace0_t*2.0*4.0*6_adata_with_transition_map.h5ad")

# %%
adata_cytotrace = scv.read("../notebooks/larry_invitro_adata_sub_raw_withcytotrace.h5ad")

# %%
cs.pl.fate_bias(adata_cospar, selected_fates=['Neutrophil','Monocyte'],used_Tmap='transition_map',
                selected_times=[2,4], plot_target_state=False,map_backward=True,sum_fate_prob_thresh=0.1)

# %%
import anndata

def get_clone_trajectory(adata, average_start_point=True, global_traj=True, times=[2, 4, 6], clone_num=None):
    if not average_start_point:
        adata.obsm['clone_vector_emb'] = np.zeros((adata.shape[0], 2))

    adatas = []
    clones = []
    centroids = []
    cen_clones = []
    print(adata.shape)
    adata.obs['clones'] = 0
    if 'noWell' in adata.obs.columns:
        for w in adata.obs.Well.unique():
            adata_w = adata[adata.obs.Well==w]
            clone_adata_w = clone_adata[clone_adata.obs.Well == w]            
            for j in range(clone_adata_w.shape[1]):
                adata_w.obs['clonei'] = 0
                # belongs to same clone
                adata_w.obs.loc[clone_adata_w[:, j].X.toarray()[:, 0] >= 1, 'clonei'] = 1
                
                if not average_start_point:
                    for i in np.where((adata_w.obs.time==2) & (adata_w.obs.clonei==1))[0]:        
                        next_time = np.where((adata_w.obs.time==4) & (adata_w.obs.clonei==1))[0]
                        adata_w.obsm['velocity_umap'][i] = adata_w.obsm['X_umap'][next_time].mean(axis=0)-adata_w.obsm['X_umap'][i]            
                    for i in np.where((adata_w.obs.time==4) & (adata_w.obs.clonei==1))[0]:
                        next_time = np.where((adata_w.obs.time==6) & (adata_w.obs.clonei==1))[0]
                        adata_w.obsm['velocity_umap'][i] = adata_w.obsm['X_umap'][next_time].mean(axis=0)-adata_w.obsm['X_umap'][i]            
                else:
                    time2 = np.where((adata_w.obs.time==2) & (adata_w.obs.clonei==1))[0]
                    time4 = np.where((adata_w.obs.time==4) & (adata_w.obs.clonei==1))[0]
                    time6 = np.where((adata_w.obs.time==6) & (adata_w.obs.clonei==1))[0]
                    if time2.shape[0] == 0 and time4.shape[0] == 0 and time6.shape[0] == 0:
                        continue
                    if time2.shape[0] > 0 and time4.shape[0] == 0 and time6.shape[0] > 0:
                        continue
                    adata_new = anndata.AnnData(np.vstack([adata_w[time2].X.toarray().mean(axis=0), 
                                                 adata_w[time4].X.toarray().mean(axis=0), 
                                                 adata_w[time6].X.toarray().mean(axis=0)]), 
                                                layers={'spliced': np.vstack([adata_w[time2].layers['spliced'].toarray().mean(axis=0), 
                                                 adata_w[time4].layers['spliced'].toarray().mean(axis=0), 
                                                 adata_w[time6].layers['spliced'].toarray().mean(axis=0)]), 
                                                       'unspliced': np.vstack([adata_w[time2].layers['unspliced'].toarray().mean(axis=0), 
                                                 adata_w[time4].layers['unspliced'].toarray().mean(axis=0), 
                                                 adata_w[time6].layers['unspliced'].toarray().mean(axis=0)])},
                                                var=adata_w.var)
                    
                    adata_new.obs.loc[:, 'time'] = [2, 4, 6]
                    adata_new.obs.loc[:, 'Cell type annotation'] = 'Centroid'
                    print(adata_w[time6].obs.clonetype.unique())
                    print(adata_w[time6].obs)
                    
                    adata_new.obs.loc[:, 'clonetype'] = adata_w[time6].obs.clonetype.unique() # use cell fate from last time point
                    adata_new.obs.loc[:, 'clones'] = int(j)
                    if 'Well' in adata_w[time6].obs.columns:
                        adata_new.obs.loc[:, 'Well'] = adata_w[time6].obs.Well.unique()
                        
                    adata_new.obsm['X_umap'] = np.vstack([adata_w[time2].obsm['X_umap'].mean(axis=0), 
                                               adata_w[time4].obsm['X_umap'].mean(axis=0),
                                               adata_w[time6].obsm['X_umap'].mean(axis=0)])
                    adata_new.obsm['velocity_umap'] = np.vstack([adata_w.obsm['X_umap'][time4].mean(axis=0) - adata_w.obsm['X_umap'][time2].mean(axis=0),
                                                             adata_w.obsm['X_umap'][time6].mean(axis=0) - adata_w.obsm['X_umap'][time4].mean(axis=0),
                                                             np.zeros(2)])
                    centroids.append(adata_new)            
                    clone_new = anndata.AnnData(np.vstack([clone_adata_w[time2].X.toarray().mean(axis=0), 
                                                clone_adata_w[time4].X.toarray().mean(axis=0), 
                                                clone_adata_w[time6].X.toarray().mean(axis=0)]), 
                                                obs=adata_new.obs)
                    clone_new.var_names = clone_adata.var_names
                    clone_new.var = clone_adata.var
                    print(clone_new.shape)
                    cen_clones.append(clone_new)
                    
            adata_new = adata_w.concatenate(centroids[0].concatenate(centroids[1:]), join='outer')
            clone_new = clone_adata_w.concatenate(cen_clones[0].concatenate(cen_clones[1:]), join='outer')
            adatas.append(adata_new)
            clones.append(clone_new)
        return adatas[0].concatenate(adatas[1]), clones[0].concatenate(clones[1])
    else:
        if clone_num is None:
            clone_num = adata.obsm['X_clone'].shape[1]
        for j in range(clone_num):
            print(j)
            adata.obs['clonei'] = 0
            print('----------aa------')
            if issparse(adata.obsm['X_clone']):
                adata.obs.loc[adata.obsm['X_clone'].toarray()[:, j] >= 1, 'clonei'] = 1        
            else:
                adata.obs.loc[adata.obsm['X_clone'][:, j] >= 1, 'clonei'] = 1                
            print('----------bb------')
                
            if not average_start_point:
                for i in np.where((adata.obs.time==2) & (adata.obs.clonei==1))[0]:        
                    next_time = np.where((adata.obs.time==4) & (adata.obs.clonei==1))[0]
                    adata.obsm['velocity_umap'][i] = adata.obsm['X_umap'][next_time].mean(axis=0)-adata.obsm['X_umap'][i]            
                for i in np.where((adata.obs.time==4) & (adata.obs.clonei==1))[0]:
                    next_time = np.where((adata.obs.time==6) & (adata.obs.clonei==1))[0]
                    adata.obsm['velocity_umap'][i] = adata.obsm['X_umap'][next_time].mean(axis=0)-adata.obsm['X_umap'][i]            
            else:                
                if global_traj:         
                    times_index = []
                    for t in times:
                        times_index.append(np.where((adata.obs.time_info==t) & (adata.obs.clonei==1))[0])
                    
                    consecutive_flag = np.array([int(time.shape[0] > 0) for time in times_index])
                    consecutive = np.diff(consecutive_flag)
                    if np.sum(consecutive_flag == 1) >= 2 and np.any(consecutive == 0): # Must be consecutive time points                        
                        print('centroid:', consecutive, times_index) 
                        adata_new = anndata.AnnData(np.vstack([np.array(adata[time].X.mean(axis=0)).squeeze() for time in times_index if time.shape[0] > 0]),
#                                                     layers={'spliced': 
#                                                             np.vstack([np.array(adata[time].layers['spliced'].mean(axis=0)) for time in times_index if time.shape[0] > 0]), 
#                                                             'unspliced': 
#                                                             np.vstack([np.array(adata[time].layers['unspliced'].mean(axis=0)) for time in times_index if time.shape[0] > 0])
#                                                            },
                                                    var=adata.var)
                        print('----------cc------')                        
                        adata.obs.iloc[np.hstack([time for time in times_index if time.shape[0] > 0]), adata.obs.columns.get_loc('clones')] = int(j)           
                        adata_new.obs.loc[:, 'time'] = [t for t, time in zip([2, 4, 6], times_index) if time.shape[0] > 0]
                        adata_new.obs.loc[:, 'clones'] = int(j)
                        adata_new.obs.loc[:, 'state_info'] = 'Centroid'
                        adata_new.obsm['X_emb'] = np.vstack([adata[time].obsm['X_emb'].mean(axis=0) 
                                                             for time in times_index if time.shape[0] > 0])
                        print('----------dd------')

                        #print(adata_new.shape)
                        #print(adata_new.obsm['X_umap'])
                        adata_new.obsm['clone_vector_emb'] = np.vstack([adata_new.obsm['X_emb'][i+1] - adata_new.obsm['X_emb'][i]
                                                                        for i in range(adata_new.obsm['X_emb'].shape[0]-1)] + [np.zeros(2)])
                        print('----------ee------')                        
                        print(adata_new.obsm['clone_vector_emb'])
                    else: 
                        print('pass-------')
                        continue
                        
                else:                    
                    time2 = np.where((adata.obs.time==t) & (adata.obs.clonei==1))[0]
                    time4 = np.where((adata.obs.time==4) & (adata.obs.clonei==1))[0]
                    time6 = np.where((adata.obs.time==6) & (adata.obs.clonei==1))[0]                    
                    adata_new = anndata.AnnData(np.vstack([adata[time2].X.toarray().mean(axis=0), 
                                                 adata[time4].X.toarray().mean(axis=0), 
                                                 adata[time6].X.toarray().mean(axis=0)]), 
                                                layers={'spliced': np.vstack([adata[time2].layers['spliced'].toarray().mean(axis=0), 
                                                 adata[time4].layers['spliced'].toarray().mean(axis=0), 
                                                 adata[time6].layers['spliced'].toarray().mean(axis=0)]), 
                                                       'unspliced': np.vstack([adata[time2].layers['unspliced'].toarray().mean(axis=0), 
                                                 adata[time4].layers['unspliced'].toarray().mean(axis=0), 
                                                 adata[time6].layers['unspliced'].toarray().mean(axis=0)])},
                                                var=adata.var)
                    
                    print(adata_new.X.sum(axis=1))
                    adata_new.obs.loc[:, 'time'] = [2, 4, 6]
                    adata_new.obs.loc[:, 'Cell type annotation'] = 'Centroid'                
                    if not global_traj:
                        adata_new.obs.loc[:, 'clonetype'] = adata[time6].obs.clonetype.unique() # use cell fate from last time point
                    adata_new.obs.loc[:, 'clones'] = j

                    if 'noWell' in adata[time6].obs.columns:
                        adata_new.obs.loc[:, 'Well'] = adata[time6].obs.Well.unique()

                    adata_new.obsm['X_umap'] = np.vstack([adata[time2].obsm['X_umap'].mean(axis=0), 
                                               adata[time4].obsm['X_umap'].mean(axis=0),
                                               adata[time6].obsm['X_umap'].mean(axis=0)])
                    adata_new.obsm['velocity_umap'] = np.vstack([adata.obsm['X_umap'][time4].mean(axis=0) - adata.obsm['X_umap'][time2].mean(axis=0),
                                                             adata.obsm['X_umap'][time6].mean(axis=0) - adata.obsm['X_umap'][time4].mean(axis=0),
                                                             np.zeros(2)])
                
                    print(adata_new.obsm['velocity_umap'])
                    clone_new = anndata.AnnData(np.vstack([clone_adata[time2].X.toarray().mean(axis=0), 
                                                clone_adata[time4].X.toarray().mean(axis=0), 
                                                clone_adata[time6].X.toarray().mean(axis=0)]), 
                                                obs=adata_new.obs)
                    clone_new.var_names = clone_adata.var_names
                    clone_new.var = clone_adata.var
                    cen_clones.append(clone_new)                      
                centroids.append(adata_new)
        print(adata.shape)
        print(len(centroids))
        adata_new = adata.concatenate(centroids[0].concatenate(centroids[1:]), join='outer')
        return adata_new


# %%
adata.obs.head()

# %%
state_global_test = get_clone_trajectory(adata)

# %%
# state_global_test = scv.read("global_gold_standard2.h5ad")

# %%
# state_global_test2 = get_clone_trajectory(adata)

# %%
state_global_test.obsm

# %%
fig, ax = plt.subplots()
fig.set_size_inches(9, 9)
scv.pl.velocity_embedding_grid(state_global_test, basis='emb', vkey='clone_vector',
                               arrow_size=3, arrow_color='black', 
                               density=0.5, color='state_info', ax=ax, show=False, 
                               scale=0.3,
                               legend_loc='right')

# %%
fig, ax = plt.subplots()
fig.set_size_inches(9, 9)
scv.pl.velocity_embedding_stream(state_global_test, basis='emb', vkey='clone_vector',
                               arrow_size=3, arrow_color='black', 
                               density=0.8, color='state_info', ax=ax, show=False, 
                               legend_loc='right')

# %%
cs.pl.fate_potency(adata_processed, used_Tmap='transition_map',
                   map_backward=True,method='norm-sum',color_bar=True,fate_count=True)

adata_processed.uns['Tmap_cell_id_t1'].shape, adata_processed.uns['Tmap_cell_id_t2'].shape

# %%
adata_test = adata_processed.copy()

# %%
from scipy.sparse import csr_matrix
graph = np.zeros((adata_processed.shape[0], adata_processed.shape[0]))
for index, t1 in enumerate(adata_processed.uns['Tmap_cell_id_t1']):
    graph[t1, adata_processed.uns['Tmap_cell_id_t2']] = adata_processed.uns['transition_map'][index].toarray()
adata_test.uns['velocity_graph'] = csr_matrix(graph)
scv.tl.velocity_embedding(adata_test, basis='emb')

fig, ax = plt.subplots()
fig.set_size_inches(9, 9)
scv.pl.velocity_embedding_grid(adata_test, basis='emb', vkey='velocity', color='state_info',show=False, 
                               scale=0.3, ax=ax, arrow_size=3, arrow_color='red',
                               density=0.5,
                               legend_loc='right')

scv.pl.velocity_embedding_grid(adata_processed, basis='emb', vkey='velocity',color='state_info',show=False, 
                               scale=0.3, ax=ax, arrow_size=3, arrow_color='blue',
                               density=0.5,
                               legend_loc='right')

scv.pl.velocity_embedding_grid(state_global_test, basis='emb', vkey='clone_vector', color='state_info',show=False, 
                               scale=0.3, ax=ax, arrow_size=3, arrow_color='black',
                               density=0.5,
                               legend_loc='right')

# %%
graph = np.zeros((adata_processed.shape[0], adata_processed.shape[0]), dtype=np.uint8)

# %%
adata_test.uns['velocity_graph'] = csr_matrix(graph)
scv.tl.velocity_embedding(adata_test, basis='emb')

# %%
fig, ax = plt.subplots()
fig.set_size_inches(9, 9)
scv.pl.velocity_embedding_grid(adata_test, basis='emb', vkey='velocity', color='state_info',show=False, 
                               scale=0.3, ax=ax, arrow_size=3, arrow_color='red',
                               density=0.5,
                               legend_loc='right')

scv.pl.velocity_embedding_grid(adata_processed, basis='emb', vkey='velocity',color='state_info',show=False, 
                               scale=0.3, ax=ax, arrow_size=3, arrow_color='blue',
                               density=0.5,
                               legend_loc='right')

scv.pl.velocity_embedding_grid(state_global_test, basis='emb', vkey='clone_vector', color='state_info',show=False, 
                               scale=0.3, ax=ax, arrow_size=3, arrow_color='black',
                               density=0.5,
                               legend_loc='right')

# %%
from scipy.sparse import csr_matrix
graph = np.zeros((adata_processed.shape[0], adata_processed.shape[0]))
for index, t1 in enumerate(adata_processed.uns['Tmap_cell_id_t1']):
    graph[t1, adata_processed.uns['Tmap_cell_id_t2']] = adata_processed.uns['transition_map'][index].toarray()
adata_test.uns['velocity_graph'] = csr_matrix(graph)
scv.tl.velocity_embedding(adata_test, basis='emb')

# %%
fig, ax = plt.subplots(1, 3)
fig.set_size_inches(18, 5)

adata_processed.obs['fate_bias'] = np.nan
adata_processed.obs.loc[adata_processed.obs['fate_bias_Neutrophil_Monocyte']!=0.5, 'fate_bias'] = adata_processed.obs['fate_bias_Neutrophil_Monocyte'][adata_processed.obs['fate_bias_Neutrophil_Monocyte']!=0.5]

scv.pl.scatter(adata_processed, basis='emb', color='fate_bias', cmap='RdBu_r', show=False, ax=ax[0])

scv.pl.scatter(adata_processed, basis='emb', color='fate_potency', cmap='inferno', show=False, ax=ax[1])

scv.pl.velocity_embedding_grid(adata_test, basis='emb', vkey='velocity', color='state_info', show=False, 
                               scale=0.3, ax=ax[2], arrow_size=3, arrow_color='red',
                               density=0.5,
                               legend_loc='right')

scv.pl.velocity_embedding_grid(adata_processed, basis='emb', vkey='velocity',color='state_info',show=False, 
                               scale=0.3, ax=ax[2], arrow_size=3, arrow_color='blue',
                               density=0.5,
                               legend_loc='right')

scv.pl.velocity_embedding_grid(state_global_test, basis='emb', vkey='clone_vector', color='state_info',show=False, 
                               scale=0.3, ax=ax[2], arrow_size=3, arrow_color='black',
                               density=0.5,
                               legend_loc='right')
ax[2].legend(bbox_to_anchor=[0.1, -0.03], ncol=5, fontsize=7)

# %%
fig.savefig("Fig3_goldstandard.pdf", facecolor=fig.get_facecolor(), bbox_inches='tight', edgecolor='none', dpi=300)

# %%
state_global_test.shape

# %%
adata_processed.write("bifurcation_all_cells_withfatebias.h5ad")
adata_test.write("global_gold_standard.h5ad")
state_global_test.write("global_gold_standard2.h5ad")

# %%
# !du -sh *h5ad

# %%
state_global_test.obs.head()

# %%
state_global_test.obs.loc[:, 'state_info'].cat.categories


# %% [markdown]
# # Local gold standard

# %%
##choose clones that span three time points and go into the specified branches 
def select_clones(df_metadata,df_clones,ratio=1.0,cutoff_timepoints=2,
                  celltypes=['Neutrophil','Monocyte','Baso','Mast','Meg']):
    import pandas as pd
    ids = np.where(df_clones)
    df_tags = pd.DataFrame(data=ids[1],
                           index=ids[0],columns=['Tag_0'])
    print(df_tags.head())
    clones_selected = list()
    clones_truth = pd.DataFrame(columns=['celltype'])
    for x in np.sort(df_tags['Tag_0'].unique()):
        cells_x = df_tags['Tag_0'][df_tags['Tag_0']==x].index
        #the number of spanned timepoints for clone x        
        n_timepoints_x = len(df_metadata.iloc[cells_x, df_metadata.columns.get_loc('time_info')].unique()) 
        if(n_timepoints_x>cutoff_timepoints):
            #the ratio of cells falling into a specific cell type
            cells_x_selected = cells_x[df_metadata.iloc[cells_x, df_metadata.columns.get_loc('time_info')]==6]
            list_anno_x = df_metadata.iloc[cells_x_selected, df_metadata.columns.get_loc('state_info')].tolist()
            celltype = max(set(list_anno_x), key = list_anno_x.count)
            pct_celltype = np.float(list_anno_x.count(celltype))/len(list_anno_x)
            
            if((celltype in celltypes) and (pct_celltype==ratio)):
                clones_selected.append(x)
                clones_truth.loc[x,] = celltype
    return clones_selected, clones_truth


# %%
from scipy.sparse import csr_matrix
clones_selected, clones_truth = select_clones(adata.obs, adata.obsm['X_clone'].toarray(), 
                                              celltypes=['Neutrophil','Monocyte'])
adata_processed_filter = adata.copy()
adata_processed_filter.obsm['X_clone'] = adata.obsm['X_clone'].toarray()[:, clones_selected]

id_cells = np.where(adata_processed_filter.obsm['X_clone'].sum(axis=1)>0)[0]
adata_processed_filter = adata_processed_filter[id_cells, :]
adata_processed_filter_res = get_clone_trajectory(adata_processed_filter, clone_num=None)

# %%
adata_processed_filter_res.write("bifurcation_unipotent_cells.h5ad")


# %%
def plot_a_clone(adata, i, ax):      
    scv.pl.scatter(adata[adata.obs['clones']!=i, :], color='state_info', basis='emb',
                   alpha=0.6, s=15, ax=ax, show=False, legend_loc='right margin',
                   )
    scv.pl.scatter(adata[(adata.obs['clones']==i) & (~np.isnan(adata.obs['clones'])), :], 
                   color='time_info', basis='emb',
                   alpha=0.6, s=200, ax=ax, show=False, legend_loc='right margin', cmap='inferno',
                   )
    scv.pl.velocity_embedding(adata[adata.obs['clones']==i, :], color='black',linewidth=1,arrow_size=5,  
                              basis='emb', vkey='clone_vector',show=False,ax=ax, s=1, alpha=1,
                              title=f'clone {i}')


# %%
adata_processed_filter_res[adata_processed_filter_res.obs.clones==0].obs

# %%
ann2 = dict(zip(adata_processed_filter_res.obs.loc[:, 'state_info'].cat.categories, 
                adata_processed_filter_res.obs.loc[:, 'state_info'].cat.categories))
ann2['Undifferentiated'] = 'Undiff'
ann2['Ba'] = 'O'
ann2['Er'] = 'O'
ann2['Mk'] = 'O'
ann2['cDC'] = 'O'
ann2['pDC1'] = 'O'
ann2['pDC2'] = 'O'

# %%
ann2

# %%
adata_processed_filter_res.obs['state_info'] = adata_processed_filter_res.obs['state_info'].map(ann2)

# %%
adata_processed_filter_res.obs.loc[adata_processed_filter_res.obs.clones==257,:]

# %%
fig, ax = plt.subplots(1, 3)
fig.set_size_inches(15, 4)
plot_a_clone(adata_processed_filter_res, 3, ax[0])
plot_a_clone(adata_processed_filter_res, 3, ax[1])
scv.pl.velocity_embedding_grid(adata_processed_filter_res, scale=0.25, color='state_info', show=False,
                               s=20, density=0.5, arrow_size=2.5, linewidth=1,  basis='emb',
                               vkey='clone_vector',
                               ax=ax[2], title='Clonal progression', arrow_color='black')
#plot_a_clone(adata_processed, 1, )

# %%
from scipy.sparse import csr_matrix
clones_selected, clones_truth = select_clones(adata.obs, adata.obsm['X_clone'].toarray(), 
                                              celltypes=['Neutrophil'])
adata_processed_filter = adata.copy()
adata_processed_filter.obsm['X_clone'] = adata.obsm['X_clone'].toarray()[:, clones_selected]

id_cells = np.where(adata_processed_filter.obsm['X_clone'].sum(axis=1)>0)[0]
adata_processed_filter = adata_processed_filter[id_cells, :]
adata_processed_filter_res = get_clone_trajectory(adata_processed_filter, clone_num=None)

# %%
adata_processed_filter_res.shape

# %%
adata_processed_filter_res.write("neu_unipotent_cells.h5ad")

# %%
from scipy.sparse import csr_matrix
clones_selected, clones_truth = select_clones(adata.obs, adata.obsm['X_clone'].toarray(), 
                                              celltypes=['Monocyte'])
adata_processed_filter = adata.copy()
adata_processed_filter.obsm['X_clone'] = adata.obsm['X_clone'].toarray()[:, clones_selected]

id_cells = np.where(adata_processed_filter.obsm['X_clone'].sum(axis=1)>0)[0]
adata_processed_filter = adata_processed_filter[id_cells, :]
adata_processed_filter_res = get_clone_trajectory(adata_processed_filter, clone_num=None)

# %%
adata_processed_filter_res.write("mono_unipotent_cells.h5ad")

# %%
adata_processed_filter_res.shape

# %%
# !du -sh *h5ad

# %%
