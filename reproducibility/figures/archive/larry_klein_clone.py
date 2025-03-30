#!/usr/bin/env python
# coding: utf-8

# In[1]:


from glob import glob
import numpy as np
from scipy.sparse import load_npz
from scipy.io import mmwrite, mmread
import scanpy as sc
import pandas as pd
import scvelo as scv
import anndata
from glob import glob
import re
import scipy


# In[2]:


all_data = glob('/data/pinello/PROJECTS/2019_11_ResidualVelocity/data/LARRY/klein_data/*')


# In[3]:


all_data


# In[24]:


counts_invivo = load_npz(all_data[-8])
genes_invivo = pd.read_table(all_data[-6],
                             header=None)
clone_invivo = load_npz(all_data[-11])
cell_meta_invivo = pd.read_table('/data/pinello/PROJECTS/2019_11_ResidualVelocity/data/LARRY/klein_data/cell_metadata_in_vivo.txt')
coordinate_invivo = pd.read_table('/data/pinello/PROJECTS/2019_11_ResidualVelocity/data/LARRY/klein_data/coordinates_in_vivo.txt', 
                                 header=None)

invivo = sc.AnnData(counts_invivo, obs=cell_meta_invivo, var=genes_invivo)
invivo_clone = sc.AnnData(clone_invivo, cell_meta_invivo)
# sc.pp.filter_cells(invivo, min_genes=200)
# sc.pp.filter_genes(invivo, min_cells=3)w
invivo_clone = invivo_clone[invivo.obs.index,]


# In[25]:


embedding = np.concatenate([coordinate_invivo.loc[:(coordinate_invivo.shape[0]//2-1), :].values, 
                            coordinate_invivo.loc[coordinate_invivo.shape[0]//2:, :].values], axis=1)
invivo.obsm['X_umap'] = embedding


# In[35]:


np.where(invivo_clone[:, c].X.toarray() == 1)


# In[37]:


# mito_genes = invivo.var_names.str.startswith('mt-')
# # for each cell compute fraction of counts in mito genes vs. all genes
# # the `.A1` is only necessary as X is sparse (to transform to a dense array after summing)
# invivo.obs['percent_mito'] = np.sum(
#     invivo[:, mito_genes].X, axis=1).A1 / np.sum(invivo.X, axis=1).A1
# # add the total counts per cell as observations-annotation to adata
# invivo.obs['n_counts'] = invivo.X.sum(axis=1).A1
# sc.pl.violin(invivo, ['n_genes', 'n_counts', 'percent_mito'],
#              jitter=0.4, multi_panel=True)

# invivo = invivo[invivo.obs.n_genes < 5000, :]
# invivo = invivo[invivo.obs.percent_mito < 0.05, :]
# invivo_clone = invivo_clone[invivo.obs.index,]
# sc.pp.normalize_total(invivo, target_sum=1e4)
# sc.pp.log1p(invivo)
# invivo.raw = invivo
# sc.pp.highly_variable_genes(invivo, min_mean=0.0125, max_mean=3, min_disp=0.5)
# invivo = invivo[:, invivo.var.highly_variable]
# # sc.pp.regress_out(invivo, ['n_counts'])
# # sc.pp.scale(invivo, max_value=10)
# sc.tl.pca(invivo, svd_solver='arpack')
# sc.pp.neighbors(invivo, n_neighbors=10, n_pcs=40)
# sc.tl.umap(invivo)

clone_invivodict = {}
for c in range(invivo_clone.shape[1]):
    # for each clone number
#    print(c, invivo_clone[:, 1].X.toarray())
    cells_clone, = np.where(invivo_clone[:, c].X.toarray()[:, 0] == 1)    
    if len(cells_clone) > 0:
        clone_invivodict[c] = invivo_clone.obs.index[cells_clone]


# In[38]:


invivo.obs.loc[:, 'clone_demo'] = 0
invivo.obs.loc[clone_invivodict[55], 'clone_demo'] = 1
invivo.obs.loc[:, 'Time point'] = invivo.obs.loc[:, 'Time point'].astype('int').astype('category')
invivo.obs.loc[:, 'clone_demo'] = invivo.obs.loc[:, 'clone_demo'].astype('category')
invivo.obs.loc[:, 'Time point'].cat.categories
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2)
fig.set_size_inches((24, 12))
sc.pl.umap(invivo,  color='Annotation', ax=ax[0],show=False, 
           legend_loc='on data', title='', frameon=False)
sc.pl.umap(invivo,  color='Time point', ax=ax[1], show=False, wspace=0.2, legend_fontsize=32,
           palette=['tab:orange', 'tab:red', 'tab:blue'],         
           legend_loc='right', title='', frameon=False)
sc.pl.umap(invivo[invivo.obs.clone_demo==1,:],  color='Time point', ax=ax[1], show=False, size=120, edges_width=0.5,
           legend_fontsize=32,
           palette=['tab:orange', 'tab:red', 'tab:blue'],
           legend_loc='on data', title='', frameon=False)
# for s, e in list(zip(invitro.obs.loc[:, 'Time point'].cat.categories[:-1], sc.plotting._utils._tmp_cluster_pos[1:])):
for s, e in zip(sc.plotting._utils._tmp_cluster_pos[:-1], sc.plotting._utils._tmp_cluster_pos[1:]):
    print(s, e)
    ax[1].arrow(s[0], s[1], (e[0]-s[0])*0.8, (e[1]-s[1])*0.8, head_width=50, head_length=60, fc='k', ec='k')


# In[14]:


invivo.obs.loc[:, 'clone_demo'] = 0
invivo.obs.loc[clone_invivodict[640], 'clone_demo'] = 1
invivo.obs.loc[:, 'Time point'] = invivo.obs.loc[:, 'Time point'].astype('int').astype('category')
invivo.obs.loc[:, 'clone_demo'] = invivo.obs.loc[:, 'clone_demo'].astype('category')
invivo.obs.loc[:, 'Time point'].cat.categories
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2)
fig.set_size_inches((24, 12))
sc.pl.umap(invivo,  color='Annotation', ax=ax[0],show=False, 
           legend_loc='on data', title='', frameon=False)
sc.pl.umap(invivo,  color='Time point', ax=ax[1], show=False, wspace=0.2, legend_fontsize=32,
           palette=['tab:orange', 'tab:red', 'tab:blue'],         
           legend_loc='right', title='', frameon=False)
sc.pl.umap(invivo[invivo.obs.clone_demo==1,:],  color='Time point', ax=ax[1], show=False, size=120, edges_width=0.5,
           legend_fontsize=32,
           palette=['tab:orange', 'tab:red', 'tab:blue'],
           legend_loc='on data', title='', frameon=False)

# for s, e in list(zip(invitro.obs.loc[:, 'Time point'].cat.categories[:-1], sc.plotting._utils._tmp_cluster_pos[1:])):
for s, e in zip(sc.plotting._utils._tmp_cluster_pos[:-1], sc.plotting._utils._tmp_cluster_pos[1:]):
    print(s, e)
    ax[1].arrow(s[0], s[1], (e[0]-s[0])*0.8, (e[1]-s[1])*0.8, head_width=50, head_length=60, fc='k', ec='k')


# In[15]:


fig


# In[36]:


counts_invitro = load_npz(all_data[6])
genes_invitro = pd.read_table(all_data[-2],
                            header=None)
clone_invitro = load_npz(all_data[2])
cell_meta_invitro = pd.read_table('/data/pinello/PROJECTS/2019_11_ResidualVelocity/data/LARRY/klein_data/cell_metadata_in_vitro.txt')
coordinate_invitro = pd.read_table('/data/pinello/PROJECTS/2019_11_ResidualVelocity/data/LARRY/klein_data/coordinates_in_vitro.txt', 
                                 header=None)
invitro = sc.AnnData(counts_invitro, obs=cell_meta_invitro, var=genes_invitro)
invitro_clone = sc.AnnData(clone_invitro, cell_meta_invitro)
# sc.pp.filter_cells(invitro, min_genes=200)
# sc.pp.filter_genes(invitro, min_cells=3)
invitro_clone = invitro_clone[invitro.obs.index,]
embedding = np.concatenate([coordinate_invitro.loc[:(coordinate_invitro.shape[0]//2-1), :].values, 
                            coordinate_invitro.loc[coordinate_invitro.shape[0]//2:, :].values], axis=1)
invitro.obsm['X_umap'] = embedding


# In[41]:


get_ipython().system('head /data/pinello/PROJECTS/2019_11_ResidualVelocity/data/LARRY/klein_data/coordinates_in_vitro.txt')


# In[290]:


# mito_genes = invitro.var_names.str.startswith('mt-')
# # for each cell compute fraction of counts in mito genes vs. all genes
# # the `.A1` is only necessary as X is sparse (to transform to a dense array after summing)
# invitro.obs['percent_mito'] = np.sum(
#     invitro[:, mito_genes].X, axis=1).A1 / np.sum(invitro.X, axis=1).A1
# # add the total counts per cell as observations-annotation to adata
# invitro.obs['n_counts'] = invitro.X.sum(axis=1).A1
# sc.pl.violin(invitro, ['n_genes', 'n_counts', 'percent_mito'],
#              jitter=0.4, multi_panel=True)
# invitro = invitro[invitro.obs.n_genes < 5000, :]
# invitro = invitro[invitro.obs.percent_mito < 0.05, :]
# invitro_clone = invitro_clone[invitro.obs.index,]
# ## sc.pp.normalize_total(invitro, target_sum=1e4) ## weird errors 
# sc.pp.log1p(invitro)
# invitro.raw = invitro
# sc.pp.highly_variable_genes(invitro, min_mean=0.0125, max_mean=3, min_disp=0.5)
# invitro = invitro[:, invitro.var.highly_variable]
# # sc.pp.regress_out(invivo, ['n_counts'])
# # sc.pp.scale(invivo, max_value=10)
# sc.tl.pca(invitro, svd_solver='arpack')
# sc.pp.neighbors(invitro, n_neighbors=50, n_pcs=50)
# sc.tl.umap(invitro)


# In[17]:


clone_dict = {}
for c in range(invitro_clone.shape[1]):
    # for each clone number
    cells_clone, = np.where(invitro_clone[:, c].X == 1)
    if len(cells_clone) > 0:
        clone_dict[c] = invitro_clone.obs.index[cells_clone]


# In[19]:


invitro_clone.obs.head()


# In[20]:


invitro.obs.loc[:, 'clone_demo'] = 0
invitro.obs.loc[clone_dict[922], 'clone_demo'] = 1
invitro.obs.loc[:, 'Time point'] = invitro.obs.loc[:, 'Time point'].astype('int').astype('category')
invitro.obs.loc[:, 'clone_demo'] = invitro.obs.loc[:, 'clone_demo'].astype('category')
invitro.obs.loc[:, 'Well'] = invitro.obs.loc[:, 'Well'].astype('category')
invitro.obs.loc[:, 'Time point'].cat.categories
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3)
fig.set_size_inches((30, 12))
sc.pl.umap(invitro,  color=['Annotation'], ax=ax[0], show=False,          
           legend_fontsize=32,
           legend_loc='on data', title='', frameon=False)
sc.pl.umap(invitro,  color=['Well'], ax=ax[1], show=False, 
           palette=['tab:orange', 'tab:purple', 'tab:blue'],      
           legend_fontsize=32,
           legend_loc='on data', title='', frameon=False)
sc.pl.umap(invitro,  color='Time point', ax=ax[2], show=False, wspace=0.2, legend_fontsize=32,
           palette=['tab:orange', 'tab:red', 'tab:blue'],   
           legend_loc='right', title='', frameon=False)
sc.pl.umap(invitro[invitro.obs.clone_demo==1,:],  color='Time point', ax=ax[2], show=False, size=120, edges_width=0.5,
           legend_fontsize=32,
           palette=['tab:orange', 'tab:red', 'tab:blue'],
           legend_loc='on data', title='', frameon=False)

# for s, e in list(zip(invitro.obs.loc[:, 'Time point'].cat.categories[:-1], sc.plotting._utils._tmp_cluster_pos[1:])):
for s, e in zip(sc.plotting._utils._tmp_cluster_pos[:-1], sc.plotting._utils._tmp_cluster_pos[1:]):
    print(s, e)
    ax[2].arrow(s[0], s[1], (e[0]-s[0])*0.8, (e[1]-s[1])*0.8, head_width=50, head_length=50, fc='k', ec='k')


# In[21]:


fig


# In[23]:


invitro.obs.loc[:, 'clone_demo'] = 0
invitro.obs.loc[clone_dict[4300], 'clone_demo'] = 1
invitro.obs.loc[:, 'Time point'] = invitro.obs.loc[:, 'Time point'].astype('int').astype('category')
invitro.obs.loc[:, 'clone_demo'] = invitro.obs.loc[:, 'clone_demo'].astype('category')
invitro.obs.loc[:, 'Time point'].cat.categories
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3)
fig.set_size_inches((24, 12))
sc.pl.umap(invitro,  color='Annotation', ax=ax[0],show=False, 
           legend_loc='on data', title='', frameon=False)
sc.pl.umap(invitro,  color='Time point', ax=ax[1], show=False, wspace=0.2, legend_fontsize=32,
           palette=['tab:orange', 'tab:red', 'tab:blue'],         
           legend_loc='right', title='', frameon=False)
sc.pl.umap(invitro[invitro.obs.clone_demo==1,:],  color='Time point', ax=ax[2], show=False, size=120, edges_width=0.5,
           legend_fontsize=32,
           palette=['tab:orange', 'tab:red', 'tab:blue'],
           legend_loc='on data', title='', frameon=False)

# for s, e in list(zip(invitro.obs.loc[:, 'Time point'].cat.categories[:-1], sc.plotting._utils._tmp_cluster_pos[1:])):
for s, e in zip(sc.plotting._utils._tmp_cluster_pos[:-1], sc.plotting._utils._tmp_cluster_pos[1:]):
    print(s, e)
    ax[2].arrow(s[0], s[1], (e[0]-s[0])*0.8, (e[1]-s[1])*0.8, head_width=50, head_length=50, fc='k', ec='k')


# In[24]:


fig


# In[25]:


invitro.obs.loc[:, 'clone_demo'] = 0
invitro.obs.loc[clone_dict[846], 'clone_demo'] = 1
invitro.obs.loc[:, 'Time point'] = invitro.obs.loc[:, 'Time point'].astype('int').astype('category')
invitro.obs.loc[:, 'clone_demo'] = invitro.obs.loc[:, 'clone_demo'].astype('category')
invitro.obs.loc[:, 'Time point'].cat.categories
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3)
fig.set_size_inches((24, 12))
sc.pl.umap(invitro,  color='Annotation', ax=ax[0],show=False, 
           legend_loc='on data', title='', frameon=False)
sc.pl.umap(invitro,  color='Time point', ax=ax[1], show=False, wspace=0.2, legend_fontsize=32,
           palette=['tab:orange', 'tab:red', 'tab:blue'],         
           legend_loc='right', title='', frameon=False)
sc.pl.umap(invitro[invitro.obs.clone_demo==1,:],  color='Time point', ax=ax[1], show=False, size=120, edges_width=0.5,
           legend_fontsize=32,
           palette=['tab:orange', 'tab:red', 'tab:blue'],
           legend_loc='on data', title='', frameon=False)

# for s, e in list(zip(invitro.obs.loc[:, 'Time point'].cat.categories[:-1], sc.plotting._utils._tmp_cluster_pos[1:])):
for s, e in zip(sc.plotting._utils._tmp_cluster_pos[:-1], sc.plotting._utils._tmp_cluster_pos[1:]):
    print(s, e)
    ax[1].arrow(s[0], s[1], (e[0]-s[0])*0.8, (e[1]-s[1])*0.8, head_width=50, head_length=50, fc='k', ec='k')


# In[27]:


invitro.obs.loc[:, 'clone_demo'] = 0
invitro.obs.loc[clone_dict[851], 'clone_demo'] = 1
invitro.obs.loc[:, 'Time point'] = invitro.obs.loc[:, 'Time point'].astype('int').astype('category')
invitro.obs.loc[:, 'clone_demo'] = invitro.obs.loc[:, 'clone_demo'].astype('category')
invitro.obs.loc[:, 'Time point'].cat.categories
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2)
fig.set_size_inches((24, 12))
sc.pl.umap(invitro,  color='Annotation', ax=ax[0],show=False, 
           legend_loc='on data', title='', frameon=False)
sc.pl.umap(invitro,  color='Time point', ax=ax[1], show=False, wspace=0.2, legend_fontsize=32,
           palette=['tab:orange', 'tab:red', 'tab:blue'],         
           legend_loc='right', title='', frameon=False)
sc.pl.umap(invitro[invitro.obs.clone_demo==1,:],  color='Time point', ax=ax[1], show=False, size=120, edges_width=0.5,
           legend_fontsize=32,
           palette=['tab:orange', 'tab:red', 'tab:blue'],
           legend_loc='on data', title='', frameon=False)

# for s, e in list(zip(invitro.obs.loc[:, 'Time point'].cat.categories[:-1], sc.plotting._utils._tmp_cluster_pos[1:])):
for s, e in zip(sc.plotting._utils._tmp_cluster_pos[:-1], sc.plotting._utils._tmp_cluster_pos[1:]):
    print(s, e)
    ax[1].arrow(s[0], s[1], (e[0]-s[0])*0.8, (e[1]-s[1])*0.8, head_width=50, head_length=50, fc='k', ec='k')


# In[28]:


fig


# In[29]:


invitro.obs.loc[:, 'clone_demo'] = 0
invitro.obs.loc[clone_dict[408], 'clone_demo'] = 1
invitro.obs.loc[:, 'Time point'] = invitro.obs.loc[:, 'Time point'].astype('int').astype('category')
invitro.obs.loc[:, 'clone_demo'] = invitro.obs.loc[:, 'clone_demo'].astype('category')
invitro.obs.loc[:, 'Time point'].cat.categories
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2)
fig.set_size_inches((24, 12))
sc.pl.umap(invitro,  color='Annotation', ax=ax[0],show=False, 
           legend_loc='on data', title='', frameon=False)
sc.pl.umap(invitro,  color='Time point', ax=ax[1], show=False, wspace=0.2, legend_fontsize=32,
           palette=['tab:orange', 'tab:red', 'tab:blue'],         
           legend_loc='right', title='', frameon=False)
sc.pl.umap(invitro[invitro.obs.clone_demo==1,:],  color='Time point', ax=ax[1], show=False, size=120, edges_width=0.5,
           legend_fontsize=32,
           palette=['tab:orange', 'tab:red', 'tab:blue'],
           legend_loc='on data', title='', frameon=False)

# for s, e in list(zip(invitro.obs.loc[:, 'Time point'].cat.categories[:-1], sc.plotting._utils._tmp_cluster_pos[1:])):
for s, e in zip(sc.plotting._utils._tmp_cluster_pos[:-1], sc.plotting._utils._tmp_cluster_pos[1:]):
    print(s, e)
    ax[1].arrow(s[0], s[1], (e[0]-s[0])*0.8, (e[1]-s[1])*0.8, head_width=50, head_length=50, fc='k', ec='k')


# In[30]:


fig


# # mapping LARRY loom with the clone information

# In[2]:


larry_invitros = []
for i in glob('../../data/LARRY/*loom'):
    larry_invitros.append(scv.read(i))
larry_invitro_adata = larry_invitros[0].concatenate(*larry_invitros[1:])


# In[3]:


n = 0
for i in larry_invitros:
    n += i.shape[0]
print(n)


# In[41]:


metadata = pd.read_table('../../data/LARRY/GSM4185642_stateFate_inVitro_metadata.txt')


# In[42]:


clone_invitro = scipy.io.mmread('../../data/LARRY/GSM4185642_stateFate_inVitro_clone_matrix.mtx')


# In[43]:


clone_invitro = clone_invitro.tocsr()


# In[44]:


metadata.shape, clone_invitro.shape


# In[45]:


cells = pd.read_table('../../data/LARRY/filtered_cell_barcodes.txt', header=None)


# In[46]:


librarys = pd.read_table('../../data/LARRY/filtered_library_names.txt', header=None)


# In[47]:


librarys[:5]


# In[48]:


genes_invitro = pd.read_table('../../data/LARRY/GSM4185642_stateFate_inVitro_gene_names.txt',
                              header=None)


# In[49]:


genes_invitro.columns = ['gene']
genes_invitro.index = genes_invitro.gene


# In[50]:


cell_meta_invitro = pd.read_table('/data/pinello/PROJECTS/2019_11_ResidualVelocity/data/LARRY/GSM4185642_stateFate_inVitro_cell_barcodes.txt',
                                  header=None)


# In[51]:


cell_meta_invitro.columns = ['barcode']


# In[52]:


cell_meta_invitro.index = cell_meta_invitro.barcode


# In[53]:


cell_meta_invitro.head()


# In[54]:


filter_barcodes = pd.concat([librarys,
                             cells.iloc[:, 0].map(lambda x: x.replace('-', ''))], axis=1)


# In[55]:


filter_barcodes.head()


# In[56]:


filter_barcodes = filter_barcodes.apply(lambda x: ':'.join(x), axis=1).values


# In[57]:


larry_invitro_adata.obs.head()


# In[58]:


filter_barcodes[:5]


# In[59]:


metadata.head()


# In[60]:


np.intersect1d(larry_invitro_adata.obs.index.map(lambda x:re.sub('x-.*', '', x)), 
               filter_barcodes).shape


# In[61]:


larry_invitro_adata_sub = larry_invitro_adata[larry_invitro_adata.obs.index.map(lambda x:re.sub('x-.*',
                                                                                                '', x)).isin(filter_barcodes)]


# In[62]:


larry_invitro_adata_sub.obs.head()


# In[63]:


def match(x, y):
    """return y index that match the x iterms"""
    ind_dict = {}
    for i, j in enumerate(y):
        ind_dict[j] = i
    inds = []
    for i in x:
        inds.append(ind_dict[i])
    return np.array(inds)


# In[64]:


(metadata.loc[:, ['Library', 'Cell barcode']].apply(lambda x: '%s:%s' %(x[0], x[1].replace('-', '')), axis=1)==filter_barcodes).sum()


# In[67]:


larry_invitro_adata_sub.obs.head()


# In[70]:


larry_invitro_adata_sub.obs.index.map(lambda x:re.sub('x-.*', '', x))


# In[68]:


filter_barcodes[:10]


# In[71]:


larry_invitro_adata_sub = larry_invitro_adata_sub[match(filter_barcodes, 
                              larry_invitro_adata_sub.obs.index.map(lambda x:re.sub('x-.*', '', x))), :]


# In[72]:


larry_invitro_adata_sub.obs = metadata
larry_invitro_adata_sub.obs.index = filter_barcodes


# In[74]:


metadata.head()


# In[75]:


larry_invitro_adata_sub.obs.head()


# In[76]:


invitro_clone = sc.AnnData(clone_invitro, obs=metadata)


# In[78]:


invitro_clone.obs.index = filter_barcodes


# In[380]:


larry_invitro_adata.write('all_invitro_loom.h5ad')


# In[ ]:


larry_invitro_adata_sub.write('sub_invitro_loom.h5ad')


# In[ ]:


invitro_clone.write('all_invitro_loom_clone.h5ad')


# In[385]:


invitro_clone.shape


# In[403]:


from resvel import preprocess
from resvel.stat import run_velocity
import scanpy as sc


# In[394]:


larry_invitro_adata_sub.obsm['X_spring'] = larry_invitro_adata_sub.obs.loc[:, ['SPRING-x', 'SPRING-y']] .values
larry_invitro_adata_sub.obsm['spring'] = larry_invitro_adata_sub.obs.loc[:, ['SPRING-x', 'SPRING-y']] .values
larry_invitro_adata_sub.obsm['X_umap'] = larry_invitro_adata_sub.obs.loc[:, ['SPRING-x', 'SPRING-y']] .values


# In[402]:


larry_invitro_adata_sub.obs.head()


# In[400]:


fig, ax = plt.subplots()
sc.pl.embedding(larry_invitro_adata_sub, basis='umap', show=False, ax=ax)
fig


# In[401]:


larry_invitro_adata_sub = preprocess(larry_invitro_adata_sub)


# In[404]:


larry_invitro_adata_sub = run_velocity(larry_invitro_adata_sub, mode='dynamical')


# In[405]:


#larry_invitro_adata_sub.write('sub_invitro_loom_dynamical_velocity.h5ad')


# In[406]:


get_ipython().system('du -sh sub_invitro_loom_dynamical_velocity.h5ad')


# In[414]:


fig, ax = plt.subplots(2, 1)
fig.set_size_inches(10, 22)
scv.pl.velocity_embedding_stream(larry_invitro_adata_sub, color='Cell type annotation', 
                               ax=ax[0],  show=False) # scale=0.5, arrow_size=2.5
scv.pl.scatter(larry_invitro_adata_sub, color='latent_time', show=False,
               ax=ax[1])
plt.tight_layout(pad=0)
fig


# In[415]:


fig.savefig('larry_invitro.png')


# In[416]:


get_ipython().system('ls larry_invitro.png')


# In[ ]:




