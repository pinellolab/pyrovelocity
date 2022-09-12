=================
Pyro-Velocity
=================


.. image:: https://img.shields.io/pypi/v/pyrovelocity.svg
        :target: https://pypi.python.org/pypi/pyrovelocity

.. image:: https://img.shields.io/travis/qinqian/pyrovelocity.svg
        :target: https://travis-ci.com/qinqian/pyrovelocity

.. image:: https://readthedocs.org/projects/pyrovelocity/badge/?version=latest
        :target: https://pyrovelocity.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status

A multivariate RNA Velocity model to estimate the uncertainty of cell future state using Pyro

* Free software: Affero GPL V3


Features
--------

* Probabilistic modeling of RNA velocity
* Direct modeling of raw spliced and unspliced read count
* Multiple uncertainty diagnostics analysis
* Synchronized cell time estimation across genes
* Multivariate denoised gene expression and velocity prediction
* Quantification of RNA velocity performance with lineage tracing data

.. image:: docs/source/readme_figure1.png
  :width: 800
  :alt: Velocity workflow comparison


Installation with miniconda
---------------------------------

.. code-block:: bash
 
        mamba create -n pyrovelocity_release python==3.8.8
        conda activate pyrovelocity_release

        pip install pyro-ppl==1.6.0
        pip install scvelo
        pip install scvi-tools==0.13.0
        pip install pytorch_lightning==1.3.0
        pip install torchmetrics==0.5.1
        pip install h5py==3.2.1 anndata==0.7.5 # h5py problem for writing anndata
        pip install adjustText
        pip install astropy
        # install corresponding cuda version
        pip install torch==1.8.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html # this fix gpu memory issue, both 1.8.0/1.8.1 work
        pip install seaborn==0.11.2 # for avoiding seaborn api changes

        git clone https://github.com/pinellolab/pyrovelocity
        cd pyrovelocity && python setup.py install

Install additional packages
---------------------------

.. code-block:: bash

        pip install cospar==0.1.9 


Brief usage
---------------------------------

After installation, now let's look at your dataset to see how Pyro-Velocity can help understand cell dynamics.

If you only have raw FASTQ files from SMART-seq, 10X genomics or inDrop sequencing protocols. First, prepare your input using cellranger+velocyto or kallisto pipeline to generate the spliced and unspliced tables in h5ad file or loom file.

Then, following the steps to traing and model and utilize all the features of Pyro-Velocity, 

Step 1. open the jupyter notebook or ipython command line, load your data(*local_file.h5ad*) with scvelo by using 

.. code-block:: python

       import scvelo as scv
       adata = scv.read("local_file.h5ad")
       
Step 2. apply a minimal preprocessing on the *adata* object:

.. code-block:: python

       adata.layers['raw_spliced']   = adata.layers['spliced']
       adata.layers['raw_unspliced'] = adata.layers['unspliced']       
       adata.obs['u_lib_size_raw'] = adata.layers['raw_spliced'].toarray().sum(-1)
       adata.obs['s_lib_size_raw'] = adata.layers['raw_spliced'].toarray().sum(-1)       
       scv.pp.filter_and_normalize(adata, min_shared_counts=30, n_top_genes=2000)
       scv.pp.moments(adata, n_pcs=30, n_neighbors=30)     

Step 3. Train the Pyro-Velocity model:

.. code-block:: python

       from pyrovelocity.api import train_model
       # Model 1
       num_epochs = 1000 # large data
       num_epochs = 4000 # small data
       adata_model_pos = train_model(adata,
                                      max_epochs=num_epochs, svi_train=True, log_every=100,
                                      patient_init=45,
                                      batch_size=4000, use_gpu=0, cell_state='state_info',
                                      include_prior=True,
                                      offset=False,
                                      library_size=True,
                                      patient_improve=1e-3,
                                      model_type='auto',
                                      guide_type='auto_t0_constraint',
                                      train_size=1.0)       
              
       # Or Model 2
       adata_model_pos = train_model(adata,
                                      max_epochs=num_epochs, svi_train=True, log_every=100,
                                      patient_init=45,
                                      batch_size=4000, use_gpu=0, cell_state='state_info',
                                      include_prior=True,
                                      offset=True,
                                      library_size=True,
                                      patient_improve=1e-3,
                                      model_type='auto',
                                      guide_type='auto',
                                      train_size=1.0)  
       # adata_model_pos is a returned list in which 0th element is the trained model, 
       # the 1st element is the posterior samples of all random variables 
       save_res = True
       if save_res:
           adata_model_pos[0].save('saved_model', overwrite=True)
           result_dict = {"adata_model_pos": adata_model_pos[1], 
                          "v_map_all": v_map_all,
                          "embeds_radian": embeds_radian, "fdri": fdri, "embed_mean": embed_mean}
           import pickle
           with open("model_posterior_samples.pkl", "wb") as f:
                pickle.dump(result_dict, f)       

Step 4: apply minimal postprocessing using scvelo and evaluate Pyro-Velocity's velocity-based trajectory uncertainty. 

.. code-block:: python

    from pyrovelocity.plot import plot_state_uncertainty
    from pyrovelocity.plot import plot_posterior_time, plot_gene_ranking,\
          vector_field_uncertainty, plot_vector_field_uncertain,\
          plot_mean_vector_field, project_grid_points,rainbowplot,denoised_umap,\
          us_rainbowplot, plot_arrow_examples
      
    embedding = 'emb' # change to umap or tsne based on your embedding method

    # This generates the posterior samples of all vector fields
    # and statistical testing results from Rayleigh test
    v_map_all, embeds_radian, fdri = vector_field_uncertainty(adata, adata_model_pos[1], 
                                                              basis=embedding, denoised=False, n_jobs=30)
    fig, ax = plt.subplots()
    # This returns the posterior mean of the vector field
    embed_mean = plot_mean_vector_field(adata_model_pos[1], adata, ax=ax, n_jobs=30, basis=embedding)                                                              
    # This plot single-cell level vector field uncertainty 
    # and averaged cell vector field uncertainty on the grid points
    # based on angular standard deviation
    fig, ax = plt.subplots(1, 2)
    fig.set_size_inches(11.5, 5)    
    plot_vector_field_uncertain(adata, embed_mean, embeds_radian, 
                                ax=ax,
                                fig=fig, cbar=False, basis=embedding, scale=None)    
                                
    # This generates shared time uncertainty plot with contour lines
    fig, ax = plt.subplots(1, 3)
    fig.set_size_inches(12, 2.8)
    adata.obs['shared_time_uncertain'] = adata_model_pos[1]['cell_time'].std(0).flatten()
    ax_cb = scv.pl.scatter(adata, c='shared_time_uncertain', ax=ax[0], show=False, cmap='inferno', fontsize=7, s=20, colorbar=True, basis=embedding)
    select = adata.obs['shared_time_uncertain'] > np.quantile(adata.obs['shared_time_uncertain'], 0.9)
    sns.kdeplot(adata.obsm[f'X_{embedding}'][:, 0][select],
                adata.obsm[f'X_{embedding}'][:, 1][select],
                ax=ax[0], levels=3, fill=False)
                
    # This generates vector field uncertainty based on Rayleigh test.
    adata.obs.loc[:, 'vector_field_rayleigh_test'] = fdri
    im = ax[1].scatter(adata.obsm[f'X_{basis}'][:, 0],
                       adata.obsm[f'X_{basis}'][:, 1], s=3, alpha=0.9,
                       c=adata.obs['vector_field_rayleigh_test'], cmap='inferno_r',
                       linewidth=0)
    set_colorbar(im, ax[1], labelsize=5, fig=fig, position='right')
    select = adata.obs['vector_field_rayleigh_test'] > np.quantile(adata.obs['vector_field_rayleigh_test'], 0.95)
    sns.kdeplot(adata.obsm[f'X_{embedding}'][:, 0][select],
                adata.obsm[f'X_{embedding}'][:, 1][select], ax=ax[1], levels=3, fill=False)
    ax[1].axis('off')
    ax[1].set_title("vector field\nrayleigh test\nfdr<0.05: %s%%" % (round((fdri < 0.05).sum()/fdri.shape[0], 2)*100), fontsize=7)                                      
Step 5: Prioritization of cell fate markers based on negative mean absolute errors and pearson correlation between denoised spliced expression and posterior mean shared time, and then visualize top selected markers with rainbow plots

.. code-block:: python

    fig = plt.figure(figsize=(7.07, 4.5))
    subfig = fig.subfigures(1, 2, wspace=0.0, hspace=0, width_ratios=[1.6, 4])
    ax = fig.subplots(1)
    # This generates the selected cell fate markers and output in DataFrame        
    volcano_data, _ = plot_gene_ranking([adata_model_pos[1]], [adata], ax=ax,
                                         time_correlation_with='st', assemble=True)
    # This generates the rainbow plots for the selected markers.
    _ = rainbowplot(volcano_data, adata, adata_model_pos[1], 
                    subfig[1], data=['st', 'ut'], num_genes=4)
                    

Examples on real dataset
---------------------------------

Pyro-Velocity on the PBMC dataset[`1`_]
=========================================
The second example dataset is a single cell RNA-seq dataset of fully mature peripheral blood mononuclear cells (PBMC). This dataset was generated using the 10X genomics kit and contains 65,877 cells with 11 fully differentiated immune cell types.

Below we show the main output generated by Pyro-Velocity Model 1 analysis.

**Vector Field with uncertainty**

.. image:: docs/source/readme_figure2.png
  :width: 800
  :alt: PBMC vector field uncertainty

These 6 plots side by side from left to right are: 1. cell types, 2. stream plot of Pyro-velocity vector field based on posterior mean of 30 posterior samples, 3. single cell vector field examples showing all 30 posterior samples; 4. single cell vector field with uncertainty based on angular standard deviation across 30 posterior samples, 5. averaged vector field uncertainty from 4. 6. Rayleigh test of posterior samples vector field, the title shows the false discovery rate using threshold 5%.

The full example can be reproduced using the `PBMC`_ jupyter notebook. 

Pyro-Velocity on the Pancreas dataset[`2`_]
=============================================
Here we apply Pyro-Velocity to a single cell RNA-seq dataset of mouse pancreas in the E15.5 embryo developmental stage. This dataset was generated using the 10X genomics kit and contains 3,696 cells with 8 cell types including progenitor cells, intermediate and terminal cell states.

Below we show the main output generated by Pyro-Velocity Model 1 analysis.

**Vector Field with uncertainty**

.. image:: docs/source/readme_figure3.png
  :width: 800
  :alt: Pancreas vector field uncertainty

These 6 plots side by side from left to right are following the same order as above PBMC dataset analysis.

**Shared time with uncertainty**

.. image:: docs/source/readme_figure4.png
  :width: 400
  :alt: Pancreas shared time uncertainty

The left figure shows the average of 30 posterior samples from Pyro-Velocity shared time per cell, the right figure shows the standard deviation across posterior samples of shared time.


**Gene selection and visualization**

We use negative mean absolute error to select the top 300 genes with best fit of velocity model, then we order the gene set by pearson correlation between denoised spliced expression and posterior mean shared time.

.. image:: docs/source/readme_figure6.png
  :width: 400
  :alt: Pancreas Volcano plot for gene selection

Below we show the phase portraits, rainbow plots, and UMAP rendering of denoised splicing gene expression across cells for the top four selected genes.

.. image:: docs/source/readme_figure7.png
  :width: 800
  :alt: Pancreas vector field uncertainty

The full example can be reproduced using the `Pancreas`_ jupyter notebook. 


Pyro-Velocity on the Larry dataset[`3`_]
=========================================

The last dataset sampled differentiation over the time course of 2, 4, and 6 days and contained 49,302 cells that each could be traced with at least one barcode. Because LARRY couples single-cell transcriptomics and direct cell lineage tracing, this data set provides a unique opportunity to quantitatively benchmark RNA Velocity and to provide insights on the correct interpretation of recovered Velocity vector fields and latent cell times. 

Below we show the main output generated by Pyro-Velocity analysis.

**Vector Field with uncertainty**

.. image:: docs/source/readme_figure8.png
  :width: 800
  :alt: LARRY vector field uncertainty

These 5 plots side by side from left to right are following: 1) Cell types, 2) Clone progression vector field by using centroid of cells belonging to the same barcode for generating directed connection between consecutive physical times, 3) single cell vector field with uncertainty based on angular standard deviation across 30 posterior samples, 4. averaged vector field uncertainty from 3. 5. Rayleigh test of posterior samples vector field, the title shows the false discovery rate using threshold 5%.

**Shared time with uncertainty**

Latent time and on the side you show shared time uncertainty, and has a well-known trajectory pattern from progenitor cell towards mature endocrine cell types. 

.. image:: docs/source/readme_figure9.png
  :width: 800
  :alt: Pancreas shared time uncertainty

The full example can be reproduced using the `LARRY`_ jupyter notebook. 

.. _Notebook: https://github.com/pinellolab/pyrovelocity/tree/master/docs/source/notebooks
.. _PBMC: https://github.com/pinellolab/pyrovelocity/blob/master/docs/source/notebooks/pbmc.ipynb
.. _Pancreas: https://github.com/pinellolab/pyrovelocity/blob/master/docs/source/notebooks/pancreas.ipynb
.. _LARRY: https://github.com/pinellolab/pyrovelocity/blob/master/docs/source/notebooks/larry.ipynb
.. _1: https://scvelo.readthedocs.io/perspectives/Perspectives/ 
.. _2: https://scvelo.readthedocs.io/VelocityBasics/
.. _3: https://figshare.com/articles/dataset/larry_invitro_adata_sub_raw_h5ad/20780344 

