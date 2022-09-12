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

.. image:: docs/readme_figure1.png
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

Step 4: apply minimal postprocessing using scvelo and evaluate Pyro-Velocity's velocity-based trajectory uncertainty. 

.. code-block:: python

    from pyrovelocity.plot import plot_state_uncertainty
    from pyrovelocity.plot import plot_posterior_time, plot_gene_ranking,\
          vector_field_uncertainty, plot_vector_field_uncertain,\
          plot_mean_vector_field, project_grid_points,rainbowplot,denoised_umap,\
          us_rainbowplot, plot_arrow_examples
      
    embedding = 'emb' # change to umap or tsne based on your embedding method

    v_map_all, embeds_radian, fdri = vector_field_uncertainty(adata, adata_model_pos[1], 
                                                              basis=embedding, denoised=False, n_jobs=30)
    fig, ax = plt.subplots()
    embed_mean = plot_mean_vector_field(adata_model_pos[1], adata, ax=ax, n_jobs=30, basis=embedding)                                                              
    fig, ax = plt.subplots(1, 2)
    fig.set_size_inches(11.5, 5)    
    plot_vector_field_uncertain(adata, embed_mean, embeds_radian, 
                                ax=ax,
                                fig=fig, cbar=False, basis=embedding, scale=None)    
                                
    fig, ax = plt.subplots(1, 3)
    fig.set_size_inches(12, 2.8)
    adata.obs['shared_time_uncertain'] = adata_model_pos[1]['cell_time'].std(0).flatten()
    ax_cb = scv.pl.scatter(adata, c='shared_time_uncertain', ax=ax[0], show=False, cmap='inferno', fontsize=7, s=20, colorbar=True, basis=embedding)

    select = adata.obs['shared_time_uncertain'] > np.quantile(adata.obs['shared_time_uncertain'], 0.9)
    sns.kdeplot(adata.obsm[f'X_{embedding}'][:, 0][select],
                adata.obsm[f'X_{embedding}'][:, 1][select],
                ax=ax[0], levels=3, fill=False)

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
    _ = plot_state_uncertainty(adata_model_pos[1], adata, kde=True, data='raw',
                               top_percentile=0.9, ax=ax[2], basis=embedding)                                
       
       
Step 5: Prioritization of cell fate markers based on negative mean absolute errors and pearson correlation between denoised spliced expression and posterior mean shared time, and then visualize top selected markers with rainbow plots

.. code-block:: python

    fig, ax = plt.subplots()
    volcano_data, _ = plot_gene_ranking([adata_model_pos[1]], [adata], ax=ax, time_correlation_with='st')

    fig = plt.figure(figsize=(7.07, 4.5))
    subfig = fig.subfigures(1, 2, wspace=0.0, hspace=0, width_ratios=[1.6, 4])
    
    ax = subfig[0].subplots(2, 1)
    plot_posterior_time(adata_model_pos[1], adata, ax=ax[0], fig=subfig[0], addition=False)
    subfig[0].subplots_adjust(hspace=0.3, wspace=0.1, left=0.01, right=0.8, top=0.92, bottom=0.17)

    volcano_data2, _ = plot_gene_ranking([adata_model_pos[1]], [adata], ax=ax[1],
                                         time_correlation_with='st', assemble=True)
    _ = rainbowplot(volcano_data2, adata, adata_model_pos[1], subfig[1], data=['st', 'ut'], num_genes=4)
        

Examples on real dataset
---------------------------------
Please see all the examples in the `notebook`_  folders. 

Pyro-Velocity on the PBMC dataset[`1`_]
=========================================
Here we apply Pyro-Velocity to a single cell RNA-seq dataset of (Biological System) This dataset was generated using the 10X genomics kit and contains 3,696 cells with 8 different cell types


Below we show the main output generated by Pyro-Velocity


Vector Field with uncertainty


6 plots side by side  (1 row, 6 columns, cell types, stream plot, single cell vector field, single cell vector field with uncertainty, averaged vector field uncertainty, Rayleigh test)

Pyro-Velocity on the Pancreas dataset[`1`_]
==========================================
Descrition as we did before

Vector Field with uncertainty

Shared time with uncertainty

Umap, umao with uncertainty



**Gene selection and visualization**

First show volcano

Below show Fig 2e



Pyro-Velocity on the Larry dataset[`1`_]


Vector Field with uncertainty

Full dataset average vector field with uncertainty  and on the side yoiu show the Rayleigh's test

Shared time with uncertainty

Latent time and on the side you show shared time uncertaintny 






, and has a well-known trajectory pattern from progenitor cell towards mature endocrine cell types. 


The main output for analyzing the dataset from our package include:  Vector field with uncertainty, Shared latent time, Rayleigh test, Gene ranking only for pancreas with Rainbow plot or denoised expression-based trajectories. The `Pancreas`_ notebook is here for user to reproduce what we show in the manuscript. 




the title of the subsection can be Pyro-Velocity on X dataset
3. the section will have a very short description of the dataset, ref, technology, number of cells
4. below you will show the main output of our package: Vector field with uncertainty, Shared latent time, Rayleigh test, Gene ranking only for pancreas with Rainbow plot or any other output that the users can find particularly useful. Here is important that we don't show the entire figures we have in the manuscript but one panel for each row so the user can see the outputs one by one.
5. Finally add the link to the notebook so user can create the figures you are showing individually


.. _Notebook: https://github.com/pinellolab/pyrovelocity/tree/master/docs/source/notebooks
.. _Pancreas: https://github.com/pinellolab/pyrovelocity/blob/master/docs/source/notebooks/pancreas.ipynb
