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

If you only have raw FASTQ files from SMART-seq, 10X genomics or inDrop sequencing protocols. 

First, prepare your input using cellranger+velocyto or kallisto pipeline to generate the spliced and unspliced tables in h5ad file (*local_file.h5ad*) or loom file.

Second load your data with scvelo is a file processed by using 

.. code-block:: python

       import pyrovelocity
       adata = scv.read("local_file.h5ad")


Examples
---------------------------------
Please see the examples in the notebooks/ folders. 


Add in the tutorial/example section of the main readme, 3 subsections one for each dataset we analyzed

the title of the subsection can be Pyro-Velocity on X dataset
3. the section will have a very short description of the dataset, ref, technology, number of cells
4. below you will show the main output of our package: Vector field with uncertainty, Shared latent time, Rayleigh test, Gene ranking only for pancreas with Rainbow plot or any other output that the users can find particularly useful. Here is important that we don't show the entire figures we have in the manuscript but one panel for each row so the user can see the outputs one by one.
5. Finally add the link to the notebook so user can create the figures you are showing individually



