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
* Documentation: https://pyrovelocity.readthedocs.io.


Features
--------

* Probabilistic modeling of RNA velocity
* Direct modeling of raw spliced and unspliced read count
* Multiple uncertainty diagnostics analysis
* Synchronized cell time estimation across genes
* Multivariate denoised gene expression and velocity prediction
* Quantification of RNA velocity performance with lineage tracing data

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

Additional packages
---------------------------

.. code-block:: bash

        pip install cospar==0.1.9 

Usage
---------------------------------

Please see the examples in the notebooks/ folders. 

