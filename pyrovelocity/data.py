import os
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import anndata
import numpy as np
import scvelo as scv
from scanpy import read
import scvi
import torch
from anndata import AnnData
from scipy import sparse as sp_sparse
from scvi.data import register_tensor_from_anndata
from scvi.data._anndata import _register_anndata
from scvi.data._anndata import _setup_batch
from scvi.data._anndata import _setup_extra_categorical_covs
from scvi.data._anndata import _setup_extra_continuous_covs
from scvi.data._anndata import _setup_labels
from scvi.data._anndata import _setup_protein_expression
from scvi.data._anndata import _setup_summary_stats
from scvi.data._anndata import _setup_x
from scvi.data._anndata import _verify_and_correct_data_format
from scvi.data._anndata import logger
from scvi.dataloaders import DataSplitter

from pyrovelocity.cytotrace import cytotrace_sparse


def load_data(
    data="pancrease",
    top_n=2000,
    min_shared_counts=30,
    eps=1e-6,
    force=False,
):
    if force or (
        not os.path.exists(f"{data}_scvelo_fitted_{top_n}_{min_shared_counts}.h5ad")
    ):
        if data == "pancrease":
            adata = scv.datasets.pancreas()
        elif data == "forebrain":
            adata = scv.datasets.forebrain()
        elif data == "dentategyrus_lamanno":
            adata = scv.datasets.dentategyrus_lamanno()
        else:
            adata = scv.datasets.dentategyrus()
        adata.layers["raw_unspliced"] = adata.layers["unspliced"]
        adata.layers["raw_spliced"] = adata.layers["spliced"]
        adata.obs["u_lib_size_raw"] = adata.layers["raw_unspliced"].toarray().sum(-1)
        adata.obs["s_lib_size_raw"] = adata.layers["raw_spliced"].toarray().sum(-1)
        cytotrace_sparse(adata, layer="spliced")
        scv.pp.filter_and_normalize(
            adata, min_shared_counts=min_shared_counts, n_top_genes=top_n
        )
        scv.pp.moments(adata, n_pcs=30, n_neighbors=30)
        scv.tl.recover_dynamics(adata, n_jobs=20, use_raw=False)
        scv.tl.velocity(adata, mode="dynamical", use_raw=False)
        scv.tl.velocity_graph(adata)
        scv.tl.velocity_embedding(adata)
        adata.write(f"{data}_scvelo_fitted_{top_n}_{min_shared_counts}.h5ad")
    else:
        adata = scv.read(f"{data}_scvelo_fitted_{top_n}_{min_shared_counts}.h5ad")
    scv.tl.latent_time(adata)
    return adata


def load_larry(file_path='data/larry.h5ad'):
    """In vitro Hemotopoiesis Larry datasets

    Data from `CALEB WEINREB et al. (2020) <DOI: 10.1126/science.aaw3381>'
    https://figshare.com/ndownloader/articles/20780344/versions/1

    Returns
    -------
    Returns `adata` object
    """
    #url = "https://ndownloader.figshare.com/files/20780344"
    url = "https://figshare.com/ndownloader/files/37028569"
    adata = read(file_path, backup_url=url, sparse=True, cache=True)
    return adata


def setup_anndata_multilayers(
    adata: anndata.AnnData,
    batch_key: Optional[str] = None,
    labels_key: Optional[str] = None,
    layer: Optional[str] = None,
    protein_expression_obsm_key: Optional[str] = None,
    protein_names_uns_key: Optional[str] = None,
    categorical_covariate_keys: Optional[List[str]] = None,
    continuous_covariate_keys: Optional[List[str]] = None,
    copy: bool = False,
    input_type: str = "knn",
    n_aux_cells: int = 10,
    cluster: str = "clusters",
) -> Optional[anndata.AnnData]:
    """
    Adapt from setup_anndata with extension to multiple layers of data

    Sets up :class:`~anndata.AnnData` object for `scvi` models.

    A mapping will be created between data fields used by `scvi` to their respective locations in adata.
    This method will also compute the log mean and log variance per batch for the library size prior.

    None of the data in adata are modified. Only adds fields to adata.

    Parameters
    ----------
    adata
        AnnData object containing raw counts. Rows represent cells, columns represent features.
    batch_key
        key in `adata.obs` for batch information. Categories will automatically be converted into integer
        categories and saved to `adata.obs['_scvi_batch']`. If `None`, assigns the same batch to all the data.
    labels_key
        key in `adata.obs` for label information. Categories will automatically be converted into integer
        categories and saved to `adata.obs['_scvi_labels']`. If `None`, assigns the same label to all the data.
    layer
        if not `None`, uses this as the key in `adata.layers` for raw count data.
    protein_expression_obsm_key
        key in `adata.obsm` for protein expression data, Required for :class:`~scvi.model.TOTALVI`.
    protein_names_uns_key
        key in `adata.uns` for protein names. If None, will use the column names of `adata.obsm[protein_expression_obsm_key]`
        if it is a DataFrame, else will assign sequential names to proteins. Only relevant but not required for :class:`~scvi.model.TOTALVI`.
    categorical_covariate_keys
        keys in `adata.obs` that correspond to categorical data. Used in some `scvi` models.
    continuous_covariate_keys
        keys in `adata.obs` that correspond to continuous data. Used in some `scvi` models.
    copy
        if `True`, a copy of adata is returned.

    Returns
    -------
    If ``copy``,  will return :class:`~anndata.AnnData`.
    Adds the following fields to adata:

    .uns['_scvi']
        `scvi` setup dictionary
    .obs['_scvi_labels']
        labels encoded as integers
    .obs['_scvi_batch']
        batch encoded as integers

    Examples
    --------
    Example setting up a scanpy dataset with random gene data and no batch nor label information

    >>> import scanpy as sc
    >>> import scvi
    >>> import numpy as np
    >>> adata = scvi.data.synthetic_iid(run_setup_anndata=False)
    >>> adata
    AnnData object with n_obs × n_vars = 400 × 100
        obs: 'batch', 'labels'
        uns: 'protein_names'
        obsm: 'protein_expression'

    Filter cells and run preprocessing before `setup_anndata`

    >>> sc.pp.filter_cells(adata, min_counts = 0)

    Since no batch_key nor labels_key was passed, setup_anndata() will assume all cells have the same batch and label

    >>> scvi.data.setup_anndata(adata)
    INFO      No batch_key inputted, assuming all cells are same batch
    INFO      No label_key inputted, assuming all cells have same label
    INFO      Using data from adata.X
    INFO      Computing library size prior per batch
    INFO      Registered keys:['X', 'batch_indices', 'local_l_mean', 'local_l_var', 'labels']
    INFO      Successfully registered anndata object containing 400 cells, 100 vars, 1 batches, 1 labels, and 0 proteins. Also registered 0 extra categorical covariates and 0 extra continuous covariates.

    Example setting up scanpy dataset with random gene data, batch, and protein expression

    >>> adata = scvi.data.synthetic_iid(run_setup_anndata=False)
    >>> scvi.data.setup_anndata(adata, batch_key='batch', protein_expression_obsm_key='protein_expression')
    INFO      Using batches from adata.obs["batch"]
    INFO      No label_key inputted, assuming all cells have same label
    INFO      Using data from adata.X
    INFO      Computing library size prior per batch
    INFO      Using protein expression from adata.obsm['protein_expression']
    INFO      Generating sequential protein names
    INFO      Registered keys:['X', 'batch_indices', 'local_l_mean', 'local_l_var', 'labels', 'protein_expression']
    INFO      Successfully registered anndata object containing 400 cells, 100 vars, 2 batches, 1 labels, and 100 proteins. Also registered 0 extra categorical covariates and 0 extra continuous covariates.

    Example setting up scanpy adata with spliced and unspliced layers of data
    >>> adata = load_data()
    >>> setup_anndata_multilayers(adata)

    """
    if copy:
        adata = adata.copy()

    if adata.is_view:
        raise ValueError(
            "Please run `adata = adata.copy()` or use the copy option in this function."
        )

    adata.uns["_scvi"] = {}
    adata.uns["_scvi"]["scvi_version"] = scvi.__version__

    batch_key = _setup_batch(adata, batch_key)
    labels_key = _setup_labels(adata, labels_key)
    assert len(layer) >= 2
    u_loc, u_key = _setup_x(adata, layer[0])
    s_loc, s_key = _setup_x(adata, layer[1])

    data_registry = {
        # VelocityCONSTANTS.X_KEY: {"attr_name": s_loc, "attr_key": s_key},
        # VelocityCONSTANTS.U_KEY: {"attr_name": u_loc, "attr_key": u_key},
        # VelocityCONSTANTS.BATCH_KEY: {"attr_name": "obs", "attr_key": batch_key},
        # VelocityCONSTANTS.LABELS_KEY: {"attr_name": "obs", "attr_key": labels_key},
        "X": {"attr_name": s_loc, "attr_key": s_key},
        "U": {"attr_name": u_loc, "attr_key": u_key},
        "batch": {"attr_name": "obs", "attr_key": batch_key},
        "label": {"attr_name": "obs", "attr_key": labels_key},
    }

    # training mask data
    # if 'training_mask' in adata.layers:
    #    m_loc, m_key = _setup_x(adata, layer[2])
    #    data_registry[VelocityCONSTANTS.MASK_KEY] = {"attr_name": m_loc, "attr_key": m_key}

    if protein_expression_obsm_key is not None:
        protein_expression_obsm_key = _setup_protein_expression(
            adata, protein_expression_obsm_key, protein_names_uns_key, batch_key
        )
        data_registry[_CONSTANTS.PROTEIN_EXP_KEY] = {
            "attr_name": "obsm",
            "attr_key": protein_expression_obsm_key,
        }

    if categorical_covariate_keys is not None:
        cat_loc, cat_key = _setup_extra_categorical_covs(
            adata, categorical_covariate_keys
        )
        data_registry[_CONSTANTS.CAT_COVS_KEY] = {
            "attr_name": cat_loc,
            "attr_key": cat_key,
        }

    if continuous_covariate_keys is not None:
        cont_loc, cont_key = _setup_extra_continuous_covs(
            adata, continuous_covariate_keys
        )
        data_registry[_CONSTANTS.CONT_COVS_KEY] = {
            "attr_name": cont_loc,
            "attr_key": cont_key,
        }

    # add the data_registry to anndata
    _register_anndata(adata, data_registry_dict=data_registry)
    logger.debug(f"Registered keys:{list(data_registry.keys())}")
    _setup_summary_stats(
        adata,
        batch_key,
        labels_key,
        protein_expression_obsm_key,
        categorical_covariate_keys,
        continuous_covariate_keys,
    )

    logger.info("Please do not further modify adata until model is trained.")

    _verify_and_correct_data_format(adata, data_registry)

    adata.obs["_indices"] = np.arange(adata.n_obs).astype("int64")
    register_tensor_from_anndata(
        adata,
        registry_key="ind_x",
        adata_attr_name="obs",
        adata_key_name="_indices",
    )

    epsilon = 1e-6
    if input_type == "knn":
        adata.obs["u_lib_size"] = np.log(
            adata.layers["Mu"].sum(axis=-1).astype("float32") + epsilon
        )
        adata.obs["s_lib_size"] = np.log(
            adata.layers["Ms"].sum(axis=-1).astype("float32") + epsilon
        )
    elif input_type == "raw_cpm":
        from scipy.sparse import issparse

        if issparse(adata.layers["unspliced"]):
            adata.obs["u_lib_size"] = np.log(
                adata.layers["unspliced"].toarray().sum(axis=-1).astype("float32")
                + epsilon
            )
            adata.obs["s_lib_size"] = np.log(
                adata.layers["spliced"].toarray().sum(axis=-1).astype("float32")
                + epsilon
            )
        else:
            adata.obs["u_lib_size"] = np.log(
                adata.layers["unspliced"].sum(axis=-1).astype("float32") + epsilon
            )
            adata.obs["s_lib_size"] = np.log(
                adata.layers["spliced"].sum(axis=-1).astype("float32") + epsilon
            )
    else:
        adata.obs["u_lib_size"] = np.log(adata.obs["u_lib_size_raw"] + epsilon)
        adata.obs["s_lib_size"] = np.log(adata.obs["s_lib_size_raw"] + epsilon)
    adata.obs["u_lib_size_mean"] = adata.obs["u_lib_size"].mean()
    adata.obs["s_lib_size_mean"] = adata.obs["s_lib_size"].mean()
    adata.obs["u_lib_size_scale"] = adata.obs["u_lib_size"].std()
    adata.obs["s_lib_size_scale"] = adata.obs["s_lib_size"].std()

    register_tensor_from_anndata(
        adata,
        registry_key="u_lib_size",
        adata_attr_name="obs",
        adata_key_name="u_lib_size",
    )
    register_tensor_from_anndata(
        adata,
        registry_key="s_lib_size",
        adata_attr_name="obs",
        adata_key_name="s_lib_size",
    )
    register_tensor_from_anndata(
        adata,
        registry_key="u_lib_size_mean",
        adata_attr_name="obs",
        adata_key_name="u_lib_size_mean",
    )
    register_tensor_from_anndata(
        adata,
        registry_key="s_lib_size_mean",
        adata_attr_name="obs",
        adata_key_name="s_lib_size_mean",
    )
    register_tensor_from_anndata(
        adata,
        registry_key="u_lib_size_scale",
        adata_attr_name="obs",
        adata_key_name="u_lib_size_scale",
    )
    register_tensor_from_anndata(
        adata,
        registry_key="s_lib_size_scale",
        adata_attr_name="obs",
        adata_key_name="s_lib_size_scale",
    )

    if "cytotrace" in adata.obs.columns:
        register_tensor_from_anndata(
            adata,
            registry_key="cytotrace",
            adata_attr_name="obs",
            adata_key_name="cytotrace",
        )

    if "kmeans10" in adata.obs.columns:
        register_tensor_from_anndata(
            adata,
            registry_key="kmeans10",
            adata_attr_name="obs",
            adata_key_name="kmeans10",
        )
    if cluster in adata.obs.columns:
        cell_state_dict = {}
        for index, cat in enumerate(adata.obs[cluster].cat.categories):
            cell_state_dict[cat] = index
        adata.obs["pyro_cell_state"] = adata.obs[cluster].map(cell_state_dict)
        print(cell_state_dict)
        print(adata.obs["pyro_cell_state"])
        register_tensor_from_anndata(
            adata,
            registry_key="pyro_cell_state",
            adata_attr_name="obs",
            adata_key_name="pyro_cell_state",
        )

    if "age(days)" in adata.obs.columns:
        time_info_dict = {}
        for index, cat in enumerate(adata.obs["age(days)"].cat.categories):
            time_info_dict[cat] = index

        adata.obs["time_info"] = adata.obs["age(days)"].map(time_info_dict)
        register_tensor_from_anndata(
            adata,
            registry_key="time_info",
            adata_attr_name="obs",
            adata_key_name="time_info",
        )

    if copy:
        return adata
