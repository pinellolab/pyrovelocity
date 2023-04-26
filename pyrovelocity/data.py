import os
from typing import List
from typing import Optional

import anndata
import anndata._core.anndata
import numpy as np
import scanpy as sc
import scvelo as scv
from scipy.sparse import issparse

from pyrovelocity.cytotrace import cytotrace_sparse
from pyrovelocity.utils import print_anndata


def copy_raw_counts(
    adata: anndata._core.anndata.AnnData,
) -> anndata._core.anndata.AnnData:
    """Copy unspliced and spliced raw counts to adata.layers and adata.obs.

    Args:
        adata (anndata._core.anndata.AnnData): AnnData object

    Returns:
        anndata._core.anndata.AnnData: AnnData object with raw counts.

    Examples:
        >>> from pyrovelocity.utils import generate_sample_data
        >>> adata = generate_sample_data()
        >>> copy_raw_counts(adata)
    """
    adata.layers["raw_unspliced"] = adata.layers["unspliced"]
    print("'raw_unspliced', raw unspliced counts (adata.layers)")
    adata.layers["raw_spliced"] = adata.layers["spliced"]
    print("'raw_spliced', raw spliced counts (adata.layers)")
    adata.obs["u_lib_size_raw"] = (
        adata.layers["raw_unspliced"].toarray().sum(-1)
        if issparse(adata.layers["raw_unspliced"])
        else adata.layers["raw_unspliced"].sum(-1)
    )
    print("'u_lib_size_raw', unspliced library size (adata.obs)")
    adata.obs["s_lib_size_raw"] = (
        adata.layers["raw_spliced"].toarray().sum(-1)
        if issparse(adata.layers["raw_spliced"])
        else adata.layers["raw_spliced"].sum(-1)
    )
    print("'s_lib_size_raw', spliced library size (adata.obs)")
    return adata


def load_data(
    data: str = "pancreas",
    top_n: int = 2000,
    min_shared_counts: int = 30,
    eps: float = 1e-6,
    force: bool = False,
    processed_path: str = None,
    process_cytotrace: bool = False,
) -> anndata._core.anndata.AnnData:
    """Preprocess data from scvelo.

    Args:
        data (str, optional): data set name. Defaults to scvelo's "pancreas" data set.
        top_n (int, optional): number of genes to retain. Defaults to 2000.
        min_shared_counts (int, optional): minimum shared counts. Defaults to 30.
        eps (float, optional): tolerance. Defaults to 1e-6.
        force (bool, optional): force reprocessing. Defaults to False.
        processed_path (str, optional): path to read/write processed AnnData. Defaults to None.

    Returns:
        anndata._core.anndata.AnnData: processed AnnData object
    """
    if processed_path is None:
        processed_path = f"{data}_scvelo_fitted_{top_n}_{min_shared_counts}.h5ad"

    if (
        os.path.isfile(processed_path)
        and os.access(processed_path, os.R_OK)
        and (not force)
    ):
        adata = sc.read(processed_path)
    else:
        if data == "pancreas":
            adata = scv.datasets.pancreas()
        elif data == "forebrain":
            adata = scv.datasets.forebrain()
        elif data == "dentategyrus_lamanno":
            adata = scv.datasets.dentategyrus_lamanno()
        elif data == "dentategyrus":
            adata = scv.datasets.dentategyrus()
        elif data == "larry":
            adata = load_larry()
        else:
            adata = sc.read(data)

        print_anndata(adata)
        copy_raw_counts(adata)
        print_anndata(adata)

        if "pbmc68k" in processed_path:
            print("Removing duplicate cells and tSNE x-parity in pbmc68k data...")
            scv.pp.remove_duplicate_cells(adata)
            adata.obsm["X_tsne"][:, 0] *= -1

        if process_cytotrace:
            print("Processing data with cytotrace ...")
            cytotrace_sparse(adata, layer="spliced")

        scv.pp.filter_and_normalize(
            adata, min_shared_counts=min_shared_counts, n_top_genes=top_n
        )
        scv.pp.moments(adata, n_pcs=30, n_neighbors=30)
        if "X_umap" not in adata.obsm.keys():
            scv.tl.umap(adata)
        if "leiden" not in adata.obs.keys():
            sc.tl.leiden(adata)
        scv.tl.recover_dynamics(adata, n_jobs=-1, use_raw=False)
        scv.tl.velocity(adata, mode="dynamical", use_raw=False)
        scv.tl.velocity_graph(adata, n_jobs=-1)

        if data == "larry":
            scv.tl.velocity_embedding(adata, basis="emb")
        else:
            scv.tl.velocity_embedding(adata)

        scv.tl.latent_time(adata)

        print_anndata(adata)
        adata.write(processed_path)

    return adata


def load_pbmc(
    data: str = None,
    processed_path: str = "pbmc68k_perspectives_processed.h5ad",
) -> anndata._core.anndata.AnnData:
    if (
        os.path.isfile(processed_path)
        and os.access(processed_path, os.R_OK)
        and (not force)
    ):
        adata = scv.sc.read(processed_path)
    else:
        if data is None:
            adata = scv.datasets.pbmc68k()
        elif os.path.isfile(data) and os.access(data, os.R_OK):
            adata = scv.sc.read(data)

        print_anndata(adata)
        copy_raw_counts(adata)
        print_anndata(adata)

        adata.obsm["X_tsne"][:, 0] *= -1
        scv.pp.remove_duplicate_cells(adata)
        scv.pp.filter_and_normalize(adata, min_shared_counts=30, n_top_genes=2000)
        scv.pp.moments(adata)
        scv.tl.velocity(adata, mode="stochastic")
        scv.tl.recover_dynamics(adata, n_jobs=-1)

        top_genes = adata.var["fit_likelihood"].sort_values(ascending=False).index
        print(top_genes[:10])
        adata_sub = adata[:, top_genes[:3]].copy()
        scv.tl.velocity_graph(adata_sub, n_jobs=-1)
        scv.tl.velocity_embedding(adata_sub)

        adata_sub.obs["u_lib_size_raw"] = np.array(
            adata_sub.layers["raw_unspliced"].sum(axis=-1), dtype=np.float32
        ).flatten()
        adata_sub.obs["s_lib_size_raw"] = np.array(
            adata_sub.layers["raw_spliced"].sum(axis=-1), dtype=np.float32
        ).flatten()
        print_anndata(adata_sub)
        adata_sub.write(processed_path)

    return adata_sub


def load_larry(file_path: str = "data/larry.h5ad") -> anndata._core.anndata.AnnData:
    """In vitro Hemotopoiesis Larry datasets

    Data from `CALEB WEINREB et al. (2020) <DOI: 10.1126/science.aaw3381>'
    https://figshare.com/ndownloader/articles/20780344/versions/1

    Returns
    -------
    Returns `adata` object
    """
    url = "https://figshare.com/ndownloader/files/37028569"
    adata = sc.read(file_path, backup_url=url, sparse=True, cache=True)
    return adata


def load_unipotent_larry(celltype: str = "mono") -> anndata._core.anndata.AnnData:
    """In vitro Hemotopoiesis Larry datasets
    Subset of Data from `CALEB WEINREB et al. (2020) <DOI: 10.1126/science.aaw3381>'
    unipotent monocytes: https://figshare.com/ndownloader/files/37028572
    unipotent neutrophils: https://figshare.com/ndownloader/files/37028575

    Returns
    -------
    Returns `adata` object
    """
    file_path = f"data/larry_{celltype}.h5ad"
    if celltype == "mono":
        url = "https://figshare.com/ndownloader/files/37028572"
    else:  # neutrophil
        url = "https://figshare.com/ndownloader/files/37028575"
    adata = sc.read(file_path, backup_url=url, sparse=True, cache=True)
    return adata
