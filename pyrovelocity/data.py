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
    use_sub: bool = False,
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
        processed_path = f"{data}_processed.h5ad"
    print(
        processed_path,
        os.path.isfile(processed_path),
        os.access(processed_path, os.R_OK),
        (not force),
    )
    if (
        os.path.isfile(processed_path)
        and os.access(processed_path, os.R_OK)
        and (not force)
    ):
        adata = sc.read(processed_path)
    else:
        print("Dataset name:", data)
        if data == "pancreas":
            adata = scv.datasets.pancreas()
        elif data == "forebrain":
            adata = scv.datasets.forebrain()
        elif data == "dentategyrus_lamanno":
            adata = scv.datasets.dentategyrus_lamanno()
        elif data == "dentategyrus":
            adata = scv.datasets.dentategyrus()
        elif "larry" in data:
            data = data.split("/")[-1].split(".")[0]
            print("Larry dataset name:", data)
            if data == "larry":
                adata = load_larry()
            elif data == "larry_tips":
                adata = load_larry()
                adata = adata[adata.obs["time_info"] == 6.0]
                adata = adata[adata.obs["state_info"] != "Undifferentiated"]
            elif data in ["larry_mono", "larry_neu"]:
                adata = load_unipotent_larry(data.split("_")[1])
                adata = adata[adata.obs.state_info != "Centroid", :]
            elif data == "larry_multilineage":
                adata_mono = load_unipotent_larry("mono")
                adata_mono_C = adata_mono[
                    adata_mono.obs.state_info != "Centroid", :
                ].copy()
                adata_neu = load_unipotent_larry("neu")
                adata_neu_C = adata_neu[
                    adata_neu.obs.state_info != "Centroid", :
                ].copy()
                adata_multilineage = adata_mono.concatenate(adata_neu)
                adata = adata_mono_C.concatenate(adata_neu_C)
                adata.layers["raw_spliced"] = adata_multilineage[
                    adata.obs_names, adata.var_names
                ].layers["spliced"]
                adata.layers["raw_unspliced"] = adata_multilineage[
                    adata.obs_names, adata.var_names
                ].layers["unspliced"]
        elif "pbmc68k" in data:
            adata = load_pbmc68k()
        else:  # pbmc10k
            adata = sc.read(data)

        print_anndata(adata)
        if "raw_unspliced" not in adata.layers:
            copy_raw_counts(adata)
            print_anndata(adata)

        if process_cytotrace:
            print("Processing data with cytotrace ...")
            cytotrace_sparse(adata, layer="spliced")

        if not "pbmc68k" in data:
            scv.pp.filter_and_normalize(
                adata, min_shared_counts=min_shared_counts, n_top_genes=top_n
            )
            scv.pp.moments(adata, n_pcs=30, n_neighbors=30)
            scv.tl.recover_dynamics(adata, n_jobs=-1, use_raw=False)
            scv.tl.velocity(adata, mode="dynamical", use_raw=False)
        if ("X_umap" not in adata.obsm.keys()) or (data == "larry_tips"):
            scv.tl.umap(adata)
        if "leiden" not in adata.obs.keys():
            sc.tl.leiden(adata)
        if use_sub:
            top_genes = adata.var["fit_likelihood"].sort_values(ascending=False).index
            print(top_genes[:10])
            adata = adata[:, top_genes[:3]].copy()
        scv.tl.velocity_graph(adata, n_jobs=-1)

        if data in ["larry", "larry_mono", "larry_neu", "larry_multilineage"]:
            scv.tl.velocity_embedding(adata, basis="emb")
        else:
            scv.tl.velocity_embedding(adata)

        scv.tl.latent_time(adata)

        print_anndata(adata)
        adata.write(processed_path)

    return adata


def load_pbmc68k(
    data: str = "pbmc68k",  # pbmc68k or pbmc10k
) -> anndata._core.anndata.AnnData:
    if data == "pbmc68k":
        adata = scv.datasets.pbmc68k()
    elif os.path.isfile(data) and os.access(data, os.R_OK):
        adata = sc.read(data)

    print_anndata(adata)
    if "raw_unspliced" not in adata.layers:
        copy_raw_counts(adata)
        print_anndata(adata)
    print("Removing duplicate cells and tSNE x-parity in pbmc68k data...")
    scv.pp.remove_duplicate_cells(adata)
    adata.obsm["X_tsne"][:, 0] *= -1
    scv.pp.filter_and_normalize(adata, min_shared_counts=30, n_top_genes=2000)
    scv.pp.moments(adata)
    scv.tl.velocity(adata, mode="stochastic")
    scv.tl.recover_dynamics(adata, n_jobs=-1)

    return adata


def load_larry(
    file_path: str = "data/external/larry.h5ad",
) -> anndata._core.anndata.AnnData:
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
    file_path = f"data/external/larry_{celltype}.h5ad"
    if celltype == "mono":
        url = "https://figshare.com/ndownloader/files/37028572"
    else:  # neutrophil
        url = "https://figshare.com/ndownloader/files/37028575"
    adata = sc.read(file_path, backup_url=url, sparse=True, cache=True)
    return adata
