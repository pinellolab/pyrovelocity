from pathlib import Path

import anndata
import scanpy as sc
import scvelo as scv
from beartype import beartype


@beartype
def pbmc5k(
    file_path: str | Path = "data/external/pbmc5k.h5ad",
) -> anndata._core.anndata.AnnData:
    """
    10x genomics 5k Peripheral blood mononuclear cells (PBMCs) from a healthy donor.

    https://www.10xgenomics.com/datasets/peripheral-blood-mononuclear-cells-pbm-cs-from-a-healthy-donor-chromium-connect-channel-1-3-1-standard-3-1-0

    Returns:
        Returns `AnnData` object
    """
    url = "https://storage.googleapis.com/pyrovelocity/data/pbmc5k.h5ad"
    adata = sc.read(file_path, backup_url=url, sparse=True, cache=True)
    return adata


@beartype
def pbmc10k(
    file_path: str | Path = "data/external/pbmc10k.h5ad",
) -> anndata._core.anndata.AnnData:
    """
    10x genomics 10k Human PBMCs.

    https://www.10xgenomics.com/datasets/10k-human-pbmcs-3-ht-v3-1-chromium-x-3-1-high

    Returns:
        Returns `AnnData` object
    """
    url = "https://storage.googleapis.com/pyrovelocity/data/pbmc10k.h5ad"
    adata = sc.read(file_path, backup_url=url, sparse=True, cache=True)
    return adata


@beartype
def pons(
    file_path: str | Path = "data/external/pons.h5ad",
) -> anndata._core.anndata.AnnData:
    """
    Pons oligodendrocyte data from `La Manno et al. (2018) <https://doi.org/https://doi.org/10.1038/s41586-018-0414-6>`__.

    Data originally obtained from https://pklab.med.harvard.edu/ruslan/velocity/oligos/oligos_info.rds
    and converted to AnnData format.

    Returns:
        Returns `AnnData` object
    """
    url = "https://storage.googleapis.com/pyrovelocity/data/oligo_lite.h5ad"
    adata = sc.read(file_path, backup_url=url, sparse=True, cache=True)
    return adata


@beartype
def larry(
    file_path: str | Path = "data/external/larry.h5ad",
) -> anndata._core.anndata.AnnData:
    """
    In vitro hematopoiesis LARRY dataset.

    Data from `Weinreb et al. (2020) <DOI: 10.1126/science.aaw3381>'
    https://figshare.com/ndownloader/articles/20780344/versions/1

    Returns:
        Returns `AnnData` object
    """
    url = "https://figshare.com/ndownloader/files/37028569"
    adata = sc.read(file_path, backup_url=url, sparse=True, cache=True)
    return adata


@beartype
def larry_neu(
    file_path: str | Path = "data/external/larry_neu.h5ad",
) -> anndata._core.anndata.AnnData:
    """
    In vitro hematopoiesis LARRY dataset.

    Subset of Data from `Weinreb et al. (2020) <DOI: 10.1126/science.aaw3381>'
    consisting of unipotent neutrophils.

    https://figshare.com/ndownloader/files/37028575

    Returns:
        Returns `AnnData` object
    """
    url = "https://figshare.com/ndownloader/files/37028575"
    adata = sc.read(file_path, backup_url=url, sparse=True, cache=True)
    adata = adata[adata.obs.state_info != "Centroid", :]
    return adata


@beartype
def larry_mono(
    file_path: str | Path = "data/external/larry_mono.h5ad",
) -> anndata._core.anndata.AnnData:
    """
    In vitro hematopoiesis LARRY dataset
    Subset of Data from `Weinreb et al. (2020) <DOI: 10.1126/science.aaw3381>'
    consisting of unipotent monocytes.

    https://figshare.com/ndownloader/files/37028572

    Returns:
        Returns `AnnData` object
    """
    url = "https://figshare.com/ndownloader/files/37028572"
    adata = sc.read(file_path, backup_url=url, sparse=True, cache=True)
    adata = adata[adata.obs.state_info != "Centroid", :]
    return adata


@beartype
def larry_cospar(
    file_path: str | Path = "data/external/larry_cospar.h5ad",
) -> anndata._core.anndata.AnnData:
    """
    In vitro hematopoiesis LARRY dataset with cospar-based fate estimates.

    Subset of Data from `Weinreb et al. (2020) <DOI: 10.1126/science.aaw3381>'

    Returns:
        Returns `AnnData` object
    """
    url = "https://storage.googleapis.com/pyrovelocity/data/larry_cospar.h5ad"
    adata = sc.read(file_path, backup_url=url, sparse=True, cache=True)
    return adata


@beartype
def larry_cytotrace(
    file_path: str | Path = "data/external/larry_cytotrace.h5ad",
) -> anndata._core.anndata.AnnData:
    """
    In vitro hematopoiesis LARRY dataset with cytotrace-based fate estimates.

    Subset of Data from `Weinreb et al. (2020) <DOI: 10.1126/science.aaw3381>'

    Returns:
        Returns `AnnData` object
    """
    url = (
        "https://storage.googleapis.com/pyrovelocity/data/larry_cytotrace.h5ad"
    )
    adata = sc.read(file_path, backup_url=url, sparse=True, cache=True)
    return adata


@beartype
def larry_dynamical(
    file_path: str | Path = "data/external/larry_dynamical.h5ad",
) -> anndata._core.anndata.AnnData:
    """
    The LARRY in vitro hematopoiesis dataset with scvelo dynamical model output.

    Subset of Data from `Weinreb et al. (2020) <DOI: 10.1126/science.aaw3381>'

    Returns:
        Returns `AnnData` object
    """
    url = (
        "https://storage.googleapis.com/pyrovelocity/data/larry_dynamical.h5ad"
    )
    adata = sc.read(file_path, backup_url=url, sparse=True, cache=True)
    return adata


@beartype
def larry_tips(
    file_path: str | Path = "data/external/larry_tips.h5ad",
) -> anndata._core.anndata.AnnData:
    """
    The differentiated subset of the LARRY in vitro hematopoiesis dataset.

    Subset of Data from `Weinreb et al. (2020) <DOI: 10.1126/science.aaw3381>'

    Returns:
        Returns `AnnData` object
    """
    adata = larry()
    adata = adata[adata.obs["time_info"] == 6.0]
    adata = adata[adata.obs["state_info"] != "Undifferentiated"]
    return adata


@beartype
def larry_multilineage(
    file_path: str | Path = "data/external/larry_multilineage.h5ad",
) -> anndata._core.anndata.AnnData:
    """
    The monocyte and neutrophil subset of the LARRY in vitro hematopoiesis dataset.

    Subset of Data from `Weinreb et al. (2020) <DOI: 10.1126/science.aaw3381>'

    Returns:
        Returns `AnnData` object
    """
    adata_larry_mono = larry_mono()
    adata_larry_neu = larry_neu()
    adata = adata_larry_mono.concatenate(adata_larry_neu)
    return adata


@beartype
def pancreas(
    file_path: str | Path = "data/external/pancreas.h5ad",
) -> anndata._core.anndata.AnnData:
    """
    Pancreatic endocrinogenesis data sourced from the scvelo library.

    Data from `Bastidas-Ponce et al. (2019) <https://doi.org/10.1242/dev.173849>`__
    will be downloaded from

    https://github.com/theislab/scvelo_notebooks/raw/master/data/Pancreas/endocrinogenesis_day15.h5ad

    Args:
        file_path (str, optional): Path to save file. Defaults to "data/external/pancreas.h5ad".

    Returns:
        Returns `AnnData` object
    """
    adata = scv.datasets.pancreas(file_path=file_path)
    return adata


@beartype
def bonemarrow(
    file_path: str | Path = "data/external/bonemarrow.h5ad",
) -> anndata._core.anndata.AnnData:
    """
    Human bone marrow data sourced from the scvelo library.

    Data from `Setty et al. (2019) <https://doi.org/10.1038/s41587-019-0068-4>`__
    will be downloaded from

    https://ndownloader.figshare.com/files/27686835

    Args:
        file_path (str, optional): Path to save file. Defaults to "data/external/bonemarrow.h5ad".

    Returns:
        Returns `AnnData` object
    """
    adata = scv.datasets.bonemarrow(file_path=file_path)
    return adata


@beartype
def pbmc68k(
    file_path: str | Path = "data/external/pbmc68k.h5ad"
) -> anndata._core.anndata.AnnData:
    """
    Peripheral blood mononuclear cells data sourced from the scvelo library.

    Data from `Zheng et al. (2017) <https://doi.org/10.1038/ncomms14049>`__
    will be downloaded from

    https://ndownloader.figshare.com/files/27686886

    Args:
        file_path (str, optional): Path to save file. Defaults to "data/external/pbmc68k.h5ad".

    Returns:
        Returns `AnnData` object
    """
    adata = scv.datasets.pbmc68k(file_path=file_path)
    scv.pp.remove_duplicate_cells(adata)
    adata.obsm["X_tsne"][:, 0] *= -1
    return adata
