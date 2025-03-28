import os
from pathlib import Path

import anndata
import scanpy as sc
import scvelo as scv
from beartype import beartype

from pyrovelocity.io.hash import hash_file
from pyrovelocity.logging import configure_logging

__all__ = [
    "pbmc5k",
    "pbmc10k",
    "pons",
    "larry",
    "larry_neu",
    "larry_mono",
    "larry_cospar",
    "larry_cytotrace",
    "larry_dynamical",
    "larry_tips",
    "larry_multilineage",
    "pancreas",
    "bonemarrow",
    "pbmc68k",
    "larry_mono_clone_trajectory",
    "larry_neu_clone_trajectory",
    "larry_multilineage_clone_trajectory",
]


logger = configure_logging(__name__)


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
    expected_hash = (
        "349028bdf7992f5b196a3c4efd7a83cebdf0624d3d42a712628967dad608ad35"
    )
    _check_hash(file_path, expected_hash)
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
    expected_hash = (
        "267d1d2710251a68413fcb82fa03f9bcfe8fe33a6bae05117603da983ebe2c5b"
    )
    _check_hash(file_path, expected_hash)
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
    expected_hash = (
        "d3a3286a6f33c307aca20bcbb127abb6ac52dbf6c968ee24df6ed9584b857de0"
    )
    _check_hash(file_path, expected_hash)
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
    expected_hash = (
        "7f567427f591e85678580ebaa8b1e59aae51e9be63864a68ef9e905a0cbe8575"
    )
    _check_hash(file_path, expected_hash)
    return adata


@beartype
def larry_neu(
    file_path: str | Path = "data/external/larry_neu.h5ad",
) -> anndata._core.anndata.AnnData:
    """
    In vitro hematopoiesis LARRY dataset.

    Subset of Data from `Weinreb et al. (2020) <DOI: 10.1126/science.aaw3381>'
    consisting of unipotent neutrophil precursors and neutrophils.

    https://figshare.com/ndownloader/files/37028575

    Returns:
        Returns `AnnData` object
    """
    url = "https://figshare.com/ndownloader/files/37028575"

    if os.path.isfile(file_path):
        adata = sc.read(file_path, backup_url=url, sparse=True, cache=True)
    else:
        adata = sc.read(file_path, backup_url=url, sparse=True, cache=True)
        premodification_hash = (
            "ae4113834a1318168c92715887173d27bf88c57ccbd715e69481b13cf2539b92"
        )
        _check_hash(file_path, premodification_hash)
        adata = adata[adata.obs.state_info != "Centroid", :]
        adata.write(file_path)

    expected_hash = (
        "384784699c10e192677c006bb407aaedbdf3e3c66f1ca1f4d8d1284ddf8fa436"
    )
    _check_hash(file_path, expected_hash)
    return adata


@beartype
def larry_mono(
    file_path: str | Path = "data/external/larry_mono.h5ad",
) -> anndata._core.anndata.AnnData:
    """
    In vitro hematopoiesis LARRY dataset
    Subset of Data from `Weinreb et al. (2020) <DOI: 10.1126/science.aaw3381>'
    consisting of unipotent monocyte precursors and monocytes.

    https://figshare.com/ndownloader/files/37028572

    Returns:
        Returns `AnnData` object
    """
    url = "https://figshare.com/ndownloader/files/37028572"

    if os.path.isfile(file_path):
        adata = sc.read(file_path, backup_url=url, sparse=True, cache=True)
    else:
        adata = sc.read(file_path, backup_url=url, sparse=True, cache=True)
        premodification_hash = (
            "b880b7f72f0ccc8b11ca63c53d340983a6d478e214d4960a529d6e02a9ccd597"
        )
        _check_hash(file_path, premodification_hash)
        adata = adata[adata.obs.state_info != "Centroid", :]
        adata.write(file_path)

    expected_hash = (
        "75e59aa7f0d47d2d013dc7444f89a858363110ba32d7a576ac3dc819cac0afa8"
    )
    _check_hash(file_path, expected_hash)
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
    expected_hash = (
        "cdf2ff25c4e3222122beeff2da65539ba7541f4426547f4622813874fd9be070"
    )
    _check_hash(file_path, expected_hash)
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
    expected_hash = (
        "fdaa99408f93c52d993cd47bf7997f5c01f7cb88be1824864075c67afb04b625"
    )
    _check_hash(file_path, expected_hash)
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
    expected_hash = (
        "d3808463203f3d0a8fd1eb76ac723d8e6ab939223eb528434c3e38926a460863"
    )
    _check_hash(file_path, expected_hash)
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
    adata.write(file_path)
    expected_hash = (
        "e04e9d0b82651d170c93aaa9a1a0f764c91a986d8ae97d9401b4ee6c496f492c"
    )
    _check_hash(file_path, expected_hash)
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
    if os.path.isfile(file_path):
        adata = sc.read(file_path, sparse=True, cache=True)
    else:
        adata_larry_mono = larry_mono()
        adata_larry_neu = larry_neu()
        adata = adata_larry_mono.concatenate(adata_larry_neu)
        adata.write(file_path)
    expected_hash = (
        "9add35ae4f736aa5e11d076eadb3b1d842dbc88102047f029bd7fa0929f46be0"
    )
    _check_hash(file_path, expected_hash)
    return adata


@beartype
def larry_mono_clone_trajectory(
    file_path: str | Path = "data/external/larry_mono_clone_trajectory.h5ad",
) -> anndata._core.anndata.AnnData:
    """
    Pre-computed clone trajectory data for the LARRY monocyte lineage.

    This contains the output of get_clone_trajectory applied to the larry_mono dataset.
    The clone trajectory information is used for visualizing clonal progression
    and calculating trajectory alignment with velocity predictions.

    Returns:
        AnnData object with clone trajectory information
    """
    url = "https://storage.googleapis.com/pyrovelocity/data/larry_mono_clone_trajectory.h5ad"
    adata = sc.read(file_path, backup_url=url, sparse=True, cache=True)
    expected_hash = (
        "f5d0dcb9baa63460c5be5a1ebdab6a97c6f3ec0b5641ab1b770d16fb96bd9fc9"
    )
    _check_hash(file_path, expected_hash)
    return adata


@beartype
def larry_neu_clone_trajectory(
    file_path: str | Path = "data/external/larry_neu_clone_trajectory.h5ad",
) -> anndata._core.anndata.AnnData:
    """
    Pre-computed clone trajectory data for the LARRY neutrophil lineage.

    This contains the output of get_clone_trajectory applied to the larry_neu dataset.
    The clone trajectory information is used for visualizing clonal progression
    and calculating trajectory alignment with velocity predictions.

    Returns:
        AnnData object with clone trajectory information
    """
    url = "https://storage.googleapis.com/pyrovelocity/data/larry_neu_clone_trajectory.h5ad"
    adata = sc.read(file_path, backup_url=url, sparse=True, cache=True)
    expected_hash = (
        "6e7dbc273c59e28f1962df31452d5eea00336089c36a44f55fcfc91f6f428396"
    )
    _check_hash(file_path, expected_hash)
    return adata


@beartype
def larry_multilineage_clone_trajectory(
    file_path: str
    | Path = "data/external/larry_multilineage_clone_trajectory.h5ad",
) -> anndata._core.anndata.AnnData:
    """
    Pre-computed clone trajectory data for the LARRY multilineage dataset.

    This contains the concatenated output of get_clone_trajectory applied to
    both larry_mono and larry_neu datasets. Using this pre-computed trajectory
    ensures consistent fate analysis across both lineages without recomputing
    trajectories separately.

    Returns:
        AnnData object with clone trajectory information
    """
    url = "https://storage.googleapis.com/pyrovelocity/data/larry_multilineage_clone_trajectory.h5ad"
    adata = sc.read(file_path, backup_url=url, sparse=True, cache=True)
    expected_hash = (
        "ffedda0332c411ca10c09562e5c8a50643af9120f65b0b3701bf30a8d5fdc97b"
    )
    _check_hash(file_path, expected_hash)
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
    expected_hash = (
        "9e3e459eca00ba06b496ec80def32941b5b2889918720e3e7aa6ffb811fbe7c6"
    )
    _check_hash(file_path, expected_hash)
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
    expected_hash = (
        "12222efe4a9dd5916fa279860a2a4fdec383bd2e0db0249a1cd18c549c4a4c2c"
    )
    _check_hash(file_path, expected_hash)
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
    if os.path.isfile(file_path):
        adata = scv.datasets.pbmc68k(file_path=file_path)
    else:
        adata = scv.datasets.pbmc68k(file_path=file_path)
        premodification_hash = (
            "c93b1ccad909b6a41539a57975737bd946ea1becce066c250aca129d8dfa26fb"
        )
        _check_hash(file_path, premodification_hash)
        scv.pp.remove_duplicate_cells(adata)
        adata.obsm["X_tsne"][:, 0] *= -1
        adata.write(file_path)
    expected_hash = (
        "c6ce6ca3dac3b97012d12c7a5e5ec1953bbd2ad5535538b0ef549f54d9276f0b"
    )
    _check_hash(file_path, expected_hash)
    return adata


@beartype
def _check_hash(file_path: str | Path, expected_hash: str) -> None:
    actual_hash = _log_hash(file_path)
    if actual_hash != expected_hash:
        logger.error(
            f"\nHash mismatch for {file_path}.\n"
            f"Expected: {expected_hash}\n"
            f"Actual: {actual_hash}"
        )
    else:
        logger.info(f"\nHash check passed: {file_path}")


@beartype
def _log_hash(file_path: str | Path) -> str:
    adata_hash = hash_file(file_path=file_path)
    logger.info(
        f"\nSuccessfully read or created file: {file_path}\n"
        f"SHA-256 hash: {adata_hash}\n"
    )
    return adata_hash
