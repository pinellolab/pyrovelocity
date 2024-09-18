from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import scanpy as sc
from anndata import AnnData
from beartype import beartype
from scipy.sparse import csr_matrix

from pyrovelocity.logging import configure_logging
from pyrovelocity.utils import print_anndata

__all__ = [
    "subset_anndata",
    "subset_vars",
    "subset_randomly",
    "subset_by_lineage_timepoints",
    "update_uns_attributes",
    "save_subset_to_file",
]

logger = configure_logging(__name__)


@beartype
def subset_anndata(
    file_path: Optional[str | Path] = None,
    adata: Optional[AnnData] = None,
    n_obs: int = 100,
    n_vars: Optional[int] = None,
    save_subset: bool = False,
    output_path: Optional[str | Path] = None,
    use_lineage_timepoints: bool = False,
) -> Tuple[AnnData, Optional[str | Path]]:
    """
    Randomly sample observations from a dataset given by file path or AnnData object.
    If use_lineage_timepoints is True, it splits the observations equally between two timepoints.

    Args:
        file_path (str): Path to a .h5ad file containing a dataset. Takes precedence over adata.
        adata (AnnData): AnnData object. If None, file_path must be provided.
        n_obs (int): Number of observations to sample. Defaults to 100.
        n_vars (Optional[int]): Number of variables to subset. If None, all variables are kept.
        save_subset (bool): If True, save the subset to a file. Defaults to False.
        output_path (str): Path to save the subset. Defaults to None.
        use_lineage_timepoints (bool): If True, split observations equally between two timepoints. Defaults to False.

    Raises:
        ValueError: If neither file_path nor adata is provided.

    Returns:
        Tuple[AnnData, Optional[str | Path]]: Subset of the dataset and the output path if saved.

    Examples:
        >>> # xdoctest: +SKIP
        >>> data_path = download_dataset(data_set_name="pancreas")
        >>> adata = subset_anndata(file_path=data_path, n_obs=100, save_subset=True)
        >>> print_anndata(adata)
        >>> print_attributes(adata)
        ...
        >>> # use_lineage_timepoints=True
        >>> from pyrovelocity.io.datasets import larry_cospar
        >>> from pyrovelocity.io.serialization import save_anndata_to_json
        >>> adata_cospar = larry_cospar()
        >>> adata_cospar_subset, outpath = subset_anndata(
        ...     adata=adata_cospar,
        ...     n_obs=4,
        ...     n_vars=2,
        ...     use_lineage_timepoints=True,
        ... )
        >>> save_anndata_to_json(adata_cospar_subset, "adata_cospar_subset.json")
    """
    if file_path is not None:
        file_path = Path(file_path)
        adata = sc.read(file_path, cache=True)
    if adata is None:
        raise ValueError("Either file_path or adata must be provided")

    logger.info("constructing data subset")
    print_anndata(adata)

    if n_vars is not None:
        adata = subset_vars(adata, n_vars)

    if use_lineage_timepoints:
        adata_subset = subset_by_lineage_timepoints(adata, n_obs)
    else:
        adata_subset = subset_randomly(adata, n_obs)

    adata_subset.obs_names_make_unique()
    adata_subset.var_names_make_unique()

    if save_subset:
        output_path = save_subset_to_file(
            adata_subset, file_path, output_path, n_obs
        )

    print_anndata(adata_subset)
    return adata_subset.copy(), output_path


@beartype
def subset_vars(adata: AnnData, n_vars: int) -> AnnData:
    """Subset variables of the AnnData object."""
    if n_vars > adata.n_vars:
        logger.warning(
            f"n_vars ({n_vars}) is greater than the number of variables in the dataset ({adata.n_vars})"
        )
        n_vars = adata.n_vars
    selected_vars_indices = np.random.choice(
        adata.n_vars, n_vars, replace=False
    )
    logger.info(f"selected {n_vars} vars from {adata.n_vars}")
    return adata[:, selected_vars_indices]


def subset_randomly(adata: AnnData, n_obs: int) -> AnnData:
    """Randomly subset observations of the AnnData object."""
    if n_obs > adata.n_obs:
        logger.warning(
            f"n_obs ({n_obs}) is greater than the number of observations in the dataset ({adata.n_obs})"
        )
        n_obs = adata.n_obs
    selected_obs_indices = np.random.choice(adata.n_obs, n_obs, replace=False)
    logger.info(f"selected {n_obs} obs from {adata.n_obs}")
    return adata[selected_obs_indices]


@beartype
def subset_by_lineage_timepoints(
    adata: AnnData,
    n_obs: int,
    n_subset_pcs=5,
) -> AnnData:
    """Subset observations equally between two lineage timepoints.

    Examples:
        >>> # xdoctest: +SKIP
        >>> data_path = download_dataset(data_set_name="pancreas")
        >>> adata, _ = subset_anndata(file_path=data_path, n_obs=100, save_subset=True)
        >>> adata_subset = subset_by_lineage_timepoints(adata, n_obs=10)
        >>> print_anndata(adata
    """
    required_keys = ["clonal_cell_id_t1", "clonal_cell_id_t2"]
    if not all(key in adata.uns for key in required_keys):
        raise ValueError(
            f"AnnData object is missing one or more required uns keys: {required_keys}"
        )

    n_obs_per_timepoint = n_obs // 2
    t1_available = len(adata.uns["clonal_cell_id_t1"])
    t2_available = len(adata.uns["clonal_cell_id_t2"])

    if n_obs_per_timepoint > min(t1_available, t2_available):
        n_obs_per_timepoint = min(t1_available, t2_available)
        logger.warning(
            f"Requested number of observations per timepoint ({n_obs // 2}) "
            f"exceeds available observations. Using {n_obs_per_timepoint} per timepoint."
        )

    t1_indices = np.random.choice(
        adata.uns["clonal_cell_id_t1"], n_obs_per_timepoint, replace=False
    )
    t2_indices = np.random.choice(
        adata.uns["clonal_cell_id_t2"], n_obs_per_timepoint, replace=False
    )
    selected_obs_indices = np.concatenate([t1_indices, t2_indices])

    logger.info(
        f"selected {n_obs_per_timepoint} obs from each timepoint, total {len(selected_obs_indices)} obs"
    )
    adata_subset = adata[selected_obs_indices].copy()
    update_uns_attributes(adata_subset, adata, t1_indices, t2_indices)
    adata_subset.obsm["X_clone"] = np.eye(adata_subset.n_obs)
    adata_subset.obsm["X_pca"] = adata_subset.obsm["X_pca"][:, :n_subset_pcs]
    adata_subset.varm["PCs"] = adata_subset.varm["PCs"][:, :n_subset_pcs]
    adata_subset.uns["pca"]["variance"] = adata_subset.uns["pca"]["variance"][
        :n_subset_pcs
    ]
    adata_subset.uns["pca"]["variance_ratio"] = adata_subset.uns["pca"][
        "variance_ratio"
    ][:n_subset_pcs]
    return adata_subset


@beartype
def update_uns_attributes(
    adata_subset: AnnData,
    adata: AnnData,
    t1_indices: np.ndarray,
    t2_indices: np.ndarray,
):
    """Update uns attributes in the subsetted AnnData object."""
    adata_subset.uns["clonal_cell_id_t1"] = np.arange(len(t1_indices))
    adata_subset.uns["clonal_cell_id_t2"] = np.arange(
        len(t1_indices), len(t1_indices) + len(t2_indices)
    )
    adata_subset.uns["Tmap_cell_id_t1"] = adata_subset.uns["clonal_cell_id_t1"]
    adata_subset.uns["Tmap_cell_id_t2"] = adata_subset.uns["clonal_cell_id_t2"]

    for key in ["intraclone_transition_map", "transition_map"]:
        if key in adata.uns:
            original_map = adata.uns[key]
            new_map = csr_matrix((len(t1_indices), len(t2_indices)))

            t1_mapping = {orig: new for new, orig in enumerate(t1_indices)}
            t2_mapping = {orig: new for new, orig in enumerate(t2_indices)}

            rows, cols = original_map.nonzero()
            for row, col in zip(rows, cols):
                if row in t1_mapping and col in t2_mapping:
                    new_row = t1_mapping[row]
                    new_col = t2_mapping[col]
                    new_map[new_row, new_col] = original_map[row, col]

            adata_subset.uns[key] = new_map.toarray()

    for key in adata.uns:
        if key not in adata_subset.uns:
            adata_subset.uns[key] = adata.uns[key]
    adata_subset.uns["sp_idx"] = np.ones(adata_subset.shape[0]).astype(bool)


@beartype
def save_subset_to_file(
    adata_subset: AnnData,
    file_path: Optional[Path],
    output_path: Optional[str | Path],
    n_obs: int,
) -> Path:
    """Save the subsetted AnnData object to a file."""
    if output_path is None and file_path is not None:
        output_path = file_path.parent / Path(
            f"{file_path.stem}_{n_obs}obs{file_path.suffix}"
        )
    if output_path is None:
        raise ValueError(
            "output_path must be provided if save_subset is True and file_path is None"
        )
    output_path = Path(output_path)
    adata_subset.write(output_path)
    logger.info(f"saved {n_obs} obs subset: {output_path}")
    return output_path
