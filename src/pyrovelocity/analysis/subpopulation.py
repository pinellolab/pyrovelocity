"""
Functions for extracting clonal subpopulations from datasets with clone information.
"""

import numpy as np
import pandas as pd
from anndata import AnnData
from beartype import beartype
from beartype.typing import List, Optional, Tuple
from scipy.sparse import issparse

from pyrovelocity.logging import configure_logging

logger = configure_logging(__name__)


@beartype
def select_clones(
    df_metadata: pd.DataFrame,
    df_clones: np.ndarray,
    ratio: float = 1.0,
    cutoff_timepoints: int = 2,
    celltypes: List[str] = ["Neutrophil", "Monocyte", "Baso", "Mast", "Meg"],
) -> Tuple[List[int], pd.DataFrame]:
    """
    Select clones that span multiple timepoints and differentiate into specified cell
    types.

    Args:
        df_metadata: DataFrame containing cell metadata
        df_clones: Binary matrix of clone assignments
        ratio: Minimum ratio of cells in a clone that must be of the same cell type
        cutoff_timepoints: Minimum number of timepoints a clone must span
        celltypes: List of cell types to consider

    Returns:
        Tuple containing:
            - List of selected clone indices
            - DataFrame mapping clone indices to cell types
    """
    ids = np.where(df_clones)
    df_tags = pd.DataFrame(data=ids[1], index=ids[0], columns=["Tag_0"])

    clones_selected = list()
    clones_truth = pd.DataFrame(columns=["celltype"])

    for x in np.sort(df_tags["Tag_0"].unique()):
        cells_x = df_tags["Tag_0"][df_tags["Tag_0"] == x].index
        n_timepoints_x = len(
            df_metadata.iloc[
                cells_x, df_metadata.columns.get_loc("time_info")
            ].unique()
        )

        if n_timepoints_x > cutoff_timepoints:
            cells_x_selected = cells_x[
                df_metadata.iloc[
                    cells_x, df_metadata.columns.get_loc("time_info")
                ]
                == 6
            ]
            list_anno_x = df_metadata.iloc[
                cells_x_selected, df_metadata.columns.get_loc("state_info")
            ].tolist()

            if len(list_anno_x) > 0:
                celltype = max(set(list_anno_x), key=list_anno_x.count)
                pct_celltype = float(list_anno_x.count(celltype)) / len(
                    list_anno_x
                )

                if (celltype in celltypes) and (pct_celltype >= ratio):
                    clones_selected.append(int(x))
                    clones_truth.loc[x,] = celltype

    return clones_selected, clones_truth


@beartype
def extract_clonal_subpopulation(
    adata: AnnData,
    cell_type: Optional[str] = None,
    cell_types: Optional[List[str]] = None,
    ratio: float = 1.0,
    cutoff_timepoints: int = 2,
) -> AnnData:
    """
    Extract cells belonging to clones that differentiate into specified cell type(s).

    Args:
        adata: AnnData object with clone information in .obsm['X_clone']
        cell_type: Single cell type to extract (e.g., 'Neutrophil')
        cell_types: List of cell types to extract (e.g., ['Neutrophil', 'Monocyte'])
        ratio: Minimum ratio of cells in a clone that must be of the same cell type
        cutoff_timepoints: Minimum number of timepoints a clone must span

    Returns:
        AnnData object containing only cells from the selected clones
    """
    if cell_type is not None and cell_types is not None:
        raise ValueError("Specify either cell_type or cell_types, not both")

    if cell_type is not None:
        cell_types = [cell_type]
    elif cell_types is None:
        raise ValueError("Must specify either cell_type or cell_types")

    clone_matrix = adata.obsm["X_clone"]
    if issparse(clone_matrix):
        clone_matrix = clone_matrix.toarray()

    clones_selected, _ = select_clones(
        adata.obs,
        clone_matrix,
        ratio=ratio,
        cutoff_timepoints=cutoff_timepoints,
        celltypes=cell_types,
    )

    adata_filtered = adata.copy()
    adata_filtered.obsm["X_clone"] = clone_matrix[:, clones_selected]

    # Keep only cells that belong to at least one selected clone
    id_cells = np.where(adata_filtered.obsm["X_clone"].sum(axis=1) > 0)[0]
    adata_filtered = adata_filtered[id_cells, :]

    logger.info(
        f"Extracted {adata_filtered.n_obs} cells from {len(clones_selected)} clones"
    )

    if cell_types:
        logger.info(
            f"Extracting cells from clones differentiated into: {', '.join(cell_types)}"
        )

    return adata_filtered


@beartype
def create_larry_subpopulations(
    output_dir: str = "data/external"
) -> Tuple[AnnData, AnnData, AnnData]:
    """
    Create and save the larry_neu, larry_mono, and larry_multilineage datasets.

    Args:
        output_dir: Directory to save the generated datasets

    Returns:
        Tuple containing (larry_neu, larry_mono, larry_multilineage) AnnData objects
    """
    import os

    from pyrovelocity.io.datasets import larry

    adata_larry = larry()

    adata_neu = extract_clonal_subpopulation(
        adata_larry, cell_type="Neutrophil", ratio=1.0, cutoff_timepoints=2
    )

    adata_mono = extract_clonal_subpopulation(
        adata_larry, cell_type="Monocyte", ratio=1.0, cutoff_timepoints=2
    )

    adata_multi = adata_neu.concatenate(adata_mono)

    os.makedirs(output_dir, exist_ok=True)
    adata_neu.write(os.path.join(output_dir, "larry_neu.h5ad"))
    adata_mono.write(os.path.join(output_dir, "larry_mono.h5ad"))
    adata_multi.write(os.path.join(output_dir, "larry_multilineage.h5ad"))

    return adata_neu, adata_mono, adata_multi
