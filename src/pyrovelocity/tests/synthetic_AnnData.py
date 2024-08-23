"""Producing synthetic AnnData for tests."""

import numpy as np
import pandas as pd
import anndata as ad

def synthetic_AnnData(
    n_cell_types: int = 3,
    n_genes: int = 10,
    cells_per_type: int = 20,
    seed: int = 42
    ):
    
    """
    Produces a simple synthetic AnnData object.

    Args:
        n_cell_types (int): Number of cell types.
        n_genes (int): Number of genes.
        cells_per_type (int): Number of cells per cell type.
        seed (int): Random seed.
    Returns:
        AnnData: Synthetic AnnData object.

    Examples:
        >>> synthetic_AnnData()
    """

    # Number of genes, cells, and cell types
    n_genes = 10
    n_cells = cells_per_type * n_cell_types

    # Create synthetic gene expression data
    # Each cell type will have slightly different expression profiles
    np.random.seed(seed)  # For reproducibility

    cells_per_type = int(n_cells/n_cell_types)
    # Generate data with different means for different cell types
    expression_data = np.vstack([
        np.random.normal(loc=i, scale=0.5, size=(cells_per_type, n_genes))
        for i in range(n_cell_types)
    ])

    # Create an AnnData object
    adata = ad.AnnData(X=expression_data)

    # Add cell type annotations
    cell_types = []
    for i in range(n_cell_types):
        cell_types += ['Type ' + str(i)] * cells_per_type
    adata.obs['cell_type'] = pd.Categorical(cell_types)

    # Add gene names (e.g., Gene1, Gene2, ..., Gene20)
    gene_names = [f'Gene{i+1}' for i in range(n_genes)]
    adata.var['gene_names'] = gene_names

    # Add cell names (e.g., Cell1, Cell2, ..., Cell30)
    cell_names = [f'Cell{i+1}' for i in range(n_cells)]
    adata.obs_names = cell_names
    adata.var_names = gene_names

    return adata
