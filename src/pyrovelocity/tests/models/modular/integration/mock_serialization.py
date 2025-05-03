"""
Mock serialization module for end-to-end tests.
"""

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from beartype import beartype
from beartype.typing import Dict, Optional, Union

def load_anndata_from_json(
    filename: Union[str, Path],
    expected_hash: Optional[str] = None,
) -> ad.AnnData:
    """
    Mock function to load AnnData from JSON for testing.
    
    Instead of loading from a file, this creates a synthetic AnnData object
    with the right dimensions and structure for testing.
    
    Args:
        filename: Path to the JSON file (ignored in this mock)
        expected_hash: Expected hash of the file (ignored in this mock)
        
    Returns:
        A synthetic AnnData object for testing
    """
    # Create a synthetic AnnData object for testing
    n_cells, n_genes = 50, 7
    
    # Create random data
    X = np.random.poisson(5, size=(n_cells, n_genes))
    u_data = np.random.poisson(5, size=(n_cells, n_genes))
    s_data = np.random.poisson(5, size=(n_cells, n_genes))
    
    # Create AnnData object
    adata = ad.AnnData(X=s_data)
    adata.layers["spliced"] = s_data
    adata.layers["unspliced"] = u_data
    
    # Add cell and gene names
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]
    
    # Add cluster information
    adata.obs["clusters"] = np.random.choice(["A", "B", "C"], size=n_cells)
    
    # Add UMAP coordinates for visualization
    adata.obsm = {}
    adata.obsm["X_umap"] = np.random.normal(0, 1, size=(n_cells, 2))
    
    # Add library size information
    adata.obs["u_lib_size_raw"] = u_data.sum(axis=1)
    adata.obs["s_lib_size_raw"] = s_data.sum(axis=1)
    adata.obs["u_lib_size"] = np.log(adata.obs["u_lib_size_raw"].astype(float) + 1e-6)
    adata.obs["s_lib_size"] = np.log(adata.obs["s_lib_size_raw"].astype(float) + 1e-6)
    
    # Add library size statistics
    adata.obs["u_lib_size_mean"] = adata.obs["u_lib_size"].mean()
    adata.obs["s_lib_size_mean"] = adata.obs["s_lib_size"].mean()
    adata.obs["u_lib_size_scale"] = adata.obs["u_lib_size"].std()
    adata.obs["s_lib_size_scale"] = adata.obs["s_lib_size"].std()
    
    # Add indices for batch processing
    adata.obs["ind_x"] = np.arange(adata.n_obs)
    
    return adata
