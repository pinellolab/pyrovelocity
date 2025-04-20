"""
AnnData integration utilities for PyroVelocity JAX/NumPyro implementation.

This module contains utilities for AnnData integration, including:

- prepare_anndata: Convert AnnData to JAX arrays
- extract_layers: Extract layers from AnnData
- store_results: Store results in AnnData
"""

from typing import Dict, Tuple, Optional, Any, List, Union
import jax
import jax.numpy as jnp
import numpy as np
import anndata
from jaxtyping import Array, Float
from beartype import beartype

@beartype
def prepare_anndata(
    adata: anndata.AnnData,
    spliced_layer: str = "spliced",
    unspliced_layer: str = "unspliced",
    use_raw: bool = False,
) -> Dict[str, jnp.ndarray]:
    """Convert AnnData to JAX arrays.
    
    Args:
        adata: AnnData object
        spliced_layer: Name of the spliced layer
        unspliced_layer: Name of the unspliced layer
        use_raw: Whether to use raw data
        
    Returns:
        Dictionary of JAX arrays
    """
    # Extract layers
    u, s = extract_layers(adata, spliced_layer, unspliced_layer, use_raw)
    
    # Extract metadata
    cell_types = None
    if "cell_type" in adata.obs:
        # Check if cell_type is categorical
        if hasattr(adata.obs["cell_type"], "cat") and hasattr(adata.obs["cell_type"].cat, "categories"):
            cell_types = jnp.array([adata.obs["cell_type"].cat.categories.get_loc(ct)
                                  for ct in adata.obs["cell_type"]])
        else:
            # If not categorical, convert to numerical representation
            unique_types = list(set(adata.obs["cell_type"]))
            cell_types = jnp.array([unique_types.index(ct) for ct in adata.obs["cell_type"]])
    elif "clusters" in adata.obs:
        # Check if clusters is categorical
        if hasattr(adata.obs["clusters"], "cat") and hasattr(adata.obs["clusters"].cat, "categories"):
            cell_types = jnp.array([adata.obs["clusters"].cat.categories.get_loc(ct)
                                  for ct in adata.obs["clusters"]])
        else:
            # If not categorical, convert to numerical representation
            unique_clusters = list(set(adata.obs["clusters"]))
            cell_types = jnp.array([unique_clusters.index(ct) for ct in adata.obs["clusters"]])
    
    gene_names = jnp.array(range(adata.n_vars))
    if "gene_name" in adata.var:
        gene_names = jnp.array([i for i in range(adata.n_vars)])
    
    # Create dictionary of JAX arrays
    data_dict = {
        "X_unspliced": u,
        "X_spliced": s,
        "cell_types": cell_types,
        "gene_names": gene_names,
    }
    
    # Add library sizes
    u_lib_size, s_lib_size = get_library_size(adata, spliced_layer, unspliced_layer, use_raw)
    data_dict["u_lib_size"] = u_lib_size
    data_dict["s_lib_size"] = s_lib_size
    
    return data_dict

@beartype
def extract_layers(
    adata: anndata.AnnData,
    spliced_layer: str = "spliced",
    unspliced_layer: str = "unspliced",
    use_raw: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Extract layers from AnnData.
    
    Args:
        adata: AnnData object
        spliced_layer: Name of the spliced layer
        unspliced_layer: Name of the unspliced layer
        use_raw: Whether to use raw data
        
    Returns:
        Tuple of (unspliced, spliced) JAX arrays
    """
    if use_raw and adata.raw is not None:
        if spliced_layer in adata.raw.layers:
            s = jnp.array(adata.raw.layers[spliced_layer].toarray())
        else:
            s = jnp.array(adata.raw.X.toarray())
        
        if unspliced_layer in adata.raw.layers:
            u = jnp.array(adata.raw.layers[unspliced_layer].toarray())
        else:
            raise ValueError(f"Layer '{unspliced_layer}' not found in adata.raw.layers")
    else:
        if spliced_layer in adata.layers:
            s = jnp.array(adata.layers[spliced_layer].toarray()
                         if hasattr(adata.layers[spliced_layer], "toarray")
                         else adata.layers[spliced_layer])
        else:
            s = jnp.array(adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X)
        
        if unspliced_layer in adata.layers:
            u = jnp.array(adata.layers[unspliced_layer].toarray()
                         if hasattr(adata.layers[unspliced_layer], "toarray")
                         else adata.layers[unspliced_layer])
        else:
            raise ValueError(f"Layer '{unspliced_layer}' not found in adata.layers")
    
    return u, s

@beartype
def store_results(
    adata: anndata.AnnData,
    results: Dict[str, jnp.ndarray],
    model_name: str = "velocity_model",
) -> anndata.AnnData:
    """Store results in AnnData.
    
    Args:
        adata: AnnData object
        results: Dictionary of results
        model_name: Name of the model
        
    Returns:
        Updated AnnData object
    """
    # Create a copy of the AnnData object to avoid modifying the original
    adata_out = adata.copy()
    
    # Store results in the AnnData object
    for key, value in results.items():
        # Convert JAX arrays to numpy arrays
        value_np = np.array(value)
        
        # Store the result based on its shape
        if value_np.ndim == 2 and value_np.shape[0] == adata.n_obs and value_np.shape[1] == adata.n_vars:
            # Cell x gene matrices go in layers
            adata_out.layers[f"{model_name}_{key}"] = value_np
        elif value_np.ndim == 1 and value_np.shape[0] == adata.n_obs:
            # Cell vectors go in obs
            adata_out.obs[f"{model_name}_{key}"] = value_np
        elif value_np.ndim == 1 and value_np.shape[0] == adata.n_vars:
            # Gene vectors go in var
            adata_out.var[f"{model_name}_{key}"] = value_np
        elif value_np.ndim == 0 or (value_np.ndim == 1 and value_np.shape[0] == 1):
            # Scalars go in uns
            adata_out.uns[f"{model_name}_{key}"] = value_np.item() if value_np.size == 1 else value_np
        else:
            # Other arrays go in uns
            adata_out.uns[f"{model_name}_{key}"] = value_np
    
    return adata_out

@beartype
def get_library_size(
    adata: anndata.AnnData,
    spliced_layer: str = "spliced",
    unspliced_layer: str = "unspliced",
    use_raw: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Get library size for spliced and unspliced data.
    
    Args:
        adata: AnnData object
        spliced_layer: Name of the spliced layer
        unspliced_layer: Name of the unspliced layer
        use_raw: Whether to use raw data
        
    Returns:
        Tuple of (unspliced_lib_size, spliced_lib_size) JAX arrays
    """
    # Extract layers
    u, s = extract_layers(adata, spliced_layer, unspliced_layer, use_raw)
    
    # Calculate library sizes (sum across genes for each cell)
    u_lib_size = jnp.sum(u, axis=1)
    s_lib_size = jnp.sum(s, axis=1)
    
    return u_lib_size, s_lib_size