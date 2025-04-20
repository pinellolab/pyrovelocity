"""
Preprocessing utilities for PyroVelocity JAX/NumPyro implementation.

This module contains minimal utilities for data preprocessing required within
the JAX model context. Major preprocessing steps (normalization, filtering,
log-transform) should be handled upstream, primarily within
`src/pyrovelocity/tasks/preprocess.py`.
"""

from typing import Dict, Tuple, Optional, Any, List, Union
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float
from beartype import beartype

@beartype
def normalize_counts(
    counts: jnp.ndarray,
    size_factors: Optional[jnp.ndarray] = None,
    log_transform: bool = True,
    pseudocount: float = 1.0,
) -> jnp.ndarray:
    """Normalize counts using size factors.
    
    Args:
        counts: Count data
        size_factors: Size factors for normalization
        log_transform: Whether to log-transform the data
        pseudocount: Pseudocount to add before log-transform
        
    Returns:
        Normalized counts
    """
    # If size factors are not provided, compute them
    if size_factors is None:
        size_factors = compute_size_factors(counts)
    
    # Reshape size factors for broadcasting
    # If counts has shape (cells, genes), size_factors should have shape (cells,)
    # We need to reshape to (cells, 1) for proper broadcasting
    if counts.ndim > 1 and size_factors.ndim == 1:
        size_factors = size_factors.reshape(-1, *([1] * (counts.ndim - 1)))
    
    # Normalize counts by size factors
    normalized = counts / size_factors
    
    # Apply log transform if requested
    if log_transform:
        normalized = jnp.log(normalized + pseudocount)
    
    return normalized

@beartype
def compute_size_factors(
    counts: jnp.ndarray,
    axis: int = 1,
) -> jnp.ndarray:
    """Compute size factors for normalization.
    
    Args:
        counts: Count data
        axis: Axis along which to compute size factors
        
    Returns:
        Size factors
    """
    # Compute the sum of counts along the specified axis
    total_counts = jnp.sum(counts, axis=axis)
    
    # Handle the case where total_counts contains zeros
    # Replace zeros with ones to avoid division by zero
    total_counts = jnp.where(total_counts > 0, total_counts, 1.0)
    
    # Compute the geometric mean of total counts
    # Use the mean of log counts to avoid numerical issues
    log_counts = jnp.log(jnp.maximum(total_counts, 1e-6))
    geo_mean = jnp.exp(jnp.mean(log_counts))
    
    # Compute size factors as the ratio of total counts to geometric mean
    size_factors = total_counts / geo_mean
    
    return size_factors

@beartype
def filter_genes(
    u_counts: jnp.ndarray,
    s_counts: jnp.ndarray,
    min_counts: int = 10,
    min_cells: int = 5,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Filter genes based on minimum counts and cells.
    
    Args:
        u_counts: Unspliced count data
        s_counts: Spliced count data
        min_counts: Minimum counts per gene
        min_cells: Minimum cells per gene
        
    Returns:
        Tuple of (filtered_u_counts, filtered_s_counts, gene_indices)
    """
    # Count the number of cells with at least min_counts for each gene
    u_cells_per_gene = jnp.sum(u_counts >= min_counts, axis=0)
    s_cells_per_gene = jnp.sum(s_counts >= min_counts, axis=0)
    
    # Keep genes that have at least min_cells in both unspliced and spliced data
    gene_mask = (u_cells_per_gene >= min_cells) & (s_cells_per_gene >= min_cells)
    gene_indices = jnp.where(gene_mask)[0]
    
    # Filter the count matrices
    u_counts_filtered = u_counts[:, gene_mask]
    s_counts_filtered = s_counts[:, gene_mask]
    
    return u_counts_filtered, s_counts_filtered, gene_indices

@beartype
def _internal_transform(
    data: Dict[str, jnp.ndarray],
) -> Dict[str, jnp.ndarray]:
    """Apply internal transformations needed by the model.
    
    This function applies minimal transformations needed by the model,
    such as ensuring non-negative values or handling special cases.
    
    Args:
        data: Dictionary of data arrays
        
    Returns:
        Dictionary of transformed data arrays
    """
    # Create a copy of the data dictionary
    transformed_data = {}
    
    # Apply transformations to each array in the dictionary
    for key, array in data.items():
        # For count matrices, ensure non-negative values
        if key.startswith("X_") or key in ["X_spliced", "X_unspliced"]:
            transformed_data[key] = jnp.maximum(array, 0)
        else:
            # For other arrays, keep them as is
            transformed_data[key] = array
    
    return transformed_data