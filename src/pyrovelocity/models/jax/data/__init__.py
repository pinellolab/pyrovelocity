"""
Data processing utilities for PyroVelocity JAX/NumPyro implementation.

This module contains utilities for data processing, including AnnData integration,
batch processing, and preprocessing.
"""

from pyrovelocity.models.jax.data.anndata import (
    prepare_anndata,
    extract_layers,
    store_results,
    get_library_size,
)

from pyrovelocity.models.jax.data.batch import (
    random_batch_indices,
    create_batch_iterator,
    batch_data,
    vmap_batch_function,
)

from pyrovelocity.models.jax.data.preprocessing import (
    normalize_counts,
    compute_size_factors,
    filter_genes,
    _internal_transform,
)

__all__ = [
    # AnnData integration
    "prepare_anndata",
    "extract_layers",
    "store_results",
    "get_library_size",
    
    # Batch processing
    "random_batch_indices",
    "create_batch_iterator",
    "batch_data",
    "vmap_batch_function",
    
    # Preprocessing
    "normalize_counts",
    "compute_size_factors",
    "filter_genes",
    "_internal_transform",
]