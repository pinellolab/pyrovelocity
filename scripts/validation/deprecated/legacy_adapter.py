#!/usr/bin/env python
"""
Legacy Adapter for PyroVelocity.

This module provides adapter functions to bridge the architectural differences
between the legacy and modular implementations of PyroVelocity.
"""

import numpy as np
import torch
from anndata import AnnData
from beartype import beartype
from typing import Dict, Any, Optional, Union

from pyrovelocity.models._velocity import PyroVelocity


@beartype
def get_velocity_from_legacy_model(
    model: PyroVelocity,
    posterior_samples: Dict[str, Any],
    adata: Optional[AnnData] = None,
) -> np.ndarray:
    """
    Extract velocity from a legacy PyroVelocity model.

    This function computes velocity from posterior samples using the legacy model's
    workflow and returns it as a numpy array. It first calls compute_statistics_from_posterior_samples
    to compute velocity and store it in the AnnData object, then extracts it from adata.layers["velocity_pyro"].

    Args:
        model: Legacy PyroVelocity model
        posterior_samples: Posterior samples from the model
        adata: AnnData object (if None, uses model.adata)

    Returns:
        Velocity as a numpy array
    """
    # Use model's AnnData if none provided
    if adata is None:
        adata = model.adata

    # Compute statistics from posterior samples (this computes velocity and stores it in adata)
    model.compute_statistics_from_posterior_samples(
        adata=adata,
        posterior_samples=posterior_samples,
        vector_field_basis="umap",
        ncpus_use=1,
        random_seed=99
    )

    # Extract velocity from adata.layers["velocity_pyro"]
    if "velocity_pyro" in adata.layers:
        return adata.layers["velocity_pyro"]
    else:
        # If velocity_pyro is not in adata.layers, compute it manually
        if ('u_scale' in posterior_samples) and ('s_scale' in posterior_samples):
            scale = posterior_samples["u_scale"] / posterior_samples["s_scale"]
        elif ('u_scale' in posterior_samples) and not ('s_scale' in posterior_samples):
            scale = posterior_samples["u_scale"]
        else:
            scale = 1

        velocity = (
            posterior_samples["beta"] * posterior_samples["ut"] / scale
            - posterior_samples["gamma"] * posterior_samples["st"]
        ).mean(0)

        return velocity


@beartype
def get_velocity_uncertainty_from_legacy_model(
    model: PyroVelocity,
    posterior_samples: Dict[str, Any],
    adata: Optional[AnnData] = None,
) -> np.ndarray:
    """
    Extract velocity uncertainty from a legacy PyroVelocity model.

    This function computes velocity uncertainty from posterior samples using the legacy model's
    workflow and returns it as a numpy array. It uses the FDR values computed during
    vector_field_uncertainty as a proxy for uncertainty, or computes standard deviation
    across posterior samples if FDR values are not available.

    Args:
        model: Legacy PyroVelocity model
        posterior_samples: Posterior samples from the model
        adata: AnnData object (if None, uses model.adata)

    Returns:
        Velocity uncertainty as a numpy array
    """
    # Use model's AnnData if none provided
    if adata is None:
        adata = model.adata

    # Check if FDR values are available in posterior_samples
    if 'fdri' in posterior_samples:
        return posterior_samples['fdri']

    # If FDR values are not available, compute standard deviation across posterior samples
    if ('u_scale' in posterior_samples) and ('s_scale' in posterior_samples):
        scale = posterior_samples["u_scale"] / posterior_samples["s_scale"]
    elif ('u_scale' in posterior_samples) and not ('s_scale' in posterior_samples):
        scale = posterior_samples["u_scale"]
    else:
        scale = 1

    velocity_samples = (
        posterior_samples["beta"] * posterior_samples["ut"] / scale
        - posterior_samples["gamma"] * posterior_samples["st"]
    )

    return np.std(velocity_samples, axis=0)


@beartype
def convert_legacy_posterior_samples_to_modular_format(
    posterior_samples: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Convert legacy posterior samples to modular format.

    This function converts posterior samples from the legacy format to the modular format,
    ensuring compatibility with the modular implementation's functions.

    Args:
        posterior_samples: Posterior samples in legacy format

    Returns:
        Posterior samples in modular format
    """
    # Create a copy of the posterior samples to avoid modifying the original
    modular_samples = posterior_samples.copy()

    # Convert numpy arrays to torch tensors if needed
    for key, value in modular_samples.items():
        if isinstance(value, np.ndarray) and not isinstance(value, torch.Tensor):
            modular_samples[key] = torch.tensor(value)

    return modular_samples
