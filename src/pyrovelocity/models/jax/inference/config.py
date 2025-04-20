"""
Inference configuration utilities for PyroVelocity JAX/NumPyro implementation.

This module contains inference configuration utilities, including:

- InferenceConfig: Configuration dataclass for inference methods
- create_inference_config: Factory function for creating configurations
- validate_config: Validate inference configuration
"""

from typing import Dict, Tuple, Optional, Any, List, Union
import jax
import jax.numpy as jnp
import numpyro
from jaxtyping import Array, Float
from beartype import beartype

from pyrovelocity.models.jax.core.state import InferenceConfig

@beartype
def create_inference_config(
    method: str = "svi",
    **kwargs
) -> InferenceConfig:
    """Factory function for creating inference configurations.
    
    Args:
        method: Inference method ("svi" or "mcmc")
        **kwargs: Additional configuration parameters
        
    Returns:
        InferenceConfig object
    """
    # Create default config based on method
    if method == "svi":
        config = InferenceConfig(
            method="svi",
            num_samples=1000,
            guide_type="auto_normal",
            optimizer="adam",
            learning_rate=0.01,
            num_epochs=1000,
            batch_size=None,
            clip_norm=None,
            early_stopping=True,
            early_stopping_patience=10,
        )
    elif method == "mcmc":
        config = InferenceConfig(
            method="mcmc",
            num_samples=1000,
            num_warmup=500,
            num_chains=1,
        )
    else:
        raise ValueError(f"Unknown inference method: {method}")
    
    # Update config with kwargs
    return config.replace(**kwargs)

@beartype
def validate_config(config: InferenceConfig) -> bool:
    """Validate inference configuration.
    
    Args:
        config: InferenceConfig object
        
    Returns:
        True if the configuration is valid, False otherwise
    """
    # Placeholder for future implementation
    raise NotImplementedError("This function will be implemented in a future phase.")

@beartype
def get_default_config(method: str = "svi") -> InferenceConfig:
    """Get default inference configuration.
    
    Args:
        method: Inference method ("svi" or "mcmc")
        
    Returns:
        Default InferenceConfig object
    """
    return create_inference_config(method=method)