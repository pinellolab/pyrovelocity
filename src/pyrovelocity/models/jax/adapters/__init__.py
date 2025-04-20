"""
Adapter utilities for PyroVelocity JAX/NumPyro implementation.

This module contains utilities for converting between PyTorch/Pyro and JAX/NumPyro,
including tensor conversion and model state conversion.
"""

from pyrovelocity.models.jax.adapters.torch_to_jax import (
    convert_tensor as convert_tensor_to_jax,
    convert_parameters as convert_parameters_to_jax,
    convert_model_state as convert_model_state_to_jax,
    convert_pyro_to_numpyro_model,
    convert_pyro_to_numpyro_guide,
    convert_pyro_to_numpyro_posterior,
)

from pyrovelocity.models.jax.adapters.jax_to_torch import (
    convert_array as convert_array_to_torch,
    convert_parameters as convert_parameters_to_torch,
    convert_model_state as convert_model_state_to_torch,
    convert_numpyro_to_pyro_model,
    convert_numpyro_to_pyro_guide,
    convert_numpyro_to_pyro_posterior,
)

__all__ = [
    # PyTorch to JAX
    "convert_tensor_to_jax",
    "convert_parameters_to_jax",
    "convert_model_state_to_jax",
    "convert_pyro_to_numpyro_model",
    "convert_pyro_to_numpyro_guide",
    "convert_pyro_to_numpyro_posterior",
    
    # JAX to PyTorch
    "convert_array_to_torch",
    "convert_parameters_to_torch",
    "convert_model_state_to_torch",
    "convert_numpyro_to_pyro_model",
    "convert_numpyro_to_pyro_guide",
    "convert_numpyro_to_pyro_posterior",
]