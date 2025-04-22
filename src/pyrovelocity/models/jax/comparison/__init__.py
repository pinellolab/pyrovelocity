"""
Model comparison utilities for PyroVelocity JAX/NumPyro implementation.
"""

from pyrovelocity.models.jax.comparison.comparison import (
    compare_models,
    compute_log_likelihood,
    compute_loo,
    compute_waic,
)
from pyrovelocity.models.jax.comparison.selection import (
    compute_predictive_performance,
    cross_validate,
    select_best_model,
)

__all__ = [
    "compute_log_likelihood",
    "compute_waic",
    "compute_loo",
    "compare_models",
    "select_best_model",
    "cross_validate",
    "compute_predictive_performance",
]
