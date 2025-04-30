"""
PyroVelocity validation framework.

This module provides tools for validating and comparing different implementations
of PyroVelocity (legacy, modular, and JAX).
"""

from pyrovelocity.validation.framework import (
    ValidationRunner,
    run_validation,
    compare_implementations,
)
from pyrovelocity.validation.metrics import (
    compute_parameter_metrics,
    compute_velocity_metrics,
    compute_uncertainty_metrics,
    compute_performance_metrics,
)
from pyrovelocity.validation.comparison import (
    compare_parameters,
    compare_velocities,
    compare_uncertainties,
    compare_performance,
    statistical_comparison,
)
from pyrovelocity.validation.visualization import (
    plot_parameter_comparison,
    plot_velocity_comparison,
    plot_uncertainty_comparison,
    plot_performance_comparison,
)

__all__ = [
    # Framework
    "ValidationRunner",
    "run_validation",
    "compare_implementations",
    # Metrics
    "compute_parameter_metrics",
    "compute_velocity_metrics",
    "compute_uncertainty_metrics",
    "compute_performance_metrics",
    # Comparison
    "compare_parameters",
    "compare_velocities",
    "compare_uncertainties",
    "compare_performance",
    "statistical_comparison",
    # Visualization
    "plot_parameter_comparison",
    "plot_velocity_comparison",
    "plot_uncertainty_comparison",
    "plot_performance_comparison",
]
