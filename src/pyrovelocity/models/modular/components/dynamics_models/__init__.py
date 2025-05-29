"""
Dynamics model components for PyroVelocity's modular architecture.

This module contains specialized dynamics model implementations that extend
the basic dynamics models with more complex mathematical formulations.
"""

# Import new piecewise dynamics model
from pyrovelocity.models.modular.components.dynamics_models.piecewise import (
    PiecewiseActivationDynamicsModel,
)

__all__ = [
    # New models
    "PiecewiseActivationDynamicsModel",
]
