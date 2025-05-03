"""
Protocol-First component implementations for PyroVelocity's modular architecture.

This package contains component implementations that directly implement the
Protocol interfaces without inheriting from base classes. These implementations
follow the Protocol-First approach, which embraces composition over inheritance
and allows for more flexible component composition.

The Protocol-First approach has several advantages:
1. Reduces code complexity by eliminating inheritance hierarchies
2. Enhances flexibility through Protocol interfaces
3. Creates perfect architectural consistency with the JAX implementation's pure functional approach
4. Allows for the discovery of natural abstractions through actual usage patterns
5. Avoids premature abstraction by initially allowing intentional duplication

Each component implementation in this package:
1. Directly implements the corresponding Protocol interface
2. Uses utility functions from the utils package for common functionality
3. Allows intentional duplication of non-critical functionality
4. Includes clear documentation of the implementation approach
"""

from pyrovelocity.models.modular.components.direct.dynamics import StandardDynamicsModelDirect
from pyrovelocity.models.modular.components.direct.priors import LogNormalPriorModelDirect
from pyrovelocity.models.modular.components.direct.likelihoods import PoissonLikelihoodModelDirect
from pyrovelocity.models.modular.components.direct.observations import StandardObservationModelDirect
from pyrovelocity.models.modular.components.direct.guides import AutoGuideFactoryDirect

__all__ = [
    "StandardDynamicsModelDirect",
    "LogNormalPriorModelDirect",
    "PoissonLikelihoodModelDirect",
    "StandardObservationModelDirect",
    "AutoGuideFactoryDirect",
]
