"""
PyroVelocity modular component implementations.

This package contains the implementations of the various components used in the
PyroVelocity modular architecture, including dynamics models, prior models,
likelihood models, observation models, and inference guides.

All components directly implement Protocol interfaces defined in interfaces.py,
following a Protocol-First approach that embraces composition over inheritance.
This approach reduces code complexity, enhances flexibility, and creates
architectural consistency with the JAX implementation's pure functional approach.
"""

# Import component implementations
from pyrovelocity.models.modular.components.dynamics import (
    NonlinearDynamicsModel,
    StandardDynamicsModel,
)
from pyrovelocity.models.modular.components.guides import (
    AutoGuideFactory,
    DeltaGuide,
    NormalGuide,
)
from pyrovelocity.models.modular.components.likelihoods import (
    NegativeBinomialLikelihoodModel,
    PoissonLikelihoodModel,
)
from pyrovelocity.models.modular.components.observations import (
    StandardObservationModel,
)
from pyrovelocity.models.modular.components.priors import (
    InformativePriorModel,
    LogNormalPriorModel,
)

__all__ = [
    # Component implementations
    "StandardDynamicsModel",
    "NonlinearDynamicsModel",
    "LogNormalPriorModel",
    "InformativePriorModel",
    "PoissonLikelihoodModel",
    "NegativeBinomialLikelihoodModel",
    "StandardObservationModel",
    "AutoGuideFactory",
    "NormalGuide",
    "DeltaGuide",
]
