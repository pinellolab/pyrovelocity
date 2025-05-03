"""
PyroVelocity modular component implementations.

This package contains the implementations of the various components used in the
PyroVelocity modular architecture, including dynamics models, prior models,
likelihood models, observation models, and inference guides.

The package includes both base class implementations and Protocol-First implementations:
- Base class implementations: Components that inherit from base classes in base.py
- Protocol-First implementations: Components that directly implement Protocol interfaces
"""

# Import base classes
from pyrovelocity.models.modular.components.base import (
    BaseDynamicsModel,
    BaseInferenceGuide,
    BaseLikelihoodModel,
    BaseObservationModel,
    BasePriorModel,
)

# Import Protocol-First implementations
from pyrovelocity.models.modular.components.direct.dynamics import (
    NonlinearDynamicsModelDirect,
    StandardDynamicsModelDirect,
)
from pyrovelocity.models.modular.components.direct.guides import (
    AutoGuideFactoryDirect,
    DeltaGuideDirect,
    NormalGuideDirect,
)
from pyrovelocity.models.modular.components.direct.likelihoods import (
    NegativeBinomialLikelihoodModelDirect,
    PoissonLikelihoodModelDirect,
)
from pyrovelocity.models.modular.components.direct.observations import (
    StandardObservationModelDirect,
)
from pyrovelocity.models.modular.components.direct.priors import (
    InformativePriorModelDirect,
    LogNormalPriorModelDirect,
)

# Import base class implementations
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
    # Base classes
    "BaseDynamicsModel",
    "BaseLikelihoodModel",
    "BaseObservationModel",
    "BasePriorModel",
    "BaseInferenceGuide",
    # Base class implementations
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
    # Protocol-First implementations
    "StandardDynamicsModelDirect",
    "NonlinearDynamicsModelDirect",
    "LogNormalPriorModelDirect",
    "InformativePriorModelDirect",
    "PoissonLikelihoodModelDirect",
    "NegativeBinomialLikelihoodModelDirect",
    "StandardObservationModelDirect",
    "AutoGuideFactoryDirect",
    "NormalGuideDirect",
    "DeltaGuideDirect",
]
