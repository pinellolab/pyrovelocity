"""
PyroVelocity modular architecture package.

This package contains the modular architecture for PyroVelocity, enabling
flexible composition of models, priors, likelihoods, and inference methods.

This package has been simplified to include only the essential components needed for
validation against the legacy implementation.
"""

# Import component registries and register components
# Import model class

# Import comparison and selection
from pyrovelocity.models.modular.comparison import (
    BayesianModelComparison,
    compute_loo,
    compute_waic,
)

# Import component implementations
from pyrovelocity.models.modular.components import (
    AutoGuideFactory,
    LegacyAutoGuideFactory,
    LegacyDynamicsModel,
    LegacyLikelihoodModel,
    LogNormalPriorModel,
    PoissonLikelihoodModel,
    StandardDynamicsModel,
    StandardObservationModel,
)

# Import data utilities
from pyrovelocity.models.modular.data.anndata import (
    extract_layers,
    get_library_size,
    prepare_anndata,
    store_results,
)

# Import factory functions
from pyrovelocity.models.modular.factory import (
    create_model,
    create_model_from_config,
    create_standard_model,
    standard_model_config,
)
from pyrovelocity.models.modular.model import ModelState, PyroVelocityModel
from pyrovelocity.models.modular.registry import (
    DynamicsModelRegistry,
    InferenceGuideRegistry,
    LikelihoodModelRegistry,
    ObservationModelRegistry,
    PriorModelRegistry,
)
from pyrovelocity.models.modular.selection import (
    ModelEnsemble,
    select_model,
)


# Ensure components are registered in registries
def _ensure_registrations():
    """
    Ensure all components are properly registered in their respective registries.
    This function is called when the module is imported.
    """
    # Import and call register_standard_components to ensure all components are registered
    from pyrovelocity.models.modular.registry import (
        register_standard_components,
    )
    register_standard_components()


# Call the function to ensure registrations
_ensure_registrations()

__all__ = [
    # Model class
    "ModelState",
    "PyroVelocityModel",
    # Data utilities
    "prepare_anndata",
    "extract_layers",
    "store_results",
    "get_library_size",
    # Component registries
    "DynamicsModelRegistry",
    "PriorModelRegistry",
    "LikelihoodModelRegistry",
    "ObservationModelRegistry",
    "InferenceGuideRegistry",
    # Component implementations
    "StandardDynamicsModel",
    "LegacyDynamicsModel",
    "LogNormalPriorModel",
    "PoissonLikelihoodModel",
    "LegacyLikelihoodModel",
    "StandardObservationModel",
    "AutoGuideFactory",
    "LegacyAutoGuideFactory",
    # Factory functions
    "create_model",
    "create_model_from_config",
    "create_standard_model",
    "standard_model_config",
    # Comparison and selection
    "BayesianModelComparison",
    "compute_waic",
    "compute_loo",
    "select_model",
    "ModelEnsemble",
]
