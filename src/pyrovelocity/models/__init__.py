"""PyroVelocity models package.

This package contains the modular components of the PyroVelocity model architecture,
including factory methods for model creation and configuration management,
direct AnnData integration, and Bayesian model comparison tools.

This package has been simplified to include only the essential components needed for
validation against the legacy implementation.
"""

# Import from modular components
# Import the legacy PyroVelocity model for backward compatibility
from pyrovelocity.models._velocity import PyroVelocity

# Import experimental implementations
from pyrovelocity.models.experimental import (
    deterministic_transcription_splicing_probabilistic_model,
    generate_posterior_inference_data,
    generate_prior_inference_data,
    generate_test_data_for_deterministic_model_inference,
    lognormal_tail_probability,
    plot_sample_phase_portraits,
    plot_sample_trajectories,
    plot_sample_trajectories_with_percentiles,
    save_inference_plots,
    solve_for_lognormal_mu_given_threshold_and_tail_mass,
    solve_for_lognormal_sigma_given_threshold_and_tail_mass,
    solve_transcription_splicing_model,
    solve_transcription_splicing_model_analytical,
)

# Import the model comparison classes and functions
from pyrovelocity.models.modular.comparison import (
    BayesianModelComparison,
    ComparisonResult,
    create_comparison_table,
    select_best_model,
)

# Base classes have been removed in favor of Protocol interfaces
from pyrovelocity.models.modular.components.dynamics import (
    LegacyDynamicsModel,
    StandardDynamicsModel,
)
from pyrovelocity.models.modular.components.guides import (
    AutoGuideFactory,
    LegacyAutoGuideFactory,
)
from pyrovelocity.models.modular.components.likelihoods import (
    LegacyLikelihoodModel,
    PoissonLikelihoodModel,
)
from pyrovelocity.models.modular.components.observations import (
    StandardObservationModel,
)
from pyrovelocity.models.modular.components.priors import (
    LogNormalPriorModel,
)

# Import the factory module for model creation and configuration
from pyrovelocity.models.modular.factory import (
    DynamicsModelConfig,
    InferenceGuideConfig,
    LikelihoodModelConfig,
    ObservationModelConfig,
    PriorModelConfig,
    PyroVelocityModelConfig,
    create_dynamics_model,
    create_inference_guide,
    create_likelihood_model,
    create_model,
    create_observation_model,
    create_prior_model,
    create_standard_model,
    standard_model_config,
)
from pyrovelocity.models.modular.interfaces import (
    DynamicsModel,
    InferenceGuide,
    LikelihoodModel,
    ObservationModel,
    PriorModel,
)

# Import the new PyroVelocityModel and ModelState
from pyrovelocity.models.modular.model import ModelState, PyroVelocityModel
from pyrovelocity.models.modular.registry import (
    DynamicsModelRegistry,
    InferenceGuideRegistry,
    LikelihoodModelRegistry,
    ObservationModelRegistry,
    PriorModelRegistry,
)

# Import the model selection classes and functions
from pyrovelocity.models.modular.selection import (
    CrossValidator,
    ModelEnsemble,
    ModelSelection,
    SelectionCriterion,
    SelectionResult,
)

__all__ = [
    # Interfaces
    "DynamicsModel",
    "LikelihoodModel",
    "ObservationModel",
    "PriorModel",
    "InferenceGuide",
    # Registries
    "DynamicsModelRegistry",
    "LikelihoodModelRegistry",
    "ObservationModelRegistry",
    "PriorModelRegistry",
    "InferenceGuideRegistry",
    # Dynamics models
    "StandardDynamicsModel",
    "LegacyDynamicsModel",
    # Prior models
    "LogNormalPriorModel",
    # Likelihood models
    "PoissonLikelihoodModel",
    "LegacyLikelihoodModel",
    # Observation models
    "StandardObservationModel",
    # Inference guides
    "AutoGuideFactory",
    "LegacyAutoGuideFactory",
    # Legacy model
    "PyroVelocity",
    # New model
    "ModelState",
    "PyroVelocityModel",

    # Factory module
    "DynamicsModelConfig",
    "PriorModelConfig",
    "LikelihoodModelConfig",
    "ObservationModelConfig",
    "InferenceGuideConfig",
    "PyroVelocityModelConfig",
    "create_dynamics_model",
    "create_prior_model",
    "create_likelihood_model",
    "create_observation_model",
    "create_inference_guide",
    "create_model",
    "standard_model_config",
    "create_standard_model",
    # Model comparison
    "BayesianModelComparison",
    "ComparisonResult",
    "create_comparison_table",
    "select_best_model",
    # Model selection
    "ModelSelection",
    "ModelEnsemble",
    "CrossValidator",
    "SelectionCriterion",
    "SelectionResult",
    # Experimental implementations
    "deterministic_transcription_splicing_probabilistic_model",
    "generate_test_data_for_deterministic_model_inference",
    "generate_prior_inference_data",
    "generate_posterior_inference_data",
    "plot_sample_phase_portraits",
    "plot_sample_trajectories",
    "plot_sample_trajectories_with_percentiles",
    "save_inference_plots",
    "solve_transcription_splicing_model",
    "solve_transcription_splicing_model_analytical",
    "lognormal_tail_probability",
    "solve_for_lognormal_sigma_given_threshold_and_tail_mass",
    "solve_for_lognormal_mu_given_threshold_and_tail_mass",
]
