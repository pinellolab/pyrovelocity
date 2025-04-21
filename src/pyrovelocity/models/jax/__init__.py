"""
PyroVelocity JAX/NumPyro Implementation.

This module contains the JAX/NumPyro implementation of PyroVelocity, a probabilistic
model for RNA velocity analysis.
"""

from pyrovelocity.models.jax.core import (
    # Utils
    create_key,
    split_key,
    set_platform_device,
    enable_x64,
    disable_x64,
    get_device_count,
    get_devices,
    check_array_shape,
    check_array_dtype,
    ensure_array,
    # State
    VelocityModelState,
    TrainingState,
    InferenceState,
    ModelConfig,
    InferenceConfig,
    # Dynamics
    standard_dynamics_model,
    nonlinear_dynamics_model,
    dynamics_ode_model,
    # Priors
    lognormal_prior,
    informative_prior,
    sample_prior_parameters,
    # Likelihoods
    poisson_likelihood,
    negative_binomial_likelihood,
    create_likelihood,
    # Model
    velocity_model,
    create_model,
)

from pyrovelocity.models.jax.data import (
    # AnnData integration
    prepare_anndata,
    extract_layers,
    store_results,
    get_library_size,
    # Batch processing
    random_batch_indices,
    create_batch_iterator,
    batch_data,
    vmap_batch_function,
    # Preprocessing
    normalize_counts,
    compute_size_factors,
    filter_genes,
    _internal_transform,
)

from pyrovelocity.models.jax.inference import (
    # Config
    create_inference_config,
    validate_config,
    get_default_config,
    # Guide
    auto_normal_guide,
    auto_delta_guide,
    custom_guide,
    create_guide,
    # SVI
    create_optimizer,
    create_svi,
    svi_step,
    run_svi_inference,
    # MCMC
    create_mcmc,
    run_mcmc_inference,
    mcmc_diagnostics,
    # Unified
    run_inference,
    extract_posterior_samples,
    posterior_predictive,
    create_inference_state,
    # Posterior
    sample_posterior,
    compute_velocity,
    compute_uncertainty,
    analyze_posterior,
    create_inference_data,
    format_anndata_output,
)

from pyrovelocity.models.jax.train import (
    # Loop
    train_model,
    evaluate_model,
    train_with_early_stopping,
    train_epoch,
    # Optimizer
    create_optimizer,
    learning_rate_schedule,
    clip_gradients,
    create_optimizer_with_schedule,
    # Metrics
    compute_loss,
    compute_elbo,
    compute_predictive_log_likelihood,
    compute_metrics,
    compute_validation_metrics,
)

from pyrovelocity.models.jax.adapters import (
    # PyTorch to JAX
    convert_tensor_to_jax,
    convert_parameters_to_jax,
    convert_model_state_to_jax,
    convert_pyro_to_numpyro_model,
    convert_pyro_to_numpyro_guide,
    convert_pyro_to_numpyro_posterior,
    # JAX to PyTorch
    convert_array_to_torch,
    convert_parameters_to_torch,
    convert_model_state_to_torch,
    convert_numpyro_to_pyro_model,
    convert_numpyro_to_pyro_guide,
    convert_numpyro_to_pyro_posterior,
)

__all__ = [
    # Utils
    "create_key",
    "split_key",
    "set_platform_device",
    "enable_x64",
    "disable_x64",
    "get_device_count",
    "get_devices",
    "check_array_shape",
    "check_array_dtype",
    "ensure_array",
    # State
    "VelocityModelState",
    "TrainingState",
    "InferenceState",
    "ModelConfig",
    "InferenceConfig",
    # Dynamics
    "standard_dynamics_model",
    "nonlinear_dynamics_model",
    "dynamics_ode_model",
    # Priors
    "lognormal_prior",
    "informative_prior",
    "sample_prior_parameters",
    # Likelihoods
    "poisson_likelihood",
    "negative_binomial_likelihood",
    "create_likelihood",
    # Model
    "velocity_model",
    "create_model",
    # AnnData integration
    "prepare_anndata",
    "extract_layers",
    "store_results",
    "get_library_size",
    # Batch processing
    "random_batch_indices",
    "create_batch_iterator",
    "batch_data",
    "vmap_batch_function",
    # Preprocessing
    "normalize_counts",
    "compute_size_factors",
    "filter_genes",
    "_internal_transform",
    # Config
    "create_inference_config",
    "validate_config",
    "get_default_config",
    # Guide
    "auto_normal_guide",
    "auto_delta_guide",
    "custom_guide",
    "create_guide",
    # SVI
    "create_optimizer",
    "create_svi",
    "svi_step",
    "run_svi_inference",
    # MCMC
    "create_mcmc",
    "run_mcmc_inference",
    "mcmc_diagnostics",
    # Unified
    "run_inference",
    "extract_posterior_samples",
    "posterior_predictive",
    "create_inference_state",
    # Posterior
    "sample_posterior",
    "compute_velocity",
    "compute_uncertainty",
    "analyze_posterior",
    "create_inference_data",
    "format_anndata_output",
    # Loop
    "train_model",
    "evaluate_model",
    "train_with_early_stopping",
    "train_epoch",
    # Optimizer
    "learning_rate_schedule",
    "clip_gradients",
    "create_optimizer_with_schedule",
    # Metrics
    "compute_loss",
    "compute_elbo",
    "compute_predictive_log_likelihood",
    "compute_metrics",
    "compute_validation_metrics",
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
