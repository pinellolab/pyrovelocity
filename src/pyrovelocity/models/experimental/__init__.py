"""
PyroVelocity experimental implementations.

This package contains experimental implementations of PyroVelocity models,
including JAX-based deterministic inference, simulation, and hyperparameter calibration.
"""

from pyrovelocity.models.experimental.deterministic_inference import (
    deterministic_transcription_splicing_probabilistic_model,
    generate_test_data_for_deterministic_model_inference,
    generate_prior_inference_data,
    generate_posterior_inference_data,
    plot_sample_phase_portraits,
    plot_sample_trajectories,
    plot_sample_trajectories_with_percentiles,
    save_inference_plots,
)
from pyrovelocity.models.experimental.deterministic_simulation import (
    solve_transcription_splicing_model,
    solve_transcription_splicing_model_analytical,
)
from pyrovelocity.models.experimental.hyperparameter_calibration import (
    lognormal_tail_probability,
    solve_for_lognormal_sigma_given_threshold_and_tail_mass,
    solve_for_lognormal_mu_given_threshold_and_tail_mass,
)

__all__ = [
    # Deterministic inference
    "deterministic_transcription_splicing_probabilistic_model",
    "generate_test_data_for_deterministic_model_inference",
    "generate_prior_inference_data",
    "generate_posterior_inference_data",
    "plot_sample_phase_portraits",
    "plot_sample_trajectories",
    "plot_sample_trajectories_with_percentiles",
    "save_inference_plots",
    # Deterministic simulation
    "solve_transcription_splicing_model",
    "solve_transcription_splicing_model_analytical",
    # Hyperparameter calibration
    "lognormal_tail_probability",
    "solve_for_lognormal_sigma_given_threshold_and_tail_mass",
    "solve_for_lognormal_mu_given_threshold_and_tail_mass",
]
