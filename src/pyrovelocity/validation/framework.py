"""
PyroVelocity validation framework.

This module provides a framework for validating and comparing different
implementations of PyroVelocity (legacy, modular, and JAX).

The framework includes:
- ValidationRunner: A class for running validation on different implementations
- run_validation: A function for running validation with default settings
- compare_implementations: A function for comparing implementation results
"""

import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import anndata as ad
import jax
import jax.numpy as jnp
import numpy as np
import torch
from anndata import AnnData
from beartype import beartype

# Import the different implementations
from pyrovelocity.models._velocity import PyroVelocity
from pyrovelocity.models.jax.factory.factory import (
    create_standard_model as create_jax_standard_model,
)
from pyrovelocity.models.modular import PyroVelocityModel, create_standard_model

# Import comparison utilities
from pyrovelocity.validation.comparison import (
    compare_parameters,
    compare_performance,
    compare_uncertainties,
    compare_velocities,
)

# Import scalene for performance profiling
try:
    from scalene import scalene_profiler
    HAS_SCALENE = True
except ImportError:
    HAS_SCALENE = False
    print("Warning: Scalene not installed. Performance profiling will be limited.")


class ValidationRunner:
    """
    Runner for validating and comparing different PyroVelocity implementations.

    This class provides methods for running validation on different implementations
    of PyroVelocity (legacy, modular, and JAX) and comparing their results.

    Attributes:
        adata: AnnData object containing gene expression data
        models: Dictionary of models to validate
        results: Dictionary of validation results
    """

    @beartype
    def __init__(self, adata: AnnData):
        """
        Initialize the ValidationRunner.

        Args:
            adata: AnnData object containing gene expression data
        """
        self.adata = adata
        self.models = {}
        self.results = {}

    @beartype
    def add_model(self, name: str, model: Any) -> None:
        """
        Add a model to the ValidationRunner.

        Args:
            name: Name of the model
            model: Model instance
        """
        self.models[name] = model

    @beartype
    def setup_legacy_model(self, **kwargs) -> None:
        """
        Set up the legacy PyroVelocity model.

        Args:
            **kwargs: Keyword arguments for PyroVelocity constructor
        """
        # Set up AnnData for legacy model
        PyroVelocity.setup_anndata(self.adata)

        # Create legacy model
        model = PyroVelocity(self.adata, **kwargs)

        # Add model to ValidationRunner
        self.add_model("legacy", model)

    @beartype
    def setup_modular_model(self, **kwargs) -> None:
        """
        Set up the modular PyroVelocity model.

        Args:
            **kwargs: Keyword arguments for create_standard_model
        """
        # Set up AnnData for modular model
        adata = PyroVelocityModel.setup_anndata(self.adata.copy())

        # Create modular model
        model = create_standard_model(**kwargs)

        # Add model to ValidationRunner
        self.add_model("modular", model)

    @beartype
    def setup_jax_model(self, **kwargs) -> None:
        """
        Set up the JAX PyroVelocity model.

        Args:
            **kwargs: Keyword arguments for create_jax_standard_model
        """
        # Create JAX model
        model = create_jax_standard_model(**kwargs)

        # Add model to ValidationRunner
        self.add_model("jax", model)

    @beartype
    def run_validation(
        self,
        max_epochs: int = 100,
        num_samples: int = 30,
        use_scalene: bool = True,
        **kwargs
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run validation on all models.

        Args:
            max_epochs: Maximum number of epochs for training
            num_samples: Number of posterior samples to generate
            use_scalene: Whether to use Scalene for performance profiling
            **kwargs: Additional keyword arguments for training

        Returns:
            Dictionary of validation results
        """
        # Initialize results dictionary
        self.results = {}

        # Run validation for each model
        for name, model in self.models.items():
            print(f"Running validation for {name} model...")

            # Initialize results for this model
            self.results[name] = {}

            # Train model with performance profiling
            if HAS_SCALENE and use_scalene:
                try:
                    # Start Scalene profiler
                    scalene_profiler.start()
                except Exception as e:
                    print(f"Warning: Failed to start Scalene profiler: {e}")
                    # Continue without Scalene

            # Record training start time
            training_start_time = time.time()

            # Train model
            if name == "legacy":
                model.train(max_epochs=max_epochs, **kwargs)
            elif name == "modular":
                model.train(adata=self.adata, max_epochs=max_epochs, **kwargs)
            elif name == "jax":
                # For JAX model, we need to prepare data from AnnData
                from pyrovelocity.models.jax.data.anndata import prepare_anndata
                data_dict = prepare_anndata(self.adata)

                # Create inference configuration
                from pyrovelocity.models.jax.core.state import InferenceConfig
                inference_config = InferenceConfig(
                    num_epochs=max_epochs,
                    **kwargs
                )

                # Run inference
                from pyrovelocity.models.jax.inference.unified import (
                    run_inference,
                )
                _, inference_state = run_inference(
                    model=model,
                    args=(),
                    kwargs=data_dict,
                    config=inference_config,
                    seed=kwargs.get("seed", 0),
                )

                # Store inference state
                self.results[name]["inference_state"] = inference_state

            # Record training end time
            training_end_time = time.time()

            # Store training time
            self.results[name]["performance"] = {
                "training_time": training_end_time - training_start_time
            }

            if HAS_SCALENE and use_scalene:
                try:
                    # Stop Scalene profiler
                    scalene_profiler.stop()
                except Exception as e:
                    print(f"Warning: Failed to stop Scalene profiler: {e}")
                    # Continue without Scalene

            # Generate posterior samples with performance profiling
            if HAS_SCALENE and use_scalene:
                try:
                    # Start Scalene profiler
                    scalene_profiler.start()
                except Exception as e:
                    print(f"Warning: Failed to start Scalene profiler: {e}")
                    # Continue without Scalene

            # Record inference start time
            inference_start_time = time.time()

            # Generate posterior samples
            if name == "legacy":
                posterior_samples = model.generate_posterior_samples(
                    model.adata, num_samples=num_samples
                )
            elif name == "modular":
                posterior_samples = model.generate_posterior_samples(
                    adata=self.adata, num_samples=num_samples
                )
            elif name == "jax":
                # For JAX model, we need to sample from the posterior
                from pyrovelocity.models.jax.inference.posterior import (
                    sample_posterior,
                )
                posterior_samples = sample_posterior(
                    inference_state=self.results[name]["inference_state"],
                    num_samples=num_samples,
                )

            # Record inference end time
            inference_end_time = time.time()

            # Store inference time
            self.results[name]["performance"]["inference_time"] = inference_end_time - inference_start_time

            if HAS_SCALENE and use_scalene:
                try:
                    # Stop Scalene profiler
                    scalene_profiler.stop()
                except Exception as e:
                    print(f"Warning: Failed to stop Scalene profiler: {e}")
                    # Continue without Scalene

            # Store posterior samples
            self.results[name]["posterior_samples"] = posterior_samples

            # Compute velocity from posterior samples
            if name == "legacy":
                # For legacy model, velocity is already computed
                velocity = posterior_samples.get("velocity", None)
            elif name == "modular":
                # For modular model, we need to compute velocity
                from pyrovelocity.models.modular.inference.posterior import (
                    compute_velocity,
                )
                velocity = compute_velocity(
                    model=model,
                    posterior_samples=posterior_samples,
                    adata=self.adata,
                )
            elif name == "jax":
                # For JAX model, we need to compute velocity
                from pyrovelocity.models.jax.inference.posterior import (
                    compute_velocity,
                )
                velocity = compute_velocity(
                    posterior_samples=posterior_samples,
                )

            # Store velocity
            self.results[name]["velocity"] = velocity

            # Compute uncertainty from posterior samples
            if name == "legacy":
                # For legacy model, uncertainty is already computed
                uncertainty = posterior_samples.get("velocity_uncertainty", None)
            elif name == "modular":
                # For modular model, we need to compute uncertainty
                from pyrovelocity.models.modular.inference.posterior import (
                    compute_uncertainty,
                )
                uncertainty = compute_uncertainty(
                    velocity=velocity,
                )
            elif name == "jax":
                # For JAX model, we need to compute uncertainty
                from pyrovelocity.models.jax.inference.posterior import (
                    compute_uncertainty,
                )
                uncertainty = compute_uncertainty(
                    velocity_samples=velocity,
                )

            # Store uncertainty
            self.results[name]["uncertainty"] = uncertainty

        # Return results
        return self.results

    @beartype
    def compare_implementations(self) -> Dict[str, Any]:
        """
        Compare the results of different implementations.

        Returns:
            Dictionary of comparison results
        """
        # Check that results are available
        if not self.results:
            raise ValueError("No results available. Run validation first.")

        # Compare parameters
        parameter_comparison = compare_parameters(self.results)

        # Compare velocities
        velocity_comparison = compare_velocities(self.results)

        # Compare uncertainties
        uncertainty_comparison = compare_uncertainties(self.results)

        # Compare performance
        performance_comparison = compare_performance(self.results)

        # Return comparison results
        return {
            "parameter_comparison": parameter_comparison,
            "velocity_comparison": velocity_comparison,
            "uncertainty_comparison": uncertainty_comparison,
            "performance_comparison": performance_comparison,
        }


@beartype
def run_validation(
    adata: AnnData,
    max_epochs: int = 100,
    num_samples: int = 30,
    use_scalene: bool = True,
    use_legacy: bool = True,
    use_modular: bool = True,
    use_jax: bool = True,
    legacy_model_kwargs: Optional[Dict[str, Any]] = None,
    modular_model_kwargs: Optional[Dict[str, Any]] = None,
    jax_model_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Run validation on different PyroVelocity implementations.

    Args:
        adata: AnnData object containing gene expression data
        max_epochs: Maximum number of epochs for training
        num_samples: Number of posterior samples to generate
        use_scalene: Whether to use Scalene for performance profiling
        use_legacy: Whether to use the legacy implementation
        use_modular: Whether to use the modular implementation
        use_jax: Whether to use the JAX implementation
        legacy_model_kwargs: Keyword arguments for legacy model
        modular_model_kwargs: Keyword arguments for modular model
        jax_model_kwargs: Keyword arguments for JAX model
        **kwargs: Additional keyword arguments for training

    Returns:
        Dictionary of validation results and comparison
    """
    # Initialize ValidationRunner
    runner = ValidationRunner(adata)

    # Set up models
    if use_legacy:
        legacy_kwargs = legacy_model_kwargs or {}
        runner.setup_legacy_model(**legacy_kwargs)

    if use_modular:
        modular_kwargs = modular_model_kwargs or {}
        runner.setup_modular_model(**modular_kwargs)

    if use_jax:
        jax_kwargs = jax_model_kwargs or {}
        runner.setup_jax_model(**jax_kwargs)

    # Run validation
    results = runner.run_validation(
        max_epochs=max_epochs,
        num_samples=num_samples,
        use_scalene=use_scalene,
        **kwargs
    )

    # Compare implementations
    comparison = runner.compare_implementations()

    # Return results and comparison
    return {
        "results": results,
        "comparison": comparison,
    }


@beartype
def compare_implementations(
    results: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Compare the results of different implementations.

    Args:
        results: Dictionary of validation results

    Returns:
        Dictionary of comparison results
    """
    # Compare parameters
    parameter_comparison = compare_parameters(results)

    # Compare velocities
    velocity_comparison = compare_velocities(results)

    # Compare uncertainties
    uncertainty_comparison = compare_uncertainties(results)

    # Compare performance
    performance_comparison = compare_performance(results)

    # Return comparison results
    return {
        "parameter_comparison": parameter_comparison,
        "velocity_comparison": velocity_comparison,
        "uncertainty_comparison": uncertainty_comparison,
        "performance_comparison": performance_comparison,
    }
