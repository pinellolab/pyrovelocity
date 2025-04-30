"""
PyroVelocity validation framework.

This module provides a framework for validating and comparing different
implementations of PyroVelocity (legacy, modular, and JAX).

The validation framework enables:
1. Running multiple implementations on the same data
2. Comparing parameter estimates across implementations
3. Comparing velocity estimates across implementations
4. Comparing uncertainty estimates across implementations
5. Comparing performance metrics across implementations

The framework includes:
- ValidationRunner: A class for running validation on different implementations
- run_validation: A function for running validation with default settings
- compare_implementations: A function for comparing implementation results

Example:
    >>> import anndata as ad
    >>> from pyrovelocity.io.datasets import pancreas
    >>> from pyrovelocity.validation.framework import run_validation
    >>>
    >>> # Load data
    >>> adata = pancreas()
    >>>
    >>> # Run validation
    >>> results = run_validation(
    ...     adata=adata,
    ...     max_epochs=100,
    ...     num_samples=30,
    ...     use_legacy=True,
    ...     use_modular=True,
    ...     use_jax=True
    ... )
    >>>
    >>> # Extract results and comparison
    >>> model_results = results["results"]
    >>> comparison = results["comparison"]
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
    It serves as a unified interface for working with all three implementations,
    handling the differences in their APIs and data requirements.

    The ValidationRunner workflow consists of:
    1. Setting up models for each implementation
    2. Running validation (training, posterior sampling, velocity computation)
    3. Comparing results across implementations

    Attributes:
        adata: AnnData object containing gene expression data
        models: Dictionary of models to validate, with keys 'legacy', 'modular', 'jax'
        results: Dictionary of validation results for each model

    Examples:
        >>> import anndata as ad
        >>> from pyrovelocity.io.datasets import pancreas
        >>> from pyrovelocity.validation.framework import ValidationRunner
        >>>
        >>> # Load data
        >>> adata = pancreas()
        >>>
        >>> # Initialize ValidationRunner
        >>> runner = ValidationRunner(adata)
        >>>
        >>> # Set up models
        >>> runner.setup_legacy_model(model_type="deterministic")
        >>> runner.setup_modular_model(model_type="standard")
        >>> runner.setup_jax_model(model_type="standard")
        >>>
        >>> # Run validation
        >>> results = runner.run_validation(
        ...     max_epochs=100,
        ...     num_samples=30
        ... )
        >>>
        >>> # Compare implementations
        >>> comparison = runner.compare_implementations()
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
        Run validation on all models that have been set up.

        This method runs the validation process on all models that have been
        added to the ValidationRunner. For each model, it:
        1. Trains the model on the data
        2. Generates posterior samples
        3. Computes velocity and uncertainty estimates
        4. Measures performance metrics

        The method handles the differences in APIs between the different
        implementations, making it easy to run the same validation process
        on all implementations.

        Args:
            max_epochs: Maximum number of epochs for training each model
            num_samples: Number of posterior samples to generate for uncertainty quantification
            use_scalene: Whether to use Scalene for performance profiling (if installed)
            **kwargs: Additional keyword arguments for training (e.g., learning_rate, batch_size)

        Returns:
            Dictionary of validation results, with keys for each implementation
            (e.g., 'legacy', 'modular', 'jax') and values containing the
            validation results for that implementation

        Examples:
            >>> # Assuming we have set up models
            >>> # runner.setup_legacy_model()
            >>> # runner.setup_modular_model()
            >>> # runner.setup_jax_model()
            >>>
            >>> # Run validation
            >>> results = runner.run_validation(
            ...     max_epochs=100,
            ...     num_samples=30,
            ...     learning_rate=0.01,
            ...     batch_size=128
            ... )
            >>>
            >>> # Access results for a specific implementation
            >>> legacy_results = results["legacy"]
            >>> modular_results = results["modular"]
            >>> jax_results = results["jax"]
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
        Compare the results of different implementations after running validation.

        This method compares the results of different PyroVelocity implementations
        that have been validated using the run_validation method. It compares:
        1. Parameters: alpha, beta, gamma estimates
        2. Velocities: Velocity vectors
        3. Uncertainties: Uncertainty estimates
        4. Performance: Training and inference times

        The method delegates to the comparison utilities in the validation.comparison
        module to perform the actual comparisons.

        Returns:
            Dictionary of comparison results with the following keys:
            - "parameter_comparison": Comparison of parameter estimates
            - "velocity_comparison": Comparison of velocity vectors
            - "uncertainty_comparison": Comparison of uncertainty estimates
            - "performance_comparison": Comparison of performance metrics

        Raises:
            ValueError: If no results are available (run_validation has not been called)

        Examples:
            >>> # Assuming we have run validation
            >>> # results = runner.run_validation(...)
            >>>
            >>> # Compare implementations
            >>> comparison = runner.compare_implementations()
            >>>
            >>> # Access specific comparisons
            >>> parameter_comparison = comparison["parameter_comparison"]
            >>> velocity_comparison = comparison["velocity_comparison"]
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

    This function provides a convenient interface for running validation on
    different PyroVelocity implementations. It creates a ValidationRunner,
    sets up the requested models, runs validation, and compares the results.

    The function allows you to specify which implementations to include in the
    validation, as well as custom parameters for each implementation. This is
    useful for comparing the behavior of different implementations with the
    same data and parameters.

    Args:
        adata: AnnData object containing gene expression data
        max_epochs: Maximum number of epochs for training
        num_samples: Number of posterior samples to generate
        use_scalene: Whether to use Scalene for performance profiling
        use_legacy: Whether to use the legacy implementation
        use_modular: Whether to use the modular implementation
        use_jax: Whether to use the JAX implementation
        legacy_model_kwargs: Keyword arguments for legacy model (e.g., model_type, latent_time)
        modular_model_kwargs: Keyword arguments for modular model (e.g., model_type, latent_time)
        jax_model_kwargs: Keyword arguments for JAX model (e.g., model_type, latent_time)
        **kwargs: Additional keyword arguments for training (e.g., learning_rate, batch_size)

    Returns:
        Dictionary with two keys:
        - "results": Dictionary of validation results for each model
        - "comparison": Dictionary of comparison results across implementations

    Examples:
        >>> import anndata as ad
        >>> from pyrovelocity.io.datasets import pancreas
        >>> from pyrovelocity.validation.framework import run_validation
        >>>
        >>> # Load data
        >>> adata = pancreas()
        >>>
        >>> # Run validation with all implementations
        >>> results = run_validation(
        ...     adata=adata,
        ...     max_epochs=100,
        ...     num_samples=30
        ... )
        >>>
        >>> # Run validation with specific implementations and parameters
        >>> results = run_validation(
        ...     adata=adata,
        ...     max_epochs=100,
        ...     num_samples=30,
        ...     use_legacy=False,
        ...     modular_model_kwargs={"model_type": "standard", "latent_time": True},
        ...     jax_model_kwargs={"model_type": "standard", "latent_time": True}
        ... )
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
    Compare the results of different PyroVelocity implementations.

    This function compares the results of different PyroVelocity implementations
    across several dimensions:
    1. Parameters: Compares alpha, beta, gamma estimates
    2. Velocities: Compares velocity vectors
    3. Uncertainties: Compares uncertainty estimates
    4. Performance: Compares training and inference times

    The function takes a dictionary of validation results, typically obtained
    from ValidationRunner.run_validation(), and returns a dictionary of
    comparison results that can be used for visualization and analysis.

    Args:
        results: Dictionary of validation results, with keys for each implementation
                (e.g., 'legacy', 'modular', 'jax') and values containing the
                validation results for that implementation

    Returns:
        Dictionary of comparison results with the following keys:
        - "parameter_comparison": Comparison of parameter estimates
        - "velocity_comparison": Comparison of velocity vectors
        - "uncertainty_comparison": Comparison of uncertainty estimates
        - "performance_comparison": Comparison of performance metrics

    Examples:
        >>> from pyrovelocity.validation.framework import ValidationRunner, compare_implementations
        >>>
        >>> # Assuming we have run validation and have results
        >>> # results = runner.run_validation(...)
        >>>
        >>> # Compare implementations
        >>> comparison = compare_implementations(results)
        >>>
        >>> # Access specific comparisons
        >>> parameter_comparison = comparison["parameter_comparison"]
        >>> velocity_comparison = comparison["velocity_comparison"]
        >>> uncertainty_comparison = comparison["uncertainty_comparison"]
        >>> performance_comparison = comparison["performance_comparison"]
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
