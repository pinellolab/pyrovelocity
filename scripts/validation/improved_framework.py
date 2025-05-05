"""
Improved PyroVelocity validation framework.

This module provides an improved version of the validation framework for
validating and comparing different implementations of PyroVelocity.

The improvements include:
1. Better error handling in setup_legacy_model and setup_modular_model
2. Explicit handling of validation_fraction in legacy model
3. Better error handling in run_validation
4. Handling of failed models in compare_implementations
"""

import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from anndata import AnnData
from beartype import beartype

# Import the different implementations
from pyrovelocity.models._velocity import PyroVelocity
from pyrovelocity.models.modular import PyroVelocityModel
from pyrovelocity.models.modular.factory import create_standard_model
from pyrovelocity.validation.comparison import (
    compare_parameters,
    compare_velocities,
    compare_uncertainties,
    compare_performance,
)

# Import legacy adapter
from legacy_adapter import (
    get_velocity_from_legacy_model,
    get_velocity_uncertainty_from_legacy_model,
)

# Import JAX implementation only if available
try:
    from pyrovelocity.models.jax.factory.factory import (
        create_standard_model as create_jax_standard_model,
    )
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    print("Warning: JAX implementation not available. JAX validation will be disabled.")

# Import Scalene profiler if available
try:
    import scalene.scalene_profiler as scalene_profiler
    HAS_SCALENE = True
except ImportError:
    HAS_SCALENE = False
    print("Warning: Scalene profiler not available. Performance profiling will be disabled.")


class ImprovedValidationRunner:
    """
    Improved runner for validating and comparing different PyroVelocity implementations.

    This class provides methods for running validation on different implementations
    of PyroVelocity (legacy, modular, and JAX) and comparing their results.
    It serves as a unified interface for working with all three implementations,
    handling the differences in their APIs and data requirements.

    The ValidationRunner workflow consists of:
    1. Setting up models for each implementation
    2. Running validation (training, posterior sampling, velocity computation)
    3. Comparing results across implementations

    The improvements include:
    1. Better error handling in setup_legacy_model and setup_modular_model
    2. Explicit handling of validation_fraction in legacy model
    3. Better error handling in run_validation
    4. Handling of failed models in compare_implementations
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
        try:
            # Set up AnnData for legacy model
            PyroVelocity.setup_anndata(self.adata)

            # Create legacy model with validation_fraction=0 to avoid validation dataloader issues
            model_kwargs = kwargs.copy()
            # Explicitly set validation_fraction to 0
            model_kwargs["validation_fraction"] = 0

            # Create legacy model
            model = PyroVelocity(self.adata, **model_kwargs)

            # Add model to ValidationRunner
            self.add_model("legacy", model)
            print("Legacy model setup successful")
        except Exception as e:
            print(f"Error setting up legacy model: {e}")
            import traceback
            traceback.print_exc()
            # Return None to indicate failure
            return None

    @beartype
    def setup_modular_model(self, **kwargs) -> None:
        """
        Set up the modular PyroVelocity model.

        Args:
            **kwargs: Additional keyword arguments for model creation
        """
        try:
            # Set up AnnData for modular model
            PyroVelocityModel.setup_anndata(self.adata.copy())

            # Create standard model with explicit configuration
            model_type = kwargs.get("model_type", "standard")

            if model_type == "standard":
                model = create_standard_model()
            else:
                raise ValueError(f"Unsupported model type: {model_type}")

            # Add model to ValidationRunner
            self.add_model("modular", model)
            print("Modular model setup successful")
        except Exception as e:
            print(f"Error setting up modular model: {e}")
            import traceback
            traceback.print_exc()
            # Return None to indicate failure
            return None

    @beartype
    def setup_jax_model(self, **kwargs) -> None:
        """
        Set up the JAX PyroVelocity model.

        Args:
            **kwargs: Additional keyword arguments for model creation

        Raises:
            ImportError: If JAX implementation is not available
        """
        try:
            if not HAS_JAX:
                raise ImportError("JAX implementation not available. Cannot set up JAX model.")

            # Create JAX standard model (only option currently available)
            model = create_jax_standard_model()

            # Add model to ValidationRunner
            self.add_model("jax", model)
            print("JAX model setup successful")
        except Exception as e:
            print(f"Error setting up JAX model: {e}")
            import traceback
            traceback.print_exc()
            # Return None to indicate failure
            return None

    @beartype
    def run_validation(
        self,
        max_epochs: int = 100,
        num_samples: int = 30,
        use_scalene: bool = False,  # Disable Scalene by default
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
            max_epochs: Maximum number of epochs for training
            num_samples: Number of posterior samples to generate
            use_scalene: Whether to use Scalene for performance profiling
            **kwargs: Additional keyword arguments for training

        Returns:
            Dictionary of results for each model
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
            try:
                if name == "legacy":
                    # For legacy model, we need to remove validation_fraction
                    # to avoid issues with the validation dataloader
                    legacy_kwargs = kwargs.copy()
                    if "validation_fraction" in legacy_kwargs:
                        del legacy_kwargs["validation_fraction"]
                    model.train(max_epochs=max_epochs, **legacy_kwargs)
                elif name == "modular":
                    model.train(adata=self.adata, max_epochs=max_epochs, **kwargs)
                elif name == "jax":
                    # For JAX model, we need to prepare data from AnnData
                    from pyrovelocity.models.jax.data.anndata import (
                        prepare_anndata,
                    )

                    # Prepare data from AnnData
                    data_dict = prepare_anndata(
                        self.adata,
                        spliced_layer="spliced",
                        unspliced_layer="unspliced",
                    )

                    # Create inference config
                    inference_config = {
                        "num_warmup": max_epochs // 2,
                        "num_samples": max_epochs // 2,
                        "num_chains": 1,
                    }

                    # Run inference
                    import jax

                    from pyrovelocity.models.jax.inference.unified import (
                        run_inference,
                    )

                    # Create a JAX random key from the seed
                    key = jax.random.PRNGKey(kwargs.get("seed", 0))

                    _, inference_state = run_inference(
                        model=model,
                        args=(),
                        kwargs=data_dict,
                        config=inference_config,
                        key=key,
                    )

                    # Store inference state
                    self.results[name]["inference_state"] = inference_state
            except Exception as e:
                print(f"Error training {name} model: {e}")
                import traceback
                traceback.print_exc()
                # Store error in results
                self.results[name]["error"] = str(e)
                self.results[name]["traceback"] = traceback.format_exc()
                # Skip the rest of the processing for this model
                continue

            # Record training end time
            training_end_time = time.time()
            training_time = training_end_time - training_start_time

            # Store training time
            self.results[name]["performance"] = {"training_time": training_time}

            # Stop Scalene profiler if it was started
            if HAS_SCALENE and use_scalene:
                try:
                    # Stop Scalene profiler
                    scalene_profiler.stop()
                except Exception as e:
                    print(f"Warning: Failed to stop Scalene profiler: {e}")

            # Generate posterior samples
            print(f"Generating posterior samples for {name} model...")
            inference_start_time = time.time()

            try:
                if name == "legacy":
                    # For legacy model, use generate_posterior_samples
                    posterior_samples = model.generate_posterior_samples(num_samples=num_samples)
                elif name == "modular":
                    # For modular model, use get_posterior_samples
                    posterior_samples = model.get_posterior_samples(
                        adata=self.adata, num_samples=num_samples
                    )
                elif name == "jax":
                    # For JAX model, extract posterior samples from inference state
                    inference_state = self.results[name]["inference_state"]
                    posterior_samples = {
                        "alpha": inference_state.posterior["alpha"],
                        "beta": inference_state.posterior["beta"],
                        "gamma": inference_state.posterior["gamma"],
                    }
            except Exception as e:
                print(f"Error generating posterior samples for {name} model: {e}")
                import traceback
                traceback.print_exc()
                # Store error in results
                self.results[name]["error"] = str(e)
                self.results[name]["traceback"] = traceback.format_exc()
                # Skip the rest of the processing for this model
                continue

            # Record inference end time
            inference_end_time = time.time()
            inference_time = inference_end_time - inference_start_time

            # Store inference time
            self.results[name]["performance"]["inference_time"] = inference_time

            # Store posterior samples
            self.results[name]["posterior_samples"] = posterior_samples

            # Compute velocity
            print(f"Computing velocity for {name} model...")
            try:
                if name == "legacy":
                    # For legacy model, use our adapter function
                    velocity = get_velocity_from_legacy_model(
                        model=model,
                        posterior_samples=posterior_samples,
                        adata=self.adata
                    )
                elif name == "modular":
                    # For modular model, use get_velocity
                    velocity = model.get_velocity(adata=self.adata)
                elif name == "jax":
                    # For JAX model, compute velocity from posterior samples
                    # This is a placeholder - in a real implementation, we would
                    # use the JAX model's velocity computation method
                    velocity = np.zeros((self.adata.shape[0], self.adata.shape[1]))
            except Exception as e:
                print(f"Error computing velocity for {name} model: {e}")
                import traceback
                traceback.print_exc()
                # Store error in results
                self.results[name]["error"] = str(e)
                self.results[name]["traceback"] = traceback.format_exc()
                # Skip the rest of the processing for this model
                continue

            # Store velocity
            self.results[name]["velocity"] = velocity

            # Compute uncertainty
            print(f"Computing uncertainty for {name} model...")
            try:
                if name == "legacy":
                    # For legacy model, use our adapter function
                    uncertainty = get_velocity_uncertainty_from_legacy_model(
                        model=model,
                        posterior_samples=posterior_samples,
                        adata=self.adata
                    )
                elif name == "modular":
                    # For modular model, use get_velocity_uncertainty
                    uncertainty = model.get_velocity_uncertainty(adata=self.adata)
                elif name == "jax":
                    # For JAX model, compute uncertainty from posterior samples
                    # This is a placeholder - in a real implementation, we would
                    # use the JAX model's uncertainty computation method
                    uncertainty = np.zeros(self.adata.shape[1])
            except Exception as e:
                print(f"Error computing uncertainty for {name} model: {e}")
                import traceback
                traceback.print_exc()
                # Store error in results
                self.results[name]["error"] = str(e)
                self.results[name]["traceback"] = traceback.format_exc()
                # Skip the rest of the processing for this model
                continue

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
            Dictionary of comparison results
        """
        # Check that results are available
        if not self.results:
            raise ValueError("No results available. Run validation first.")

        # Check if any models failed
        failed_models = []
        for model_name, model_result in self.results.items():
            if "error" in model_result:
                failed_models.append(model_name)

        # If any models failed, return a minimal comparison result
        if failed_models:
            print(f"Skipping detailed comparison due to failed models: {failed_models}")
            return {"failed_models": failed_models}

        # Get model names
        model_names = list(self.results.keys())

        # Check that we have at least two models to compare
        if len(model_names) < 2:
            raise ValueError("Need at least two models to compare.")

        # Initialize comparison results
        comparison_results = {}

        # Compare parameters
        parameter_comparison = {}
        for param in ["alpha", "beta", "gamma"]:
            param_comparison = {}
            for i in range(len(model_names)):
                for j in range(i + 1, len(model_names)):
                    model1 = model_names[i]
                    model2 = model_names[j]
                    comp_name = f"{model1}_vs_{model2}"

                    # Get parameter samples
                    param_samples1 = self.results[model1]["posterior_samples"][param]
                    param_samples2 = self.results[model2]["posterior_samples"][param]

                    # Compare parameters
                    param_metrics = compare_parameters(
                        {param: param_samples1}, {param: param_samples2}
                    )[param]

                    # Store comparison results
                    param_comparison[comp_name] = param_metrics

            # Store parameter comparison results
            parameter_comparison[param] = param_comparison

        # Store parameter comparison results
        comparison_results["parameter_comparison"] = parameter_comparison

        # Compare velocities
        velocity_comparison = {}
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                model1 = model_names[i]
                model2 = model_names[j]
                comp_name = f"{model1}_vs_{model2}"

                # Get velocities
                velocity1 = self.results[model1]["velocity"]
                velocity2 = self.results[model2]["velocity"]

                # Compare velocities
                velocity_metrics = compare_velocities(velocity1, velocity2)

                # Store comparison results
                velocity_comparison[comp_name] = velocity_metrics

        # Store velocity comparison results
        comparison_results["velocity_comparison"] = velocity_comparison

        # Compare uncertainties
        uncertainty_comparison = {}
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                model1 = model_names[i]
                model2 = model_names[j]
                comp_name = f"{model1}_vs_{model2}"

                # Get uncertainties
                uncertainty1 = self.results[model1]["uncertainty"]
                uncertainty2 = self.results[model2]["uncertainty"]

                # Compare uncertainties
                uncertainty_metrics = compare_uncertainties(uncertainty1, uncertainty2)

                # Store comparison results
                uncertainty_comparison[comp_name] = uncertainty_metrics

        # Store uncertainty comparison results
        comparison_results["uncertainty_comparison"] = uncertainty_comparison

        # Compare performance
        performance_comparison = {}
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                model1 = model_names[i]
                model2 = model_names[j]
                comp_name = f"{model1}_vs_{model2}"

                # Get performance metrics
                performance1 = self.results[model1]["performance"]
                performance2 = self.results[model2]["performance"]

                # Compare performance
                performance_metrics = compare_performance(performance1, performance2)

                # Store comparison results
                performance_comparison[comp_name] = performance_metrics

        # Store performance comparison results
        comparison_results["performance_comparison"] = performance_comparison

        # Return comparison results
        return comparison_results
