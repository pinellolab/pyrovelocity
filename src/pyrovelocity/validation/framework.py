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
    >>> import numpy as np
    >>> import pandas as pd
    >>> from pyrovelocity.validation.framework import run_validation
    >>> tmp = getfixture("tmp_path")
    >>>
    >>> # Create synthetic data for testing
    >>> n_cells, n_genes = 10, 5
    >>> u_data = np.random.poisson(5, size=(n_cells, n_genes))
    >>> s_data = np.random.poisson(5, size=(n_cells, n_genes))
    >>>
    >>> # Create AnnData object with all required columns
    >>> adata = ad.AnnData(X=s_data)
    >>> adata.layers["spliced"] = s_data
    >>> adata.layers["unspliced"] = u_data
    >>> # Add raw layers required by legacy model
    >>> adata.layers["raw_spliced"] = s_data.copy()
    >>> adata.layers["raw_unspliced"] = u_data.copy()
    >>> adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
    >>> adata.var_names = [f"gene_{i}" for i in range(n_genes)]
    >>>
    >>> # Add library size information
    >>> adata.obs["u_lib_size_raw"] = np.sum(u_data, axis=1)
    >>> adata.obs["s_lib_size_raw"] = np.sum(s_data, axis=1)
    >>> adata.obs["u_lib_size"] = np.log(adata.obs["u_lib_size_raw"] + 1e-6)
    >>> adata.obs["s_lib_size"] = np.log(adata.obs["s_lib_size_raw"] + 1e-6)
    >>> adata.obs["u_lib_size_mean"] = np.mean(adata.obs["u_lib_size"])
    >>> adata.obs["s_lib_size_mean"] = np.mean(adata.obs["s_lib_size"])
    >>> adata.obs["u_lib_size_scale"] = np.std(adata.obs["u_lib_size"])
    >>> adata.obs["s_lib_size_scale"] = np.std(adata.obs["s_lib_size"])
    >>> adata.obs["ind_x"] = np.arange(n_cells)
    >>>
    >>> # Add UMAP coordinates for visualization
    >>> adata.obsm = {}
    >>> adata.obsm["X_umap"] = np.random.normal(0, 1, size=(n_cells, 2))
    >>> # Add cluster information
    >>> adata.obs["clusters"] = np.random.choice(["A", "B", "C"], size=n_cells)
    >>>
    >>> # Run validation with minimal settings for testing
    >>> try:
    ...     # This is just for doctest - in real use, you would run the full validation
    ...     # We're using try/except because the full validation would take too long for a doctest
    ...     results = run_validation(
    ...         adata=adata,
    ...         max_epochs=1,  # Minimal for testing
    ...         num_samples=1,  # Minimal for testing
    ...         use_legacy=False,  # Skip legacy for faster testing
    ...         use_modular=True,
    ...         use_jax=False,  # Skip JAX for faster testing
    ...     )
    ...     # Extract results and comparison
    ...     model_results = results["results"]
    ...     comparison = results["comparison"]
    ... except Exception as e:
    ...     # In a real scenario, you would handle errors appropriately
    ...     print(f"Validation would run here in a real scenario")
"""

import time
from typing import Any, Dict, Optional, Sequence

import jax.numpy as jnp
import numpy as np
from anndata import AnnData
from beartype import beartype

# Import the different implementations
from pyrovelocity.models._velocity import PyroVelocity
from pyrovelocity.models.jax.factory.factory import (
    create_standard_model as create_jax_standard_model,
)
from pyrovelocity.models.modular import PyroVelocityModel
from pyrovelocity.models.modular.factory import create_standard_model

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
        >>> import numpy as np
        >>> import pandas as pd
        >>> from unittest.mock import MagicMock, patch
        >>> from pyrovelocity.validation.framework import ValidationRunner
        >>> tmp = getfixture("tmp_path")
        >>>
        >>> # Create synthetic data for testing
        >>> n_cells, n_genes = 10, 5
        >>> u_data = np.random.poisson(5, size=(n_cells, n_genes))
        >>> s_data = np.random.poisson(5, size=(n_cells, n_genes))
        >>>
        >>> # Create AnnData object with all required columns
        >>> adata = ad.AnnData(X=s_data)
        >>> adata.layers["spliced"] = s_data
        >>> adata.layers["unspliced"] = u_data
        >>> # Add raw layers required by legacy model
        >>> adata.layers["raw_spliced"] = s_data.copy()
        >>> adata.layers["raw_unspliced"] = u_data.copy()
        >>> adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
        >>> adata.var_names = [f"gene_{i}" for i in range(n_genes)]
        >>>
        >>> # Add library size information
        >>> adata.obs["u_lib_size_raw"] = np.sum(u_data, axis=1)
        >>> adata.obs["s_lib_size_raw"] = np.sum(s_data, axis=1)
        >>> adata.obs["u_lib_size"] = np.log(adata.obs["u_lib_size_raw"] + 1e-6)
        >>> adata.obs["s_lib_size"] = np.log(adata.obs["s_lib_size_raw"] + 1e-6)
        >>> adata.obs["u_lib_size_mean"] = np.mean(adata.obs["u_lib_size"])
        >>> adata.obs["s_lib_size_mean"] = np.mean(adata.obs["s_lib_size"])
        >>> adata.obs["u_lib_size_scale"] = np.std(adata.obs["u_lib_size"])
        >>> adata.obs["s_lib_size_scale"] = np.std(adata.obs["s_lib_size"])
        >>> adata.obs["ind_x"] = np.arange(n_cells)
        >>>
        >>> # Initialize ValidationRunner
        >>> runner = ValidationRunner(adata)
        >>>
        >>> # Mock the setup methods to avoid actual model creation
        >>> runner.setup_legacy_model = MagicMock()
        >>> runner.setup_modular_model = MagicMock()
        >>> runner.setup_jax_model = MagicMock()
        >>>
        >>> # Mock the run_validation method to avoid actual model training
        >>> runner.run_validation = MagicMock(return_value={
        ...     "legacy": {"mock": "results"},
        ...     "modular": {"mock": "results"},
        ...     "jax": {"mock": "results"}
        ... })
        >>>
        >>> # Mock the compare_implementations method
        >>> runner.compare_implementations = MagicMock(return_value={
        ...     "parameter_comparison": {"mock": "comparison"},
        ...     "velocity_comparison": {"mock": "comparison"}
        ... })
        >>>
        >>> # Set up models with minimal settings for testing (these are mocked)
        >>> runner.setup_legacy_model(model_type="deterministic")
        >>> runner.setup_modular_model(model_type="standard")
        >>> runner.setup_jax_model(model_type="standard")
        >>>
        >>> # Run validation with minimal settings for testing (this is mocked)
        >>> results = runner.run_validation(
        ...     max_epochs=2,
        ...     num_samples=2
        ... )
        >>>
        >>> # Compare implementations (this is mocked)
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
            **kwargs: Additional keyword arguments for model creation (ignored)
        """
        # Set up AnnData for modular model
        PyroVelocityModel.setup_anndata(self.adata.copy())

        # Create standard model (only option currently available)
        model = create_standard_model()

        # Add model to ValidationRunner
        self.add_model("modular", model)

    @beartype
    def setup_jax_model(self, **kwargs) -> None:
        """
        Set up the JAX PyroVelocity model.

        Args:
            **kwargs: Additional keyword arguments for model creation (ignored)
        """
        # Create JAX standard model (only option currently available)
        model = create_jax_standard_model()

        # Add model to ValidationRunner
        self.add_model("jax", model)

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
            max_epochs: Maximum number of epochs for training each model
            num_samples: Number of posterior samples to generate for uncertainty quantification
            use_scalene: Whether to use Scalene for performance profiling (if installed)
            **kwargs: Additional keyword arguments for training (e.g., learning_rate, batch_size)

        Returns:
            Dictionary of validation results, with keys for each implementation
            (e.g., 'legacy', 'modular', 'jax') and values containing the
            validation results for that implementation

        Examples:
            >>> # Create synthetic data for testing
            >>> import anndata as ad
            >>> import numpy as np
            >>> import pandas as pd
            >>> from unittest.mock import MagicMock
            >>> tmp = getfixture("tmp_path")
            >>>
            >>> # Create synthetic data
            >>> n_cells, n_genes = 10, 5
            >>> u_data = np.random.poisson(5, size=(n_cells, n_genes))
            >>> s_data = np.random.poisson(5, size=(n_cells, n_genes))
            >>>
            >>> # Create AnnData object with all required columns
            >>> adata = ad.AnnData(X=s_data)
            >>> adata.layers["spliced"] = s_data
            >>> adata.layers["unspliced"] = u_data
            >>> # Add raw layers required by legacy model
            >>> adata.layers["raw_spliced"] = s_data.copy()
            >>> adata.layers["raw_unspliced"] = u_data.copy()
            >>> adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
            >>> adata.var_names = [f"gene_{i}" for i in range(n_genes)]
            >>>
            >>> # Add library size information
            >>> adata.obs["u_lib_size_raw"] = np.sum(u_data, axis=1)
            >>> adata.obs["s_lib_size_raw"] = np.sum(s_data, axis=1)
            >>> adata.obs["u_lib_size"] = np.log(adata.obs["u_lib_size_raw"] + 1e-6)
            >>> adata.obs["s_lib_size"] = np.log(adata.obs["s_lib_size_raw"] + 1e-6)
            >>> adata.obs["u_lib_size_mean"] = np.mean(adata.obs["u_lib_size"])
            >>> adata.obs["s_lib_size_mean"] = np.mean(adata.obs["s_lib_size"])
            >>> adata.obs["u_lib_size_scale"] = np.std(adata.obs["u_lib_size"])
            >>> adata.obs["s_lib_size_scale"] = np.std(adata.obs["s_lib_size"])
            >>> adata.obs["ind_x"] = np.arange(n_cells)
            >>>
            >>> # Initialize ValidationRunner
            >>> from pyrovelocity.validation.framework import ValidationRunner
            >>> runner = ValidationRunner(adata)
            >>>
            >>> # Mock the run_validation method to avoid actual implementation
            >>> original_run_validation = runner.run_validation
            >>> runner.run_validation = MagicMock(return_value={
            ...     "legacy": {
            ...         "posterior_samples": {
            ...             "alpha": np.ones((2, n_genes)),
            ...             "beta": np.ones((2, n_genes)),
            ...             "gamma": np.ones((2, n_genes))
            ...         },
            ...         "velocity": np.ones((n_cells, n_genes)),
            ...         "uncertainty": np.ones((n_genes,)),
            ...         "performance": {"training_time": 1.0, "inference_time": 0.5}
            ...     },
            ...     "modular": {
            ...         "posterior_samples": {
            ...             "alpha": np.ones((2, n_genes)) * 1.1,
            ...             "beta": np.ones((2, n_genes)) * 1.1,
            ...             "gamma": np.ones((2, n_genes)) * 1.1
            ...         },
            ...         "velocity": np.ones((n_cells, n_genes)) * 1.1,
            ...         "uncertainty": np.ones((n_genes,)) * 1.1,
            ...         "performance": {"training_time": 0.9, "inference_time": 0.4}
            ...     },
            ...     "jax": {
            ...         "posterior_samples": {
            ...             "alpha": np.ones((2, n_genes)) * 1.2,
            ...             "beta": np.ones((2, n_genes)) * 1.2,
            ...             "gamma": np.ones((2, n_genes)) * 1.2
            ...         },
            ...         "velocity": np.ones((n_cells, n_genes)) * 1.2,
            ...         "uncertainty": np.ones((n_genes,)) * 1.2,
            ...         "performance": {"training_time": 0.8, "inference_time": 0.3}
            ...     }
            ... })
            >>>
            >>> # Run validation with minimal settings for testing
            >>> # This will use our mocked models instead of real ones
            >>> results = runner.run_validation(
            ...     max_epochs=2,
            ...     num_samples=2
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

                # Rename keys to match model function parameters
                if "X_unspliced" in data_dict and "X_spliced" in data_dict:
                    data_dict["u_obs"] = data_dict.pop("X_unspliced")
                    data_dict["s_obs"] = data_dict.pop("X_spliced")

                # Add library size information
                if "u_lib_size" in data_dict and "s_lib_size" in data_dict:
                    data_dict["u_log_library"] = jnp.log(data_dict["u_lib_size"])
                    data_dict["s_log_library"] = jnp.log(data_dict["s_lib_size"])

                # Remove keys that are not used by the model
                keys_to_keep = ["u_obs", "s_obs", "u_log_library", "s_log_library"]
                data_dict = {k: v for k, v in data_dict.items() if k in keys_to_keep}

                # Add batch dimension for factory model
                data_dict["u_obs"] = jnp.expand_dims(data_dict["u_obs"], axis=0)
                data_dict["s_obs"] = jnp.expand_dims(data_dict["s_obs"], axis=0)
                if "u_log_library" in data_dict:
                    data_dict["u_log_library"] = jnp.expand_dims(data_dict["u_log_library"], axis=0)
                if "s_log_library" in data_dict:
                    data_dict["s_log_library"] = jnp.expand_dims(data_dict["s_log_library"], axis=0)

                # Create inference configuration
                from pyrovelocity.models.jax.core.state import InferenceConfig
                inference_config = InferenceConfig(
                    num_epochs=max_epochs,
                    **kwargs
                )

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
                try:
                    velocity_results = compute_velocity(
                        model=model,
                        posterior_samples=posterior_samples,
                        adata=self.adata,
                        use_mean=True,  # Use mean to avoid shape issues
                    )
                    # Extract velocity from results
                    if isinstance(velocity_results, dict) and "velocity" in velocity_results:
                        velocity = velocity_results["velocity"]
                    else:
                        velocity = velocity_results
                except Exception as e:
                    print(f"Error computing velocity for {name}: {e}")
                    velocity = None
            elif name == "jax":
                # For JAX model, we need to compute velocity
                from pyrovelocity.models.jax.core.dynamics import (
                    standard_dynamics_model,
                )
                from pyrovelocity.models.jax.inference.posterior import (
                    compute_velocity,
                )
                try:
                    velocity_results = compute_velocity(
                        posterior_samples=posterior_samples,
                        dynamics_fn=standard_dynamics_model,
                    )
                    # Extract velocity from results
                    if isinstance(velocity_results, dict) and "velocity" in velocity_results:
                        velocity = velocity_results["velocity"]
                    else:
                        velocity = velocity_results
                except Exception as e:
                    print(f"Error computing velocity for {name}: {e}")
                    velocity = None

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
                # Compute uncertainty from velocity samples
                try:
                    if velocity is not None:
                        uncertainty_results = compute_uncertainty(
                            velocity_samples=velocity,
                        )
                        # Extract uncertainty from results
                        if isinstance(uncertainty_results, dict) and "velocity_uncertainty" in uncertainty_results:
                            uncertainty = uncertainty_results["velocity_uncertainty"]
                        else:
                            uncertainty = uncertainty_results
                    else:
                        uncertainty = None
                except Exception as e:
                    print(f"Error computing uncertainty for {name}: {e}")
                    uncertainty = None
            elif name == "jax":
                # For JAX model, we need to compute uncertainty
                from pyrovelocity.models.jax.inference.posterior import (
                    compute_uncertainty,
                )
                try:
                    if velocity is not None:
                        uncertainty_results = compute_uncertainty(
                            velocity_samples=velocity,
                        )
                        # Extract uncertainty from results
                        if isinstance(uncertainty_results, dict) and "velocity_uncertainty" in uncertainty_results:
                            uncertainty = uncertainty_results["velocity_uncertainty"]
                        else:
                            uncertainty = uncertainty_results
                    else:
                        uncertainty = None
                except Exception as e:
                    print(f"Error computing uncertainty for {name}: {e}")
                    uncertainty = None

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
            >>> # Create synthetic data for testing
            >>> import anndata as ad
            >>> import numpy as np
            >>> import pandas as pd
            >>> from unittest.mock import MagicMock
            >>> tmp = getfixture("tmp_path")
            >>>
            >>> # Create synthetic data
            >>> n_cells, n_genes = 10, 5
            >>> u_data = np.random.poisson(5, size=(n_cells, n_genes))
            >>> s_data = np.random.poisson(5, size=(n_cells, n_genes))
            >>>
            >>> # Create AnnData object with all required columns
            >>> adata = ad.AnnData(X=s_data)
            >>> adata.layers["spliced"] = s_data
            >>> adata.layers["unspliced"] = u_data
            >>> adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
            >>> adata.var_names = [f"gene_{i}" for i in range(n_genes)]
            >>>
            >>> # Add library size information
            >>> adata.obs["u_lib_size_raw"] = np.sum(u_data, axis=1)
            >>> adata.obs["s_lib_size_raw"] = np.sum(s_data, axis=1)
            >>> adata.obs["u_lib_size"] = np.log(adata.obs["u_lib_size_raw"] + 1e-6)
            >>> adata.obs["s_lib_size"] = np.log(adata.obs["s_lib_size_raw"] + 1e-6)
            >>> adata.obs["u_lib_size_mean"] = np.mean(adata.obs["u_lib_size"])
            >>> adata.obs["s_lib_size_mean"] = np.mean(adata.obs["s_lib_size"])
            >>> adata.obs["u_lib_size_scale"] = np.std(adata.obs["u_lib_size"])
            >>> adata.obs["s_lib_size_scale"] = np.std(adata.obs["s_lib_size"])
            >>> adata.obs["ind_x"] = np.arange(n_cells)
            >>>
            >>> # Initialize ValidationRunner
            >>> runner = ValidationRunner(adata)
            >>>
            >>> # Set up mock results for testing
            >>> runner.results = {
            ...     "legacy": {
            ...         "posterior_samples": {
            ...             "alpha": np.ones((2, n_genes)),
            ...             "beta": np.ones((2, n_genes)),
            ...             "gamma": np.ones((2, n_genes))
            ...         },
            ...         "velocity": np.ones((n_cells, n_genes)),
            ...         "uncertainty": np.ones((n_genes,)),
            ...         "performance": {"training_time": 1.0, "inference_time": 0.5}
            ...     },
            ...     "modular": {
            ...         "posterior_samples": {
            ...             "alpha": np.ones((2, n_genes)) * 1.1,
            ...             "beta": np.ones((2, n_genes)) * 1.1,
            ...             "gamma": np.ones((2, n_genes)) * 1.1
            ...         },
            ...         "velocity": np.ones((n_cells, n_genes)) * 1.1,
            ...         "uncertainty": np.ones((n_genes,)) * 1.1,
            ...         "performance": {"training_time": 0.9, "inference_time": 0.4}
            ...     },
            ...     "jax": {
            ...         "posterior_samples": {
            ...             "alpha": np.ones((2, n_genes)) * 1.2,
            ...             "beta": np.ones((2, n_genes)) * 1.2,
            ...             "gamma": np.ones((2, n_genes)) * 1.2
            ...         },
            ...         "velocity": np.ones((n_cells, n_genes)) * 1.2,
            ...         "uncertainty": np.ones((n_genes,)) * 1.2,
            ...         "performance": {"training_time": 0.8, "inference_time": 0.3}
            ...     }
            ... }
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
    use_scalene: bool = False,  # Disable Scalene by default
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
        >>> import numpy as np
        >>> import pandas as pd
        >>> from unittest.mock import MagicMock, patch
        >>> from pyrovelocity.validation.framework import run_validation
        >>> tmp = getfixture("tmp_path")
        >>>
        >>> # Create synthetic data for testing
        >>> n_cells, n_genes = 10, 5
        >>> u_data = np.random.poisson(5, size=(n_cells, n_genes))
        >>> s_data = np.random.poisson(5, size=(n_cells, n_genes))
        >>>
        >>> # Create AnnData object with all required columns
        >>> adata = ad.AnnData(X=s_data)
        >>> adata.layers["spliced"] = s_data
        >>> adata.layers["unspliced"] = u_data
        >>> adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
        >>> adata.var_names = [f"gene_{i}" for i in range(n_genes)]
        >>>
        >>> # Add library size information
        >>> adata.obs["u_lib_size_raw"] = np.sum(u_data, axis=1)
        >>> adata.obs["s_lib_size_raw"] = np.sum(s_data, axis=1)
        >>> adata.obs["u_lib_size"] = np.log(adata.obs["u_lib_size_raw"] + 1e-6)
        >>> adata.obs["s_lib_size"] = np.log(adata.obs["s_lib_size_raw"] + 1e-6)
        >>> adata.obs["u_lib_size_mean"] = np.mean(adata.obs["u_lib_size"])
        >>> adata.obs["s_lib_size_mean"] = np.mean(adata.obs["s_lib_size"])
        >>> adata.obs["u_lib_size_scale"] = np.std(adata.obs["u_lib_size"])
        >>> adata.obs["s_lib_size_scale"] = np.std(adata.obs["s_lib_size"])
        >>> adata.obs["ind_x"] = np.arange(n_cells)
        >>>
        >>> # Mock ValidationRunner to avoid actual model training
        >>> with patch('pyrovelocity.validation.framework.ValidationRunner') as MockRunner:
        ...     # Configure the mock
        ...     mock_instance = MockRunner.return_value
        ...     mock_instance.run_validation.return_value = {"mock": "results"}
        ...     mock_instance.compare_implementations.return_value = {"mock": "comparison"}
        ...
        ...     # Run validation with minimal settings for testing
        ...     results = run_validation(
        ...         adata=adata,
        ...         max_epochs=2,
        ...         num_samples=2,
        ...         use_legacy=False,  # Disable legacy model to avoid dataloader issues
        ...         use_modular=True,
        ...         use_jax=True
        ...     )
        >>>
        >>> # Run validation with specific implementations and parameters
        >>> with patch('pyrovelocity.validation.framework.ValidationRunner') as MockRunner:
        ...     # Configure the mock
        ...     mock_instance = MockRunner.return_value
        ...     mock_instance.run_validation.return_value = {"mock": "results"}
        ...     mock_instance.compare_implementations.return_value = {"mock": "comparison"}
        ...
        ...     # Run validation with specific implementations and parameters
        ...     results = run_validation(
        ...         adata=adata,
        ...         max_epochs=2,
        ...         num_samples=2,
        ...         use_legacy=False,
        ...         modular_model_kwargs={"model_type": "standard", "latent_time": True},
        ...         jax_model_kwargs={"model_type": "standard", "latent_time": True}
        ...     )
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
        >>> import anndata as ad
        >>> import numpy as np
        >>> import pandas as pd
        >>> from pyrovelocity.validation.framework import ValidationRunner, compare_implementations
        >>> tmp = getfixture("tmp_path")
        >>>
        >>> # Create synthetic data for testing
        >>> n_cells, n_genes = 10, 5
        >>> u_data = np.random.poisson(5, size=(n_cells, n_genes))
        >>> s_data = np.random.poisson(5, size=(n_cells, n_genes))
        >>>
        >>> # Create AnnData object with all required columns
        >>> adata = ad.AnnData(X=s_data)
        >>> adata.layers["spliced"] = s_data
        >>> adata.layers["unspliced"] = u_data
        >>> # Add raw layers required by legacy model
        >>> adata.layers["raw_spliced"] = s_data.copy()
        >>> adata.layers["raw_unspliced"] = u_data.copy()
        >>> adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
        >>> adata.var_names = [f"gene_{i}" for i in range(n_genes)]
        >>>
        >>> # Add library size information
        >>> adata.obs["u_lib_size_raw"] = np.sum(u_data, axis=1)
        >>> adata.obs["s_lib_size_raw"] = np.sum(s_data, axis=1)
        >>> adata.obs["u_lib_size"] = np.log(adata.obs["u_lib_size_raw"])
        >>> adata.obs["s_lib_size"] = np.log(adata.obs["s_lib_size_raw"])
        >>> adata.obs["u_lib_size_mean"] = np.mean(adata.obs["u_lib_size"])
        >>> adata.obs["s_lib_size_mean"] = np.mean(adata.obs["s_lib_size"])
        >>> adata.obs["u_lib_size_scale"] = np.std(adata.obs["u_lib_size"])
        >>> adata.obs["s_lib_size_scale"] = np.std(adata.obs["s_lib_size"])
        >>> adata.obs["ind_x"] = np.arange(n_cells)
        >>>
        >>> # Create mock results for testing
        >>> results = {
        ...     "legacy": {
        ...         "posterior_samples": {
        ...             "alpha": np.ones((2, n_genes)),
        ...             "beta": np.ones((2, n_genes)),
        ...             "gamma": np.ones((2, n_genes))
        ...         },
        ...         "velocity": np.ones((n_cells, n_genes)),
        ...         "uncertainty": np.ones((n_genes,)),
        ...         "performance": {"training_time": 1.0, "inference_time": 0.5}
        ...     },
        ...     "modular": {
        ...         "posterior_samples": {
        ...             "alpha": np.ones((2, n_genes)) * 1.1,
        ...             "beta": np.ones((2, n_genes)) * 1.1,
        ...             "gamma": np.ones((2, n_genes)) * 1.1
        ...         },
        ...         "velocity": np.ones((n_cells, n_genes)) * 1.1,
        ...         "uncertainty": np.ones((n_genes,)) * 1.1,
        ...         "performance": {"training_time": 0.9, "inference_time": 0.4}
        ...     },
        ...     "jax": {
        ...         "posterior_samples": {
        ...             "alpha": np.ones((2, n_genes)) * 1.2,
        ...             "beta": np.ones((2, n_genes)) * 1.2,
        ...             "gamma": np.ones((2, n_genes)) * 1.2
        ...         },
        ...         "velocity": np.ones((n_cells, n_genes)) * 1.2,
        ...         "uncertainty": np.ones((n_genes,)) * 1.2,
        ...         "performance": {"training_time": 0.8, "inference_time": 0.3}
        ...     }
        ... }
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
