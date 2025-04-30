"""
PyroVelocity model implementation that composes component models.

This module provides the core model class for PyroVelocity's modular architecture,
implementing a composable approach where the full model is built from specialized
component models (dynamics, priors, likelihoods, observations, guides).

The modular architecture follows a component-based design pattern, where each
component is responsible for a specific aspect of the model. This approach enables:

1. Flexibility: Components can be swapped out independently
2. Extensibility: New components can be added without modifying existing code
3. Testability: Components can be tested in isolation
4. Reusability: Components can be reused across different models

The main class in this module is `PyroVelocityModel`, which composes the different
components into a cohesive probabilistic model. The model uses functional composition
for the forward method, enabling railway-oriented programming patterns.

Example:
    >>> import torch
    >>> import pyro
    >>> from pyrovelocity.models.modular.factory import create_standard_model
    >>> from pyrovelocity.models.modular.model import PyroVelocityModel
    >>>
    >>> # Create a standard model with default components
    >>> model = create_standard_model()
    >>>
    >>> # Generate synthetic data
    >>> u_obs = torch.randn(10, 5)  # 10 cells, 5 genes
    >>> s_obs = torch.randn(10, 5)  # 10 cells, 5 genes
    >>>
    >>> # Run the model forward
    >>> results = model.forward(u_obs=u_obs, s_obs=s_obs)
    >>>
    >>> # Access model parameters
    >>> alpha = results.get("alpha")
    >>> beta = results.get("beta")
    >>> gamma = results.get("gamma")
"""

from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
)

import anndata
import numpy as np
import pyro
import torch
from anndata import AnnData
from beartype import beartype
from jaxtyping import Array, Float, Int

from pyrovelocity.models.modular.components.base import BaseDynamicsModel
from pyrovelocity.models.modular.components.guides import AutoGuideFactory
from pyrovelocity.models.modular.components.likelihoods import (
    PoissonLikelihoodModel,
)
from pyrovelocity.models.modular.components.observations import (
    StandardObservationModel,
)
from pyrovelocity.models.modular.components.priors import LogNormalPriorModel
from pyrovelocity.models.modular.data.anndata import (
    extract_layers,
    get_library_size,
    prepare_anndata,
    store_results,
)
from pyrovelocity.models.modular.interfaces import (
    DynamicsModel as DynamicsModelProtocol,
)
from pyrovelocity.models.modular.interfaces import (
    InferenceGuide as GuideModelProtocol,
)
from pyrovelocity.models.modular.interfaces import (
    LikelihoodModel as LikelihoodModelProtocol,
)
from pyrovelocity.models.modular.interfaces import (
    ObservationModel as ObservationModelProtocol,
)
from pyrovelocity.models.modular.interfaces import (
    PriorModel as PriorModelProtocol,
)


@dataclass(frozen=True)
class ModelState:
    """
    Immutable state container for the PyroVelocityModel.

    This dataclass holds the state of all component models and ensures immutability
    through the frozen=True parameter.

    Attributes:
        dynamics_state: State of the dynamics model component
        prior_state: State of the prior model component
        likelihood_state: State of the likelihood model component
        observation_state: State of the observation model component
        guide_state: State of the inference guide component
        metadata: Optional dictionary for additional metadata
    """

    dynamics_state: Dict[str, Any]
    prior_state: Dict[str, Any]
    likelihood_state: Dict[str, Any]
    observation_state: Dict[str, Any]
    guide_state: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate state after initialization."""
        # Since the class is frozen, we can't modify attributes directly
        # This method is for validation only
        pass


class PyroVelocityModel:
    """
    Composable PyroVelocity model that integrates specialized component models.

    This class implements the core model for PyroVelocity's modular architecture,
    composing specialized component models (dynamics, priors, likelihoods,
    observations, guides) into a cohesive probabilistic model. It uses functional
    composition for the forward method, enabling railway-oriented programming
    patterns.

    The model follows a clear separation of concerns:
    - Dynamics model: Defines the velocity vector field
    - Prior model: Specifies priors for model parameters
    - Likelihood model: Defines the observation likelihood
    - Observation model: Handles data preprocessing and transformation
    - Guide model: Implements the inference guide for posterior approximation

    The PyroVelocityModel provides methods for:
    1. Running the model forward to compute expected RNA counts
    2. Training the model using SVI (Stochastic Variational Inference)
    3. Generating posterior samples for uncertainty quantification
    4. Computing RNA velocity from posterior samples
    5. Storing results in AnnData objects for visualization and analysis

    Attributes:
        dynamics_model: Component handling velocity vector field modeling
        prior_model: Component handling prior distributions
        likelihood_model: Component handling likelihood distributions
        observation_model: Component handling data observations
        guide_model: Component handling inference guide
        state: Immutable state container for all model components

    Examples:
        >>> # Create a model with standard components
        >>> from pyrovelocity.models.modular.factory import create_standard_model
        >>> model = create_standard_model()
        >>>
        >>> # Create synthetic AnnData for testing
        >>> import anndata as ad
        >>> import numpy as np
        >>> import torch
        >>> import pandas as pd
        >>> import os
        >>> # Create a temporary directory for any file operations
        >>> tmp_dir = os.path.join(os.getcwd(), "tmp_test_dir")
        >>> os.makedirs(tmp_dir, exist_ok=True)
        >>>
        >>> # Create synthetic data
        >>> n_cells, n_genes = 10, 5
        >>> u_data = np.random.poisson(5, size=(n_cells, n_genes))
        >>> s_data = np.random.poisson(5, size=(n_cells, n_genes))
        >>>
        >>> # Create AnnData object
        >>> adata = ad.AnnData(X=s_data)
        >>> adata.layers["spliced"] = s_data
        >>> adata.layers["unspliced"] = u_data
        >>> adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
        >>> adata.var_names = [f"gene_{i}" for i in range(n_genes)]
        >>>
        >>> # Set up AnnData for PyroVelocity
        >>> adata = PyroVelocityModel.setup_anndata(adata)
        >>>
        >>> # Train the model with minimal epochs for testing
        >>> model.train(adata=adata, max_epochs=2)
        >>>
        >>> # Generate posterior samples
        >>> posterior_samples = model.generate_posterior_samples(
        ...     adata=adata, num_samples=2
        ... )
        >>>
        >>> # Store results in AnnData
        >>> adata = model.store_results_in_anndata(
        ...     adata=adata, posterior_samples=posterior_samples
        ... )
        >>>
        >>> # Clean up temporary directory
        >>> import shutil
        >>> if os.path.exists(tmp_dir):
        ...     shutil.rmtree(tmp_dir)
    """

    @beartype
    def __init__(
        self,
        dynamics_model: DynamicsModelProtocol,
        prior_model: PriorModelProtocol,
        likelihood_model: LikelihoodModelProtocol,
        observation_model: ObservationModelProtocol,
        guide_model: GuideModelProtocol,
        state: Optional[ModelState] = None,
    ):
        """
        Initialize the PyroVelocityModel with component models.

        Args:
            dynamics_model: Model component for velocity vector field
            prior_model: Model component for prior distributions
            likelihood_model: Model component for likelihood distributions
            observation_model: Model component for data observations
            guide_model: Model component for inference guide
            state: Optional pre-initialized model state
        """
        # Store the component models
        self.dynamics_model = dynamics_model
        self.prior_model = prior_model
        self.likelihood_model = likelihood_model
        self.observation_model = observation_model
        self.guide_model = guide_model

        # Initialize state if not provided
        if state is None:
            self.state = ModelState(
                dynamics_state=getattr(self.dynamics_model, "state", {}),
                prior_state=getattr(self.prior_model, "state", {}),
                likelihood_state=getattr(self.likelihood_model, "state", {}),
                observation_state=getattr(self.observation_model, "state", {}),
                guide_state=getattr(self.guide_model, "state", {}),
            )
        else:
            self.state = state

    def __call__(self, *args, **kwargs):
        """Make the model callable for Pyro's autoguide.

        This method delegates to the forward method, making the model compatible
        with Pyro's autoguide system. It handles both the case where x and time_points
        are provided directly and the case where u_obs and s_obs are provided.

        Args:
            *args: Positional arguments passed to forward
            **kwargs: Keyword arguments passed to forward

        Returns:
            Result from the forward method
        """
        # Handle the case where u_obs and s_obs are provided instead of x and time_points
        if not args and "x" not in kwargs and "time_points" not in kwargs:
            if "u_obs" in kwargs and "s_obs" in kwargs:
                # Get u_obs and s_obs but don't remove them from kwargs
                u_obs = kwargs["u_obs"]
                s_obs = kwargs["s_obs"]

                # Create a dummy time_points tensor
                time_points = torch.tensor([0.0, 1.0])

                # Pass the original u_obs and s_obs to forward
                return self.forward(time_points=time_points, **kwargs)

        return self.forward(*args, **kwargs)

    @beartype
    def forward(
        self,
        x: Optional[Union[torch.Tensor, Dict[str, Any]]] = None,
        time_points: Optional[torch.Tensor] = None,
        u_obs: Optional[torch.Tensor] = None,
        s_obs: Optional[torch.Tensor] = None,
        cell_state: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Forward pass through the model using functional composition.

        This method implements the forward pass through all model components
        using functional composition, following a railway-oriented programming
        approach where each component processes the data and passes it to the
        next component.

        The forward pass follows this sequence:
        1. Observation model: Processes input data and prepares it for the dynamics model
        2. Dynamics model: Computes expected RNA counts based on dynamics equations
        3. Prior model: Applies prior distributions to model parameters
        4. Likelihood model: Computes likelihood of observed data given expected counts

        The method can be called in two ways:
        - With x and time_points: For general input data and time points
        - With u_obs and s_obs: For RNA velocity-specific input data

        Args:
            x: Optional input data tensor of shape [batch_size, n_features]
            time_points: Optional time points for the dynamics model of shape [n_time_points]
            u_obs: Observed unspliced RNA counts of shape [batch_size, n_genes]
            s_obs: Observed spliced RNA counts of shape [batch_size, n_genes]
            cell_state: Optional dictionary with cell state information
            **kwargs: Additional keyword arguments passed to component models

        Returns:
            Dictionary containing model outputs and intermediate results, including:
            - alpha: Transcription rate parameter
            - beta: Splicing rate parameter
            - gamma: Degradation rate parameter
            - u_expected: Expected unspliced RNA counts
            - s_expected: Expected spliced RNA counts
            - tau: Latent time (if enabled)
            - Other model-specific outputs

        Examples:
            >>> # Run forward pass with RNA count data
            >>> import torch
            >>> import os
            >>> from pyrovelocity.models.modular.factory import create_standard_model
            >>>
            >>> # Create a temporary directory for any file operations
            >>> tmp_dir = os.path.join(os.getcwd(), "tmp_test_dir")
            >>> os.makedirs(tmp_dir, exist_ok=True)
            >>>
            >>> # Create model and synthetic data
            >>> model = create_standard_model()
            >>> u_obs = torch.randn(10, 5)  # 10 cells, 5 genes
            >>> s_obs = torch.randn(10, 5)  # 10 cells, 5 genes
            >>>
            >>> # Run forward pass
            >>> results = model.forward(u_obs=u_obs, s_obs=s_obs)
            >>>
            >>> # Access results
            >>> alpha = results["alpha"]
            >>> beta = results["beta"]
            >>> gamma = results["gamma"]
            >>> u_expected = results["u_expected"]
            >>> s_expected = results["s_expected"]
            >>>
            >>> # Clean up temporary directory
            >>> import shutil
            >>> if os.path.exists(tmp_dir):
            ...     shutil.rmtree(tmp_dir)
        """
        # Initialize the context dictionary to pass between components
        context = {
            "cell_state": cell_state or {},
            **kwargs,
        }

        # Add optional parameters to context if provided
        if x is not None:
            context["x"] = x
        if time_points is not None:
            context["time_points"] = time_points
        if u_obs is not None:
            context["u_obs"] = u_obs
        if s_obs is not None:
            context["s_obs"] = s_obs

        # Process data through the observation model
        observation_context = self.observation_model.forward(context)

        # Process through the dynamics model
        dynamics_context = self.dynamics_model.forward(observation_context)

        # Apply prior distributions
        prior_context = self.prior_model.forward(dynamics_context)

        # Apply likelihood model
        likelihood_context = self.likelihood_model.forward(prior_context)

        # Return the final context with all model outputs
        return likelihood_context

    @beartype
    def guide(
        self,
        x: Optional[Union[torch.Tensor, Dict[str, Any]]] = None,
        time_points: Optional[torch.Tensor] = None,
        u_obs: Optional[torch.Tensor] = None,
        s_obs: Optional[torch.Tensor] = None,
        cell_state: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Inference guide for the model used in variational inference.

        This method delegates to the guide model component to implement
        the inference guide for posterior approximation. The guide defines
        the variational distribution that approximates the true posterior
        distribution over model parameters.

        During training, this guide is used by Pyro's SVI (Stochastic Variational
        Inference) to optimize the variational parameters. After training, the guide
        can be used to sample from the approximate posterior distribution.

        The method handles two cases:
        1. When data is provided (x, u_obs, or s_obs): Processes data through the
           observation model before passing to the guide
        2. When no data is provided: Assumes posterior sampling mode and passes
           directly to the guide

        Args:
            x: Optional input data tensor of shape [batch_size, n_features]
            time_points: Optional time points for the dynamics model of shape [n_time_points]
            u_obs: Observed unspliced RNA counts of shape [batch_size, n_genes]
            s_obs: Observed spliced RNA counts of shape [batch_size, n_genes]
            cell_state: Optional dictionary with cell state information
            **kwargs: Additional keyword arguments passed to the guide model

        Returns:
            Dictionary containing guide outputs, typically including variational parameters

        Examples:
            >>> # Use guide for posterior sampling
            >>> import torch
            >>> from pyrovelocity.models.modular.factory import create_standard_model
            >>>
            >>> # Create model and synthetic data
            >>> model = create_standard_model()
            >>> u_obs = torch.randn(10, 5)  # 10 cells, 5 genes
            >>> s_obs = torch.randn(10, 5)  # 10 cells, 5 genes
            >>>
            >>> # Train model (simplified)
            >>> model.guide_model.create_guide(model.forward)
            >>>
            >>> # Use guide for inference
            >>> guide_results = model.guide(u_obs=u_obs, s_obs=s_obs)
        """
        # Initialize the context dictionary to pass to the guide
        context = {
            "cell_state": cell_state or {},
            **kwargs,
        }

        # Add optional parameters to context if provided
        if x is not None:
            context["x"] = x
        if time_points is not None:
            context["time_points"] = time_points
        if u_obs is not None:
            context["u_obs"] = u_obs
        if s_obs is not None:
            context["s_obs"] = s_obs

        # If no data is provided, we're in posterior sampling mode
        # In this case, we can skip the observation model and go directly to the guide
        if x is None and u_obs is None and s_obs is None:
            # Pass directly to the guide model without observation processing
            guide_fn = self.guide_model.get_guide()
            return guide_fn(context)

        # Process through the observation model first to prepare data
        observation_context = self.observation_model.forward(context)

        # Delegate to the guide model
        guide_fn = self.guide_model.get_guide()
        return guide_fn(observation_context)

    @property
    def name(self) -> str:
        """Return the model name."""
        return "PyroVelocityModel"

    def get_state(self) -> ModelState:
        """Return the current model state."""
        return self.state

    def with_state(self, state: ModelState) -> "PyroVelocityModel":
        """
        Create a new model instance with the given state.

        This method implements the immutable state pattern, returning a new
        model instance with the updated state rather than modifying the
        current instance.

        Args:
            state: New model state

        Returns:
            New PyroVelocityModel instance with the given state
        """
        return PyroVelocityModel(
            dynamics_model=self.dynamics_model,
            prior_model=self.prior_model,
            likelihood_model=self.likelihood_model,
            observation_model=self.observation_model,
            guide_model=self.guide_model,
            state=state,
        )

    @beartype
    def predict(
        self, x: Any, time_points: Optional[Any] = None, **kwargs
    ) -> Dict[str, Any]:
        """
        Generate predictions using the model without sampling from the posterior.

        This method processes the input data through the observation model and dynamics model
        to generate predictions. Unlike the forward method, this method is focused on
        prediction rather than inference, and doesn't apply prior distributions or
        likelihood models.

        The method is useful for:
        1. Making predictions with fixed parameter values
        2. Simulating RNA dynamics over time
        3. Testing the dynamics model in isolation

        Args:
            x: Input data tensor of shape [batch_size, n_features]
            time_points: Time points for the dynamics model of shape [n_time_points]
            **kwargs: Additional keyword arguments passed to component models

        Returns:
            Dictionary containing predictions, typically including:
            - u_predicted: Predicted unspliced RNA counts
            - s_predicted: Predicted spliced RNA counts
            - velocity: Predicted RNA velocity

        Examples:
            >>> # Make predictions with the model
            >>> import torch
            >>> from pyrovelocity.models.modular.factory import create_standard_model
            >>>
            >>> # Create model and synthetic data
            >>> model = create_standard_model()
            >>> x = torch.randn(10, 5)  # 10 cells, 5 genes
            >>> time_points = torch.linspace(0, 1, 10)  # 10 time points
            >>>
            >>> # Generate predictions
            >>> predictions = model.predict(x, time_points)
            >>>
            >>> # Access predictions
            >>> u_predicted = predictions.get("u_predicted")
            >>> s_predicted = predictions.get("s_predicted")
            >>> velocity = predictions.get("velocity")
        """
        # Initialize the context dictionary
        context = {
            "x": x,
            "time_points": time_points,
            **kwargs,
        }

        # Process data through the observation model
        observation_context = self.observation_model.forward(context)

        # Process through the dynamics model
        dynamics_context = self.dynamics_model.forward(observation_context)

        # Extract predictions
        predictions = dynamics_context.get("predictions", {})

        return predictions

    @beartype
    def predict_future_states(
        self, current_state: Tuple[torch.Tensor, torch.Tensor], time_delta: torch.Tensor, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict future RNA states based on current state and time delta.

        This method delegates to the dynamics model to predict future states of
        unspliced and spliced RNA counts given the current state and a time delta.
        It's useful for simulating RNA dynamics over time and predicting cell
        trajectories.

        The method solves the RNA velocity differential equations to predict
        how the RNA counts will evolve over the specified time delta.

        Args:
            current_state: Current state as a tuple of (u, s) where:
                - u: Unspliced RNA counts of shape [batch_size, n_genes]
                - s: Spliced RNA counts of shape [batch_size, n_genes]
            time_delta: Time delta for prediction, either a scalar or tensor of shape [batch_size]
            **kwargs: Additional keyword arguments passed to the dynamics model,
                including parameters like alpha, beta, gamma

        Returns:
            Tuple of (u_future, s_future) representing the predicted future state:
                - u_future: Predicted unspliced RNA counts of shape [batch_size, n_genes]
                - s_future: Predicted spliced RNA counts of shape [batch_size, n_genes]

        Examples:
            >>> # Predict future states
            >>> import torch
            >>> from pyrovelocity.models.modular.factory import create_standard_model
            >>>
            >>> # Create model and synthetic data
            >>> model = create_standard_model()
            >>> u_current = torch.ones(10, 5)  # 10 cells, 5 genes
            >>> s_current = torch.ones(10, 5)  # 10 cells, 5 genes
            >>> current_state = (u_current, s_current)
            >>>
            >>> # Predict future state after time_delta
            >>> time_delta = torch.tensor(0.5)
            >>> u_future, s_future = model.predict_future_states(
            ...     current_state, time_delta
            ... )
        """
        # Delegate to the dynamics model
        return self.dynamics_model.predict_future_states(
            current_state, time_delta, **kwargs
        )

    @classmethod
    @beartype
    def setup_anndata(
        cls,
        adata: AnnData,
        spliced_layer: str = "spliced",
        unspliced_layer: str = "unspliced",
        use_raw: bool = False,
        **kwargs
    ) -> AnnData:
        """
        Set up AnnData object for use with PyroVelocityModel.

        This method prepares the AnnData object for use with the PyroVelocityModel by:
        1. Verifying that required layers exist
        2. Computing library sizes for normalization
        3. Computing library size statistics
        4. Adding indices for batch processing

        The method modifies the AnnData object in-place by adding several new columns
        to the obs attribute:
        - u_lib_size_raw: Raw library size for unspliced counts
        - s_lib_size_raw: Raw library size for spliced counts
        - u_lib_size: Log-transformed library size for unspliced counts
        - s_lib_size: Log-transformed library size for spliced counts
        - u_lib_size_mean: Mean of log library size for unspliced counts
        - s_lib_size_mean: Mean of log library size for spliced counts
        - u_lib_size_scale: Standard deviation of log library size for unspliced counts
        - s_lib_size_scale: Standard deviation of log library size for spliced counts
        - ind_x: Indices for batch processing

        Args:
            adata: AnnData object to set up
            spliced_layer: Name of the spliced layer in adata.layers
            unspliced_layer: Name of the unspliced layer in adata.layers
            use_raw: Whether to use raw data from adata.raw
            **kwargs: Additional keyword arguments (unused)

        Returns:
            Modified AnnData object with additional columns

        Raises:
            ValueError: If required layers are not found in the AnnData object

        Examples:
            >>> # Set up AnnData for PyroVelocity
            >>> import anndata as ad
            >>> import numpy as np
            >>> import os
            >>> from pyrovelocity.models.modular.model import PyroVelocityModel
            >>>
            >>> # Create a temporary directory for any file operations
            >>> tmp_dir = os.path.join(os.getcwd(), "tmp_test_dir")
            >>> os.makedirs(tmp_dir, exist_ok=True)
            >>>
            >>> # Create synthetic data
            >>> n_cells, n_genes = 10, 5
            >>> u_data = np.random.poisson(5, size=(n_cells, n_genes))
            >>> s_data = np.random.poisson(5, size=(n_cells, n_genes))
            >>>
            >>> # Create AnnData object
            >>> adata = ad.AnnData(X=s_data)
            >>> adata.layers["spliced"] = s_data
            >>> adata.layers["unspliced"] = u_data
            >>> adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
            >>> adata.var_names = [f"gene_{i}" for i in range(n_genes)]
            >>>
            >>> # Set up AnnData
            >>> adata = PyroVelocityModel.setup_anndata(
            ...     adata,
            ...     spliced_layer="spliced",
            ...     unspliced_layer="unspliced"
            ... )
            >>>
            >>> # Check that library sizes were computed
            >>> print("u_lib_size in adata.obs:", "u_lib_size" in adata.obs)
            >>> print("s_lib_size in adata.obs:", "s_lib_size" in adata.obs)
            >>>
            >>> # Clean up temporary directory
            >>> import shutil
            >>> if os.path.exists(tmp_dir):
            ...     shutil.rmtree(tmp_dir)
        """
        # Make sure the required layers exist
        required_layers = [spliced_layer, unspliced_layer]
        for layer in required_layers:
            if layer not in adata.layers:
                raise ValueError(f"Layer '{layer}' not found in AnnData object")

        # Compute library sizes if they don't exist
        if "u_lib_size_raw" not in adata.obs:
            u, _ = extract_layers(adata, spliced_layer, unspliced_layer, use_raw)
            adata.obs["u_lib_size_raw"] = u.sum(dim=1).cpu().numpy()

        if "s_lib_size_raw" not in adata.obs:
            _, s = extract_layers(adata, spliced_layer, unspliced_layer, use_raw)
            adata.obs["s_lib_size_raw"] = s.sum(dim=1).cpu().numpy()

        # Compute log library sizes
        if "u_lib_size" not in adata.obs:
            adata.obs["u_lib_size"] = np.log(adata.obs["u_lib_size_raw"].astype(float) + 1e-6)

        if "s_lib_size" not in adata.obs:
            adata.obs["s_lib_size"] = np.log(adata.obs["s_lib_size_raw"].astype(float) + 1e-6)

        # Compute library size statistics
        if "u_lib_size_mean" not in adata.obs:
            adata.obs["u_lib_size_mean"] = adata.obs["u_lib_size"].mean()

        if "s_lib_size_mean" not in adata.obs:
            adata.obs["s_lib_size_mean"] = adata.obs["s_lib_size"].mean()

        if "u_lib_size_scale" not in adata.obs:
            adata.obs["u_lib_size_scale"] = adata.obs["u_lib_size"].std()

        if "s_lib_size_scale" not in adata.obs:
            adata.obs["s_lib_size_scale"] = adata.obs["s_lib_size"].std()

        # Add indices for batch processing
        if "ind_x" not in adata.obs:
            adata.obs["ind_x"] = np.arange(adata.n_obs)

        return adata

    @beartype
    def train(
        self,
        adata: AnnData,
        max_epochs: int = 1000,
        batch_size: Optional[int] = None,
        train_size: float = 0.8,
        valid_size: Optional[float] = 0.2,
        early_stopping: bool = True,
        early_stopping_patience: int = 10,
        learning_rate: float = 0.01,
        use_gpu: Union[str, bool, int] = "auto",
        **kwargs
    ) -> "PyroVelocityModel":
        """
        Train the model using the provided AnnData object.

        This method trains the model using the data in the AnnData object. It performs
        the following steps:
        1. Prepares data from the AnnData object
        2. Sets up the inference configuration
        3. Runs SVI (Stochastic Variational Inference)
        4. Stores the inference state in the model state

        The method supports mini-batch training, early stopping, and GPU acceleration.
        After training, the model's state is updated with the trained parameters.

        Args:
            adata: AnnData object containing the data
            max_epochs: Maximum number of epochs to train for
            batch_size: Batch size for mini-batch training (None for full-batch)
            train_size: Fraction of data to use for training
            valid_size: Fraction of data to use for validation
            early_stopping: Whether to use early stopping
            early_stopping_patience: Patience for early stopping
            learning_rate: Learning rate for the optimizer
            use_gpu: Whether to use GPU for training ("auto", True, False, or GPU index)
            **kwargs: Additional keyword arguments for training

        Returns:
            The model instance with updated state (for method chaining)

        Examples:
            >>> # Create a model and train it
            >>> from pyrovelocity.models.modular.factory import create_standard_model
            >>> import anndata as ad
            >>> import numpy as np
            >>> import os
            >>>
            >>> # Create a temporary directory for any file operations
            >>> tmp_dir = os.path.join(os.getcwd(), "tmp_test_dir")
            >>> os.makedirs(tmp_dir, exist_ok=True)
            >>>
            >>> # Create synthetic data
            >>> n_cells, n_genes = 10, 5
            >>> u_data = np.random.poisson(5, size=(n_cells, n_genes))
            >>> s_data = np.random.poisson(5, size=(n_cells, n_genes))
            >>>
            >>> # Create AnnData object
            >>> adata = ad.AnnData(X=s_data)
            >>> adata.layers["spliced"] = s_data
            >>> adata.layers["unspliced"] = u_data
            >>> adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
            >>> adata.var_names = [f"gene_{i}" for i in range(n_genes)]
            >>>
            >>> # Prepare AnnData
            >>> adata = PyroVelocityModel.setup_anndata(adata)
            >>>
            >>> # Create and train the model
            >>> model = create_standard_model()
            >>> model.train(
            ...     adata=adata,
            ...     max_epochs=2,  # Use small number for testing
            ...     batch_size=5,
            ...     learning_rate=0.01,
            ...     use_gpu=False  # Set to True if GPU is available
            ... )
            >>>
            >>> # Clean up temporary directory
            >>> import shutil
            >>> if os.path.exists(tmp_dir):
            ...     shutil.rmtree(tmp_dir)
        """
        # Enable Pyro validation
        pyro.enable_validation(True)

        # Prepare data from AnnData
        data_dict = prepare_anndata(adata)

        # Extract unspliced and spliced data
        u_obs = data_dict["X_unspliced"]
        s_obs = data_dict["X_spliced"]

        # Extract library sizes
        u_lib_size = data_dict.get("u_lib_size")
        s_lib_size = data_dict.get("s_lib_size")

        # Move to GPU if requested
        if use_gpu == "auto":
            use_gpu = torch.cuda.is_available()

        if use_gpu:
            device = torch.device("cuda" if isinstance(use_gpu, bool) else f"cuda:{use_gpu}")
            u_obs = u_obs.to(device)
            s_obs = s_obs.to(device)
            if u_lib_size is not None:
                u_lib_size = u_lib_size.to(device)
            if s_lib_size is not None:
                s_lib_size = s_lib_size.to(device)

        # Create training data dictionary
        train_data = {
            "u_obs": u_obs,
            "s_obs": s_obs,
        }

        if u_lib_size is not None:
            train_data["u_log_library"] = u_lib_size
        if s_lib_size is not None:
            train_data["s_log_library"] = s_lib_size

        # Import inference utilities
        from pyrovelocity.models.modular.inference.config import InferenceConfig
        from pyrovelocity.models.modular.inference.unified import run_inference

        # Create inference configuration
        inference_config = InferenceConfig(
            method="svi",
            num_epochs=max_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            early_stopping=early_stopping,
            early_stopping_patience=early_stopping_patience,
            train_size=train_size,
            valid_size=valid_size,
            **kwargs
        )

        # Ensure the guide is created before running inference
        # This is necessary because the guide needs to be created with the model
        if isinstance(self.guide_model, AutoGuideFactory):
            # Create the guide with the model
            self.guide_model.create_guide(self.forward)

        # Run inference
        inference_state = run_inference(
            model=self.forward,
            guide=self.guide,
            kwargs=train_data,
            config=inference_config,
            seed=kwargs.get("seed", None),
        )

        # Store inference state in model state
        self.state = ModelState(
            dynamics_state=self.state.dynamics_state,
            prior_state=self.state.prior_state,
            likelihood_state=self.state.likelihood_state,
            observation_state=self.state.observation_state,
            guide_state=inference_state.params,
            metadata={
                "inference_state": inference_state,
                "training_config": inference_config,
            }
        )

        # Return the model for method chaining
        return self

    @beartype
    def generate_posterior_samples(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        batch_size: Optional[int] = None,
        num_samples: int = 100,
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """
        Generate posterior samples using the trained model.

        This method generates posterior samples for model parameters using the trained model.
        The samples represent the posterior distribution over parameters like alpha, beta, and gamma,
        which can be used for uncertainty quantification and downstream analysis.

        The method requires that the model has been trained first. It uses the guide
        from the inference state to sample from the approximate posterior distribution.

        Args:
            adata: Optional AnnData object (if None, uses the one from training)
            indices: Optional sequence of indices to generate samples for (for batch processing)
            batch_size: Batch size for generating samples (for memory efficiency)
            num_samples: Number of posterior samples to generate
            **kwargs: Additional keyword arguments including 'seed' for reproducibility

        Returns:
            Dictionary of posterior samples with keys for model parameters (alpha, beta, gamma, etc.)
            and values as numpy arrays of shape [num_samples, num_genes]

        Examples:
            >>> # Generate posterior samples after training
            >>> from pyrovelocity.models.modular.factory import create_standard_model
            >>> import anndata as ad
            >>> import numpy as np
            >>> import os
            >>>
            >>> # Create a temporary directory for any file operations
            >>> tmp_dir = os.path.join(os.getcwd(), "tmp_test_dir")
            >>> os.makedirs(tmp_dir, exist_ok=True)
            >>>
            >>> # Create synthetic data
            >>> n_cells, n_genes = 10, 5
            >>> u_data = np.random.poisson(5, size=(n_cells, n_genes))
            >>> s_data = np.random.poisson(5, size=(n_cells, n_genes))
            >>>
            >>> # Create AnnData object
            >>> adata = ad.AnnData(X=s_data)
            >>> adata.layers["spliced"] = s_data
            >>> adata.layers["unspliced"] = u_data
            >>> adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
            >>> adata.var_names = [f"gene_{i}" for i in range(n_genes)]
            >>>
            >>> # Prepare AnnData
            >>> adata = PyroVelocityModel.setup_anndata(adata)
            >>>
            >>> # Create, train the model, and generate samples
            >>> model = create_standard_model()
            >>> model.train(adata=adata, max_epochs=2)  # Use small number for testing
            >>> posterior_samples = model.generate_posterior_samples(
            ...     adata=adata, num_samples=2, seed=42  # Use small number for testing
            ... )
            >>>
            >>> # Access parameter samples
            >>> alpha_samples = posterior_samples["alpha"]  # Shape: [2, num_genes]
            >>> beta_samples = posterior_samples["beta"]    # Shape: [2, num_genes]
            >>> gamma_samples = posterior_samples["gamma"]  # Shape: [2, num_genes]
            >>>
            >>> # Clean up temporary directory
            >>> import shutil
            >>> if os.path.exists(tmp_dir):
            ...     shutil.rmtree(tmp_dir)
        """
        # Import inference utilities
        from pyrovelocity.models.modular.inference.posterior import (
            sample_posterior,
        )

        # Get inference state from model state
        inference_state = self.state.metadata.get("inference_state")
        if inference_state is None:
            raise ValueError("Model has not been trained yet")

        # Generate posterior samples
        posterior_samples = sample_posterior(
            model=self,
            state=inference_state,
            num_samples=num_samples,
            seed=kwargs.get("seed", None),
        )

        # Convert PyTorch tensors to NumPy arrays
        posterior_samples_np = {
            k: v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v
            for k, v in posterior_samples.items()
        }

        return posterior_samples_np

    @beartype
    def store_results_in_anndata(
        self,
        adata: AnnData,
        posterior_samples: Dict[str, np.ndarray],
        model_name: str = "pyrovelocity",
    ) -> AnnData:
        """
        Store model results in AnnData for visualization and analysis.

        This method computes velocity from posterior samples and stores all results
        in the AnnData object. The results are stored in various slots of the AnnData object:
        - adata.obs: Cell-level information (e.g., latent time)
        - adata.var: Gene-level information (e.g., kinetic parameters)
        - adata.layers: Matrix data (e.g., velocity, uncertainty)
        - adata.uns: Unstructured data (e.g., model configuration)

        The stored results can be used with visualization tools like scVelo for
        creating velocity stream plots and other visualizations.

        Args:
            adata: AnnData object to store results in
            posterior_samples: Dictionary of posterior samples from generate_posterior_samples()
            model_name: Name prefix to use for storing results (to distinguish between models)

        Returns:
            Updated AnnData object with model results

        Examples:
            >>> # Store results in AnnData after generating posterior samples
            >>> from pyrovelocity.models.modular.factory import create_standard_model
            >>> import anndata as ad
            >>> import numpy as np
            >>> tmp = getfixture("tmp_path")  # For any temporary file operations
            >>>
            >>> # Create synthetic data
            >>> n_cells, n_genes = 10, 5
            >>> u_data = np.random.poisson(5, size=(n_cells, n_genes))
            >>> s_data = np.random.poisson(5, size=(n_cells, n_genes))
            >>>
            >>> # Create AnnData object
            >>> adata = ad.AnnData(X=s_data)
            >>> adata.layers["spliced"] = s_data
            >>> adata.layers["unspliced"] = u_data
            >>> adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
            >>> adata.var_names = [f"gene_{i}" for i in range(n_genes)]
            >>> # Add UMAP coordinates for visualization
            >>> adata.obsm = {}
            >>> adata.obsm["X_umap"] = np.random.normal(0, 1, size=(n_cells, 2))
            >>> # Add cluster information
            >>> adata.obs["clusters"] = np.random.choice(["A", "B", "C"], size=n_cells)
            >>>
            >>> # Prepare AnnData
            >>> adata = PyroVelocityModel.setup_anndata(adata)
            >>>
            >>> # Create, train model, and generate samples
            >>> model = create_standard_model()
            >>> model.train(adata=adata, max_epochs=2)  # Use small number for testing
            >>> posterior_samples = model.generate_posterior_samples(
            ...     adata=adata, num_samples=2  # Use small number for testing
            ... )
            >>>
            >>> # Store results in AnnData
            >>> adata_out = model.store_results_in_anndata(
            ...     adata=adata,
            ...     posterior_samples=posterior_samples,
            ...     model_name="standard_model"
            ... )
            >>>
            >>> # Check that results were stored
            >>> assert "standard_model_alpha" in adata_out.var
            >>> assert "standard_model_beta" in adata_out.var
            >>> assert "standard_model_gamma" in adata_out.var
        """
        # Import inference utilities
        from pyrovelocity.models.modular.inference.posterior import (
            compute_velocity,
        )

        # Compute velocity from posterior samples
        velocity_results = compute_velocity(
            model=self,
            posterior_samples=posterior_samples,
            adata=adata,
        )

        # Combine results
        results = {**posterior_samples, **velocity_results}

        # Store results in AnnData
        return store_results(adata, results, model_name)
