"""
Core interfaces for PyroVelocity's modular architecture.

This module defines the Protocol interfaces for the different component types
in the PyroVelocity modular architecture. These interfaces establish the contract
that component implementations must follow.

The modular architecture is based on Protocol interfaces rather than abstract base
classes to enable composition over inheritance. This approach allows for more
flexible component composition and easier testing.

Each Protocol defines a specific role in the RNA velocity model

- DynamicsModel: Defines the mathematical relationships for RNA velocity
- PriorModel: Defines prior distributions for model parameters
- LikelihoodModel: Defines observation distributions
- ObservationModel: Handles data preprocessing and transformation
- InferenceGuide: Defines variational distributions for inference

Components communicate through a shared context dictionary that is passed
between components during model execution. Each component updates the context
with its outputs, which are then used by subsequent components.

Examples:
    ```python
    # Create components
    dynamics_model = StandardDynamicsModel()
    prior_model = LogNormalPriorModel()
    likelihood_model = PoissonLikelihoodModel()
    observation_model = StandardObservationModel()
    guide_model = AutoGuideFactory(guide_type="AutoNormal")

    # Create the full model
    model = PyroVelocityModel(
        dynamics_model=dynamics_model,
        prior_model=prior_model,
        likelihood_model=likelihood_model,
        observation_model=observation_model,
        guide_model=guide_model,
    )

    # Use the model with AnnData
    adata = # ... AnnData object
    model.train(adata)
    ```
"""

from typing import (
    Any,
    Dict,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
    Union,
    runtime_checkable,
)

import pyro
import pyro.distributions as dist
import torch
from beartype.typing import Callable
from jaxtyping import Array, Float, Int

# Type aliases for improved readability
Tensor = torch.Tensor
BatchTensor = Float[Tensor, "batch ..."]
ParamTensor = Float[Tensor, "..."]
CountMatrix = Int[Tensor, "batch cells genes"]
RateMatrix = Float[Tensor, "batch cells genes"]
VelocityTensor = Float[Tensor, "... cells genes"]
LatentCountTensor = Float[Tensor, "... cells genes"]
KineticParamTensor = Float[Tensor, "... genes"]
ModelState = Dict[str, Any]


@runtime_checkable
class DynamicsModel(Protocol):
    """
    Protocol for dynamics models that define the RNA velocity equations.

    Dynamics models define the mathematical relationships between transcription,
    splicing, and degradation rates in the RNA velocity model. They implement
    analytical or numerical solutions to the RNA velocity differential equations.

    The standard RNA velocity model is based on the following system of ODEs

        - du/dt = α - βu
        - ds/dt = βu - γs

    Where

        - u: unspliced mRNA abundance
        - s: spliced mRNA abundance
        - α: transcription rate
        - β: splicing rate
        - γ: degradation rate

    Implementations must

    1. Provide a forward method that computes expected unspliced and spliced counts
    2. Provide a steady_state method that computes steady-state values
    3. Handle both analytical and numerical solutions as appropriate
    4. Properly validate input parameters
    5. Handle edge cases (e.g., zero rates, negative values)
    """

    def forward(
        self,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Compute the expected unspliced and spliced RNA counts based on the dynamics model.

        This method takes a context dictionary containing observed data and parameters,
        computes the expected unspliced and spliced counts according to the dynamics model,
        and updates the context with the results.

        Args:
            Dictionary containing model context with the following required keys

                - u_obs: Observed unspliced counts (BatchTensor)
                - s_obs: Observed spliced counts (BatchTensor)
                - alpha: Transcription rate (ParamTensor)
                - beta: Splicing rate (ParamTensor)
                - gamma: Degradation rate (ParamTensor)

                And optional keys

                - scaling: Scaling factor (ParamTensor)
                - t: Time points (BatchTensor)

        Returns:
            Updated context dictionary with additional keys

                - u_expected: Expected unspliced counts (BatchTensor)
                - s_expected: Expected spliced counts (BatchTensor)

        Raises:
            ValueError: If required parameters are missing from the context
        """
        ...

    def steady_state(
        self,
        alpha: ParamTensor,
        beta: ParamTensor,
        gamma: ParamTensor,
        **kwargs: Any,
    ) -> Tuple[ParamTensor, ParamTensor]:
        """
        Compute the steady-state unspliced and spliced RNA counts.

        The steady state is defined as the equilibrium point where du/dt = 0 and ds/dt = 0.
        For the standard RNA velocity model, this is:
            u_ss = α/β
            s_ss = α/γ

        Note that the steady state may not exist for all parameter combinations in
        non-standard dynamics models. Implementations should handle these
        cases appropriately.

        Args:
            alpha: Transcription rate (shape: [genes] or [batch, genes])
            beta: Splicing rate (shape: [genes] or [batch, genes])
            gamma: Degradation rate (shape: [genes] or [batch, genes])
            **kwargs: Additional model-specific parameters for extended models

        Returns:
            Tuple of (steady-state unspliced counts, steady-state spliced counts)

        Raises:
            ValueError: If any rate parameters are zero or negative
        """
        ...

    def compute_velocity(
        self,
        ut: LatentCountTensor,
        st: LatentCountTensor,
        alpha: KineticParamTensor,
        beta: KineticParamTensor,
        gamma: KineticParamTensor,
        **kwargs: Any,
    ) -> VelocityTensor:
        """
        Compute RNA velocity from latent RNA counts and kinetic parameters.

        This method computes the RNA velocity (ds/dt) based on the dynamics model's
        differential equations. For the standard RNA velocity model:
            ds/dt = βu - γs

        Where the velocity is computed using the latent (true) RNA counts ut and st
        rather than the observed counts.

        Args:
            ut: Latent unspliced RNA counts (shape: [samples, cells, genes] or [cells, genes])
            st: Latent spliced RNA counts (shape: [samples, cells, genes] or [cells, genes])
            alpha: Transcription rate (shape: [samples, genes] or [genes])
            beta: Splicing rate (shape: [samples, genes] or [genes])
            gamma: Degradation rate (shape: [samples, genes] or [genes])
            **kwargs: Additional model-specific parameters (e.g., scaling factors)

        Returns:
            RNA velocity tensor with same shape as ut/st

        Raises:
            ValueError: If tensor shapes are incompatible
        """
        ...


@runtime_checkable
class PriorModel(Protocol):
    """
    Protocol for prior models that define the prior distributions for model parameters.

    Prior models define the prior distributions for the model parameters (alpha, beta, gamma)
    and any additional parameters required by the model. They use Pyro's probabilistic
    programming framework to register samples with the model.

    Implementations must

    1. Define appropriate prior distributions for all model parameters
    2. Use Pyro's plate mechanism for batch dimensions
    3. Register all samples with Pyro
    4. Support disabling priors during guide execution (if needed)
    5. Handle hyperparameter configuration

    Example implementation

        ```python
        class LogNormalPriorModel(BasePriorModel):
            def __init__(
                self,
                name: str = "lognormal_prior",
                alpha_loc: float = -0.5,
                alpha_scale: float = 1.0,
                beta_loc: float = -0.5,
                beta_scale: float = 1.0,
                gamma_loc: float = -0.5,
                gamma_scale: float = 1.0,
            ):
                super().__init__(name=name)
                self.register_buffer("alpha_loc", torch.tensor(alpha_loc))
                self.register_buffer("alpha_scale", torch.tensor(alpha_scale))
                self.register_buffer("beta_loc", torch.tensor(beta_loc))
                self.register_buffer("beta_scale", torch.tensor(beta_scale))
                self.register_buffer("gamma_loc", torch.tensor(gamma_loc))
                self.register_buffer("gamma_scale", torch.tensor(gamma_scale))

            def sample_parameters(self, n_genes: int) -> Dict[str, torch.Tensor]:
                # Sample parameters from prior distributions
                alpha = pyro.sample(
                    f"{self.name}_alpha",
                    dist.LogNormal(self.alpha_loc, self.alpha_scale).expand([n_genes]),
                )
                beta = pyro.sample(
                    f"{self.name}_beta",
                    dist.LogNormal(self.beta_loc, self.beta_scale).expand([n_genes]),
                )
                gamma = pyro.sample(
                    f"{self.name}_gamma",
                    dist.LogNormal(self.gamma_loc, self.gamma_scale).expand([n_genes]),
                )

                return {"alpha": alpha, "beta": beta, "gamma": gamma}
        ```
    """

    def forward(
        self,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Sample model parameters from prior distributions.

        This method takes a context dictionary containing observed data,
        samples model parameters from prior distributions using Pyro's
        probabilistic programming framework, and updates the context with
        the sampled parameters.

        Args:
            context: Dictionary containing model context with the following required keys:
                - u_obs: Observed unspliced counts (BatchTensor)
                - s_obs: Observed spliced counts (BatchTensor)

                And optional keys:
                - plate: Pyro plate for batch dimensions
                - include_prior: Whether to include prior distributions

        Returns:
            Updated context dictionary with additional keys:
                - alpha: Sampled transcription rate (ParamTensor)
                - beta: Sampled splicing rate (ParamTensor)
                - gamma: Sampled degradation rate (ParamTensor)
                - Additional model-specific parameters

        Raises:
            ValueError: If required parameters are missing from the context
        """
        ...


@runtime_checkable
class LikelihoodModel(Protocol):
    """
    Protocol for likelihood models that define the observation distributions.

    Likelihood models define how the expected RNA counts (from the dynamics model)
    relate to the observed counts through probability distributions. They use Pyro's
    probabilistic programming framework to register observations with the model.

    Implementations must:
    1. Define appropriate likelihood distributions for observed data
    2. Use Pyro's observe method to register observations
    3. Handle scaling factors and other transformations
    4. Support different distribution types (e.g., Poisson, Negative Binomial)
    5. Handle zero-inflation and other data characteristics
    """

    def forward(
        self,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Define the likelihood distributions for observed data given expected values.

        This method takes a context dictionary containing observed data and expected values,
        defines likelihood distributions for the observed data, registers observations with
        Pyro, and updates the context with the likelihood distributions.

        Args:
            context: Dictionary containing model context with the following required keys:
                - u_obs: Observed unspliced counts (BatchTensor)
                - s_obs: Observed spliced counts (BatchTensor)
                - u_expected: Expected unspliced counts (BatchTensor)
                - s_expected: Expected spliced counts (BatchTensor)

                And optional keys:
                - plate: Pyro plate for batch dimensions
                - scaling: Scaling factors for observations

        Returns:
            Updated context dictionary with additional keys:
                - u_dist: Distribution for unspliced counts
                - s_dist: Distribution for spliced counts

        Raises:
            ValueError: If required parameters are missing from the context
        """
        ...


@runtime_checkable
class ObservationModel(Protocol):
    """
    Protocol for observation models that transform raw data for the model.

    Observation models handle data preprocessing, transformation, and normalization
    before feeding it into the dynamics and likelihood models. They are responsible
    for preparing the data for the model, including handling missing values, scaling,
    and other transformations.

    Implementations must:
    1. Handle data preprocessing and normalization
    2. Compute scaling factors if needed
    3. Handle missing values and zeros
    4. Support different data types and formats
    5. Maintain data integrity during transformations
    """

    def forward(
        self,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Transform observed data for model input.

        This method takes a context dictionary containing observed data,
        performs preprocessing and transformations, and updates the context
        with the transformed data and any computed scaling factors.

        Args:
            context: Dictionary containing model context with the following required keys:
                - u_obs: Observed unspliced counts (BatchTensor)
                - s_obs: Observed spliced counts (BatchTensor)

                And optional keys:
                - use_size_factor: Whether to compute size factors
                - other model-specific parameters

        Returns:
            Updated context dictionary with additional keys:
                - u_transformed: Transformed unspliced counts (BatchTensor)
                - s_transformed: Transformed spliced counts (BatchTensor)
                - scaling: Scaling factors (optional)

        Raises:
            ValueError: If required parameters are missing from the context
        """
        ...


@runtime_checkable
class InferenceGuide(Protocol):
    """
    Protocol for inference guides that define the variational distribution.

    Inference guides define the variational distribution used for approximate
    Bayesian inference in the model. They are responsible for creating guide
    functions that are compatible with the model and can be used by Pyro's
    inference algorithms.

    Implementations must:
    1. Create guide functions compatible with the model
    2. Support different variational families (e.g., Normal, Delta)
    3. Handle parameter initialization
    4. Support different inference algorithms
    5. Provide appropriate configuration options
    """

    def __call__(
        self,
        model: Union[Callable, Any],
        *args: Any,
        **kwargs: Any,
    ) -> Callable:
        """
        Create a guide function for the given model.

        This method takes a model function and creates a guide function
        that defines the variational distribution for approximate Bayesian
        inference. The guide function should be compatible with the model
        and usable by Pyro's inference algorithms.

        Args:
            model: The model function to create a guide for
            *args: Additional positional arguments passed to the guide constructor
            **kwargs: Additional keyword arguments passed to the guide constructor

        Returns:
            A guide function compatible with the model and usable by Pyro's
            inference algorithms (e.g., SVI)

        Raises:
            ValueError: If the guide cannot be created for the given model
        """
        ...
