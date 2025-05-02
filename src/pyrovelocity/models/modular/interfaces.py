"""
Core interfaces for PyroVelocity's modular architecture.

This module defines the Protocol interfaces for the different component types
in the PyroVelocity modular architecture. These interfaces establish the contract
that component implementations must follow.

The modular architecture is based on Protocol interfaces rather than abstract base
classes to enable composition over inheritance. This approach allows for more
flexible component composition and easier testing.

Each Protocol defines a specific role in the RNA velocity model:
- DynamicsModel: Defines the mathematical relationships for RNA velocity
- PriorModel: Defines prior distributions for model parameters
- LikelihoodModel: Defines observation distributions
- ObservationModel: Handles data preprocessing and transformation
- InferenceGuide: Defines variational distributions for inference

Components communicate through a shared context dictionary that is passed
between components during model execution. Each component updates the context
with its outputs, which are then used by subsequent components.

Example:
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
ModelState = Dict[str, Any]


@runtime_checkable
class DynamicsModel(Protocol):
    """
    Protocol for dynamics models that define the RNA velocity equations.

    Dynamics models define the mathematical relationships between transcription,
    splicing, and degradation rates in the RNA velocity model. They implement
    analytical or numerical solutions to the RNA velocity differential equations.

    The standard RNA velocity model is based on the following system of ODEs:
        du/dt = α - βu
        ds/dt = βu - γs

    Where:
        u: unspliced mRNA abundance
        s: spliced mRNA abundance
        α: transcription rate
        β: splicing rate
        γ: degradation rate

    Implementations must:
    1. Provide a forward method that computes expected unspliced and spliced counts
    2. Provide a steady_state method that computes steady-state values
    3. Handle both analytical and numerical solutions as appropriate
    4. Properly validate input parameters
    5. Handle edge cases (e.g., zero rates, negative values)

    Example implementation:
        ```python
        class StandardDynamicsModel(BaseDynamicsModel):
            def _forward_impl(
                self,
                u: BatchTensor,
                s: BatchTensor,
                alpha: ParamTensor,
                beta: ParamTensor,
                gamma: ParamTensor,
                scaling: Optional[ParamTensor] = None,
                t: Optional[BatchTensor] = None,
            ) -> Tuple[BatchTensor, BatchTensor]:
                # Compute steady state
                u_ss, s_ss = self.steady_state(alpha, beta, gamma)

                # Compute expected counts
                u_exp = u_ss - (u_ss - u) * torch.exp(-beta * t)
                s_exp = s_ss - (s_ss - s) * torch.exp(-gamma * t) - (
                    beta * (u_ss - u) / (gamma - beta)
                ) * (torch.exp(-beta * t) - torch.exp(-gamma * t))

                return u_exp, s_exp

            def _steady_state_impl(
                self,
                alpha: torch.Tensor,
                beta: torch.Tensor,
                gamma: torch.Tensor,
                **kwargs: Any,
            ) -> Tuple[torch.Tensor, torch.Tensor]:
                u_ss = alpha / beta
                s_ss = alpha / gamma
                return u_ss, s_ss
        ```
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
            context: Dictionary containing model context with the following required keys:
                - u_obs: Observed unspliced counts (BatchTensor)
                - s_obs: Observed spliced counts (BatchTensor)
                - alpha: Transcription rate (ParamTensor)
                - beta: Splicing rate (ParamTensor)
                - gamma: Degradation rate (ParamTensor)

                And optional keys:
                - scaling: Scaling factor (ParamTensor)
                - t: Time points (BatchTensor)

        Returns:
            Updated context dictionary with additional keys:
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

        Args:
            alpha: Transcription rate (shape: [genes] or [batch, genes])
            beta: Splicing rate (shape: [genes] or [batch, genes])
            gamma: Degradation rate (shape: [genes] or [batch, genes])
            **kwargs: Additional model-specific parameters for extended models

        Returns:
            Tuple of (steady-state unspliced counts, steady-state spliced counts)

        Raises:
            ValueError: If any rate parameters are zero or negative

        Note:
            The steady state may not exist for all parameter combinations in
            non-standard dynamics models. Implementations should handle these
            cases appropriately.
        """
        ...


@runtime_checkable
class PriorModel(Protocol):
    """
    Protocol for prior models that define the prior distributions for model parameters.

    Prior models define the prior distributions for the model parameters (alpha, beta, gamma)
    and any additional parameters required by the model. They use Pyro's probabilistic
    programming framework to register samples with the model.

    Implementations must:
    1. Define appropriate prior distributions for all model parameters
    2. Use Pyro's plate mechanism for batch dimensions
    3. Register all samples with Pyro
    4. Support disabling priors during guide execution (if needed)
    5. Handle hyperparameter configuration

    Example implementation:
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

    Example implementation:
        ```python
        class PoissonLikelihoodModel(BaseLikelihoodModel):
            def __init__(self, name: str = "poisson_likelihood"):
                super().__init__(name=name)

            def _forward_impl(
                self,
                u_obs: BatchTensor,
                s_obs: BatchTensor,
                u_expected: BatchTensor,
                s_expected: BatchTensor,
                plate: pyro.plate,
                **kwargs: Any,
            ) -> Dict[str, Any]:
                # Register observations with Pyro
                with plate:
                    u_dist = dist.Poisson(u_expected)
                    s_dist = dist.Poisson(s_expected)

                    pyro.sample(
                        f"{self.name}_u_obs",
                        u_dist,
                        obs=u_obs,
                    )

                    pyro.sample(
                        f"{self.name}_s_obs",
                        s_dist,
                        obs=s_obs,
                    )

                return {"u_dist": u_dist, "s_dist": s_dist}
        ```
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

    Example implementation:
        ```python
        class StandardObservationModel(BaseObservationModel):
            def __init__(
                self,
                name: str = "standard_observation",
                use_size_factor: bool = True,
            ):
                super().__init__(name=name)
                self.use_size_factor = use_size_factor

            def _forward_impl(
                self,
                u_obs: BatchTensor,
                s_obs: BatchTensor,
                **kwargs: Any,
            ) -> Dict[str, Any]:
                # Compute size factors if needed
                if self.use_size_factor:
                    size_factor = torch.sum(s_obs, dim=1, keepdim=True)
                    size_factor = size_factor / torch.mean(size_factor)
                else:
                    size_factor = torch.ones_like(s_obs[:, :1])

                # Return transformed data and scaling factors
                return {
                    "u_transformed": u_obs,
                    "s_transformed": s_obs,
                    "scaling": size_factor,
                }
        ```
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

    Example implementation:
        ```python
        class AutoGuideFactory(BaseInferenceGuide):
            def __init__(
                self,
                name: str = "auto_guide",
                guide_type: str = "AutoNormal",
                init_loc_fn: Optional[Callable] = None,
            ):
                super().__init__(name=name)
                self.guide_type = guide_type
                self.init_loc_fn = init_loc_fn or pyro.infer.autoguide.init_to_median

            def __call__(
                self,
                model: Callable,
                *args: Any,
                **kwargs: Any,
            ) -> Callable:
                # Create the appropriate guide based on guide_type
                if self.guide_type == "AutoNormal":
                    return pyro.infer.autoguide.AutoNormal(
                        model,
                        init_loc_fn=self.init_loc_fn,
                    )
                elif self.guide_type == "AutoDelta":
                    return pyro.infer.autoguide.AutoDelta(
                        model,
                        init_loc_fn=self.init_loc_fn,
                    )
                else:
                    raise ValueError(f"Unknown guide type: {self.guide_type}")
        ```
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
