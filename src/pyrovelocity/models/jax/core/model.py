"""
NumPyro model definition for RNA velocity.

This module contains the NumPyro model definition for RNA velocity, including:

- velocity_model: Main NumPyro model for RNA velocity
- create_model: Factory function for creating models with different components

The JAX/NumPyro implementation of PyroVelocity provides several advantages:
1. JIT compilation for faster execution
2. Automatic vectorization for better hardware utilization
3. Functional programming approach for composability
4. Immutable state containers for thread safety
5. Automatic differentiation for gradient-based inference

Example:
    >>> import jax
    >>> import jax.numpy as jnp
    >>> import numpyro
    >>> from pyrovelocity.models.jax.core.model import create_model
    >>> from pyrovelocity.models.jax.core.state import ModelConfig
    >>>
    >>> # Create a model configuration
    >>> config = ModelConfig(
    ...     dynamics="standard",
    ...     likelihood="poisson",
    ...     prior="lognormal",
    ...     latent_time=True,
    ...     include_prior=True
    ... )
    >>>
    >>> # Create the model function
    >>> model_fn = create_model(config)
    >>>
    >>> # Generate synthetic data with proper type conversion
    >>> key = jax.random.PRNGKey(0)
    >>> key1, key2 = jax.random.split(key)
    >>> # Generate integer counts
    >>> u_counts = jax.random.poisson(key1, jnp.ones((10, 5)) * 5.0)
    >>> s_counts = jax.random.poisson(key2, jnp.ones((10, 5)) * 5.0)
    >>> # Convert to float arrays to satisfy type annotations
    >>> u_obs = jnp.asarray(u_counts, dtype=jnp.float32)
    >>> s_obs = jnp.asarray(s_counts, dtype=jnp.float32)
    >>>
    >>> # Run the model with numpyro seed handler
    >>> with numpyro.handlers.seed(rng_seed=0):
    ...     results = model_fn(u_obs=u_obs, s_obs=s_obs)
"""

from typing import Any, Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from beartype import beartype
from jaxtyping import Array, Float

from pyrovelocity.models.jax.core.dynamics import (
    dynamics_ode_model,
    nonlinear_dynamics_model,
    standard_dynamics_model,
)
from pyrovelocity.models.jax.core.likelihoods import (
    create_likelihood,
    negative_binomial_likelihood,
    poisson_likelihood,
)
from pyrovelocity.models.jax.core.priors import (
    informative_prior,
    lognormal_prior,
    sample_prior_parameters,
)
from pyrovelocity.models.jax.core.state import (
    ModelConfig,
)


@beartype
def velocity_model(
    u_obs: Float[Array, "cell gene"],
    s_obs: Float[Array, "cell gene"],
    u_log_library: Optional[Float[Array, "cell"]] = None,
    s_log_library: Optional[Float[Array, "cell"]] = None,
    dynamics_fn: Callable = standard_dynamics_model,
    likelihood_fn: Callable = poisson_likelihood,
    prior_fn: Callable = sample_prior_parameters,
    latent_time: bool = True,
    include_prior: bool = True,
) -> Dict[str, Float[Array, "..."]]:
    """Main NumPyro model for RNA velocity.

    This function defines the probabilistic model for RNA velocity analysis using
    NumPyro. The model consists of several components:

    1. Prior distributions for model parameters (alpha, beta, gamma)
    2. Latent time for each cell (optional)
    3. RNA dynamics equations that relate unspliced and spliced RNA counts
    4. Likelihood distributions for observed RNA counts

    The model follows the standard RNA velocity framework where:
    - alpha: Transcription rate
    - beta: Splicing rate
    - gamma: Degradation rate
    - tau: Latent time for each cell

    The dynamics are governed by the differential equations:
    du/dt = alpha - beta * u
    ds/dt = beta * u - gamma * s

    Args:
        u_obs: Observed unspliced RNA counts of shape [cell, gene]
        s_obs: Observed spliced RNA counts of shape [cell, gene]
        u_log_library: Log library size for unspliced RNA of shape [cell]
        s_log_library: Log library size for spliced RNA of shape [cell]
        dynamics_fn: Function that implements RNA dynamics equations
        likelihood_fn: Function that defines the likelihood distribution
        prior_fn: Function that defines prior distributions for parameters
        latent_time: Whether to use latent time for each cell
        include_prior: Whether to include prior distributions in the model

    Returns:
        Dictionary of model outputs including:
        - alpha: Transcription rate parameter of shape [gene]
        - beta: Splicing rate parameter of shape [gene]
        - gamma: Degradation rate parameter of shape [gene]
        - tau: Latent time of shape [cell] (if latent_time=True)
        - u_expected: Expected unspliced RNA counts of shape [cell, gene]
        - s_expected: Expected spliced RNA counts of shape [cell, gene]

    Examples:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> import numpyro
        >>> tmp = getfixture("tmp_path")
        >>> # Generate synthetic data with proper type conversion
        >>> key = jax.random.PRNGKey(0)
        >>> key1, key2 = jax.random.split(key)
        >>> # Generate integer counts
        >>> u_counts = jax.random.poisson(key1, jnp.ones((10, 5)) * 5.0)
        >>> s_counts = jax.random.poisson(key2, jnp.ones((10, 5)) * 5.0)
        >>> # Convert to float arrays to satisfy type annotations
        >>> u_obs = jnp.asarray(u_counts, dtype=jnp.float32)
        >>> s_obs = jnp.asarray(s_counts, dtype=jnp.float32)
        >>> # Run the model
        >>> with numpyro.handlers.seed(rng_seed=0):
        ...     results = velocity_model(u_obs=u_obs, s_obs=s_obs)
    """
    # Get dimensions
    num_cells, num_genes = u_obs.shape

    # Create default log library sizes if not provided
    if u_log_library is None:
        u_log_library = jnp.log(jnp.sum(u_obs, axis=1))
    if s_log_library is None:
        s_log_library = jnp.log(jnp.sum(s_obs, axis=1))

    # Sample model parameters
    with numpyro.plate("gene", num_genes):
        # Sample RNA velocity parameters from prior
        if include_prior:
            # Create a deterministic key for reproducibility
            # This is the proper way to handle random keys in JAX
            rng_key = jax.random.PRNGKey(0)

            # Use the prior function to sample parameters with a valid key
            params = prior_fn(rng_key, num_genes)

            # Register parameters with the model
            alpha = numpyro.sample("alpha", dist.Delta(params["alpha"]))
            beta = numpyro.sample("beta", dist.Delta(params["beta"]))
            gamma = numpyro.sample("gamma", dist.Delta(params["gamma"]))
        else:
            # Sample parameters directly
            alpha = numpyro.sample("alpha", dist.LogNormal(0.0, 1.0))
            beta = numpyro.sample("beta", dist.LogNormal(0.0, 1.0))
            gamma = numpyro.sample("gamma", dist.LogNormal(0.0, 1.0))

    # Sample latent time for each cell
    if latent_time:
        with numpyro.plate("cell", num_cells):
            tau = numpyro.sample("tau", dist.Normal(0.0, 1.0))
    else:
        # Use fixed time points if latent_time is False
        tau = jnp.linspace(0.0, 1.0, num_cells)
        numpyro.deterministic("tau", tau)

    # Compute RNA dynamics
    params = {"alpha": alpha, "beta": beta, "gamma": gamma}

    # Initial conditions (steady state)
    u0 = alpha / beta
    s0 = alpha / gamma

    # Reshape parameters for broadcasting
    # tau has shape (num_cells,), parameters have shape (num_genes,)
    # We need to reshape to allow proper broadcasting in the dynamics function
    tau_expanded = tau[:, jnp.newaxis]  # Shape: (num_cells, 1)
    u0_expanded = u0[jnp.newaxis, :]  # Shape: (1, num_genes)
    s0_expanded = s0[jnp.newaxis, :]  # Shape: (1, num_genes)

    # Create expanded parameters dictionary
    expanded_params = {
        "alpha": alpha[jnp.newaxis, :],  # Shape: (1, num_genes)
        "beta": beta[jnp.newaxis, :],  # Shape: (1, num_genes)
        "gamma": gamma[jnp.newaxis, :],  # Shape: (1, num_genes)
    }

    # Add scaling parameter if it exists in the parameters
    if "scaling" in params:
        expanded_params["scaling"] = params["scaling"][jnp.newaxis, :]

    # Apply dynamics model to get expected counts
    u_expected, s_expected = dynamics_fn(
        tau_expanded, u0_expanded, s0_expanded, expanded_params
    )

    # Register expected counts with the model
    numpyro.deterministic("u_expected", u_expected)
    numpyro.deterministic("s_expected", s_expected)

    # Create scaling parameters for likelihood
    scaling_params = {
        "u_log_library": u_log_library,
        "s_log_library": s_log_library,
    }

    # Get likelihood distributions
    u_dist, s_dist = likelihood_fn(u_expected, s_expected, scaling_params)

    # Observe data
    with numpyro.plate("cell_gene", num_cells * num_genes, dim=-1):
        # Reshape observations to match plate dimension
        u_obs_flat = u_obs.reshape(-1)
        s_obs_flat = s_obs.reshape(-1)

        # Reshape expected counts to match plate dimension
        u_expected_flat = u_expected.reshape(-1)
        s_expected_flat = s_expected.reshape(-1)

        # Create likelihood distributions for flattened data
        u_dist_flat, s_dist_flat = likelihood_fn(
            u_expected_flat, s_expected_flat, scaling_params
        )

        # Observe data
        numpyro.sample("u_obs", u_dist_flat, obs=u_obs_flat)
        numpyro.sample("s_obs", s_dist_flat, obs=s_obs_flat)

    # Return model outputs
    return {
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "tau": tau,
        "u_expected": u_expected,
        "s_expected": s_expected,
    }


@beartype
def create_model(
    config: ModelConfig,
) -> Callable:
    """Factory function for creating RNA velocity models with different components.

    This function creates a model function based on the provided configuration.
    It selects the appropriate dynamics function, likelihood function, and prior
    function based on the configuration, and returns a model function that can
    be used for inference.

    The factory pattern allows for flexible composition of different model components
    without modifying the core model definition. This enables users to experiment
    with different combinations of dynamics, likelihoods, and priors.

    Args:
        config: Model configuration object with the following attributes:
            - dynamics: Type of dynamics function ("standard", "nonlinear", or "ode")
            - likelihood: Type of likelihood function ("poisson" or "negative_binomial")
            - prior: Type of prior function ("lognormal" or "informative")
            - latent_time: Whether to use latent time
            - include_prior: Whether to include prior in the model

    Returns:
        A model function that takes observed data and returns model outputs

    Examples:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> from pyrovelocity.models.jax.core.state import ModelConfig
        >>> tmp = getfixture("tmp_path")
        >>> # Create a model configuration
        >>> config = ModelConfig(
        ...     dynamics="standard",
        ...     likelihood="poisson",
        ...     prior="lognormal",
        ...     latent_time=True,
        ...     include_prior=True
        ... )
        >>>
        >>> # Create a model function
        >>> model_fn = create_model(config)
        >>>
        >>> # Generate synthetic data with proper type conversion
        >>> key = jax.random.PRNGKey(0)
        >>> key1, key2 = jax.random.split(key)
        >>> # Generate integer counts
        >>> u_counts = jax.random.poisson(key1, jnp.ones((10, 5)) * 5.0)
        >>> s_counts = jax.random.poisson(key2, jnp.ones((10, 5)) * 5.0)
        >>> # Convert to float arrays to satisfy type annotations
        >>> u_obs = jnp.asarray(u_counts, dtype=jnp.float32)
        >>> s_obs = jnp.asarray(s_counts, dtype=jnp.float32)
        >>>
        >>> # Use the model function for inference with numpyro seed handler
        >>> import numpyro
        >>> with numpyro.handlers.seed(rng_seed=0):
        ...     results = model_fn(u_obs=u_obs, s_obs=s_obs)
    """
    # Select dynamics function
    if config.dynamics == "standard":
        dynamics_fn = standard_dynamics_model
    elif config.dynamics == "nonlinear":
        dynamics_fn = nonlinear_dynamics_model
    elif config.dynamics == "ode":
        dynamics_fn = dynamics_ode_model
    else:
        raise ValueError(f"Unknown dynamics type: {config.dynamics}")

    # Select likelihood function
    likelihood_fn = create_likelihood(config.likelihood)

    # Create a properly typed wrapper function for the prior
    def typed_prior_fn(
        key: Optional[jnp.ndarray], num_genes: int
    ) -> Dict[str, Float[Array, "gene"]]:
        """Wrapper for sample_prior_parameters with proper type annotations.

        This function ensures that a valid key is always passed to sample_prior_parameters.
        If key is None, it creates a new deterministic key using JAX's PRNGKey.
        This is important for JAX's functional programming model where randomness
        is explicitly managed through PRNGKeys.

        The function returns a dictionary of parameter samples with the correct
        shape and type annotations, which is important for JAX's type checking
        and shape inference.

        Args:
            key: JAX random key (can be None)
            num_genes: Number of genes to generate parameters for

        Returns:
            Dictionary of parameter samples with keys:
            - alpha: Transcription rate parameter of shape [gene]
            - beta: Splicing rate parameter of shape [gene]
            - gamma: Degradation rate parameter of shape [gene]
            - scaling: Optional scaling parameter of shape [gene] (if applicable)
        """
        # Handle the case where key is None
        if key is None:
            # Create a deterministic key for reproducibility using JAX's PRNGKey
            key = jax.random.PRNGKey(0)

        # Now we can safely call sample_prior_parameters with a valid key
        return sample_prior_parameters(key, num_genes, config.prior)

    # Create model function
    def model_fn(
        u_obs: Float[Array, "cell gene"],
        s_obs: Float[Array, "cell gene"],
        u_log_library: Optional[Float[Array, "cell"]] = None,
        s_log_library: Optional[Float[Array, "cell"]] = None,
    ) -> Dict[str, Float[Array, "..."]]:
        """NumPyro model function for RNA velocity with configured components.

        This function is the actual model function that will be used for inference.
        It's created by the factory function with the specified configuration and
        component functions.

        The function takes observed RNA counts and library sizes, and returns
        a dictionary of model outputs. It delegates to the velocity_model function
        with the configured dynamics, likelihood, and prior functions.

        Args:
            u_obs: Observed unspliced RNA counts of shape [cell, gene]
            s_obs: Observed spliced RNA counts of shape [cell, gene]
            u_log_library: Log library size for unspliced RNA of shape [cell]
            s_log_library: Log library size for spliced RNA of shape [cell]

        Returns:
            Dictionary of model outputs including:
            - alpha: Transcription rate parameter of shape [gene]
            - beta: Splicing rate parameter of shape [gene]
            - gamma: Degradation rate parameter of shape [gene]
            - tau: Latent time of shape [cell] (if latent_time=True)
            - u_expected: Expected unspliced RNA counts of shape [cell, gene]
            - s_expected: Expected spliced RNA counts of shape [cell, gene]
        """
        return velocity_model(
            u_obs=u_obs,
            s_obs=s_obs,
            u_log_library=u_log_library,
            s_log_library=s_log_library,
            dynamics_fn=dynamics_fn,
            likelihood_fn=likelihood_fn,
            prior_fn=lambda key, num_genes: sample_prior_parameters(
                key=key, num_genes=num_genes, prior_type=config.prior
            ),
            latent_time=config.latent_time,
            include_prior=config.include_prior,
        )

    return model_fn
