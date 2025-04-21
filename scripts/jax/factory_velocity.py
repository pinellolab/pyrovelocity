"""
Example of RNA velocity analysis using PyroVelocity JAX/NumPyro factory system.

This example demonstrates:
1. Creating a model using the factory system
2. Running inference with the model
3. Analyzing and visualizing results
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

from pyrovelocity.models.jax import (
    # Core components
    create_key,
    
    # Factory system
    DynamicsFunctionConfig,
    PriorFunctionConfig,
    LikelihoodFunctionConfig,
    ObservationFunctionConfig,
    GuideFunctionConfig,
    ModelConfig,
    create_model,
    create_standard_model,
    standard_model_config,
)


def main():
    # Set random seed for reproducibility
    key = create_key(0)

    # Generate synthetic data
    n_cells = 100
    n_genes = 10

    # Generate true parameters
    alpha = jnp.exp(jnp.array([-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4]))
    beta = jnp.exp(jnp.array([-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4]))
    gamma = jnp.exp(jnp.array([-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4]))

    # Generate latent time
    key, subkey = jax.random.split(key)
    tau = jax.random.normal(subkey, (n_cells,))

    # Compute steady state
    u0 = alpha / beta
    s0 = alpha / gamma

    # Compute expected counts using the standard dynamics model
    tau_expanded = tau[:, jnp.newaxis]  # Shape: (n_cells, 1)
    u0_expanded = u0[jnp.newaxis, :]  # Shape: (1, n_genes)
    s0_expanded = s0[jnp.newaxis, :]  # Shape: (1, n_genes)

    # Create expanded parameters dictionary
    expanded_params = {
        "alpha": alpha[jnp.newaxis, :],  # Shape: (1, n_genes)
        "beta": beta[jnp.newaxis, :],  # Shape: (1, n_genes)
        "gamma": gamma[jnp.newaxis, :],  # Shape: (1, n_genes)
    }

    # Get the standard dynamics function from the registry
    from pyrovelocity.models.jax.registry import get_dynamics
    standard_dynamics_fn = get_dynamics("standard")
    
    # Apply dynamics model to get expected counts
    u_expected, s_expected = standard_dynamics_fn(
        tau_expanded, u0_expanded, s0_expanded, expanded_params
    )

    # Generate observed counts
    key, subkey = jax.random.split(key)
    u_obs = jax.random.poisson(subkey, u_expected)

    key, subkey = jax.random.split(key)
    s_obs = jax.random.poisson(subkey, s_expected)
    
    # Add batch dimension for the model
    u_obs = u_obs[jnp.newaxis, :, :]  # Shape: (1, n_cells, n_genes)
    s_obs = s_obs[jnp.newaxis, :, :]  # Shape: (1, n_cells, n_genes)

    # Method 1: Create a standard model using the factory system
    print("Method 1: Using create_standard_model()")
    model1 = create_standard_model()

    # Method 2: Create a model with custom configuration
    print("Method 2: Using create_model() with standard_model_config()")
    model2 = create_model(standard_model_config())

    # Method 3: Create a model with explicit configuration
    print("Method 3: Using create_model() with explicit ModelConfig")
    model_config = ModelConfig(
        dynamics_function=DynamicsFunctionConfig(name="standard"),
        prior_function=PriorFunctionConfig(name="lognormal"),
        likelihood_function=LikelihoodFunctionConfig(name="poisson"),
        observation_function=ObservationFunctionConfig(name="standard"),
        guide_function=GuideFunctionConfig(name="auto"),
    )
    model3 = create_model(model_config)

    # Run inference with the first model
    print("Running inference...")
    nuts_kernel = NUTS(model1)
    mcmc = MCMC(nuts_kernel, num_warmup=500, num_samples=1000)
    mcmc.run(jax.random.PRNGKey(0), u_obs, s_obs)

    # Get posterior samples
    samples = mcmc.get_samples()

    # Plot results
    plt.figure(figsize=(12, 8))

    # Plot true vs. inferred parameters
    plt.subplot(2, 2, 1)
    plt.scatter(alpha, jnp.mean(samples["alpha"], axis=0))
    plt.plot([0, 2], [0, 2], "k--")
    plt.xlabel("True alpha")
    plt.ylabel("Inferred alpha")

    plt.subplot(2, 2, 2)
    plt.scatter(beta, jnp.mean(samples["beta"], axis=0))
    plt.plot([0, 2], [0, 2], "k--")
    plt.xlabel("True beta")
    plt.ylabel("Inferred beta")

    plt.subplot(2, 2, 3)
    plt.scatter(gamma, jnp.mean(samples["gamma"], axis=0))
    plt.plot([0, 2], [0, 2], "k--")
    plt.xlabel("True gamma")
    plt.ylabel("Inferred gamma")

    plt.subplot(2, 2, 4)
    plt.scatter(tau, jnp.mean(samples["tau"], axis=0))
    plt.plot([-3, 3], [-3, 3], "k--")
    plt.xlabel("True tau")
    plt.ylabel("Inferred tau")

    plt.tight_layout()
    plt.savefig("factory_velocity_results.png")
    plt.close()
    
    print("Results saved to factory_velocity_results.png")


if __name__ == "__main__":
    main()
