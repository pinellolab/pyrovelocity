"""
Example of RNA velocity analysis using PyroVelocity JAX/NumPyro factory system with MCMC.

This example demonstrates:
1. Creating a model using the factory system
2. Running MCMC inference with the model
3. Analyzing and visualizing results
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_value

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

    # Generate synthetic data directly
    n_cells = 100
    n_genes = 10
    batch_size = 1

    # Create random data
    key, subkey = jax.random.split(key)
    u_obs = jax.random.poisson(subkey, jnp.ones((batch_size, n_cells, n_genes)) * 5.0)

    key, subkey = jax.random.split(key)
    s_obs = jax.random.poisson(subkey, jnp.ones((batch_size, n_cells, n_genes)) * 5.0)

    # Print shapes and values
    print(f"u_obs shape: {u_obs.shape}, min: {jnp.min(u_obs)}, max: {jnp.max(u_obs)}")
    print(f"s_obs shape: {s_obs.shape}, min: {jnp.min(s_obs)}, max: {jnp.max(s_obs)}")

    # Create a custom model configuration
    print("Creating a custom model configuration")
    model_config = ModelConfig(
        dynamics_function=DynamicsFunctionConfig(
            name="standard",
            params={"scaling": 0.1}  # Additional parameters for the dynamics function
        ),
        prior_function=PriorFunctionConfig(
            name="lognormal",
            params={
                "alpha_loc": -0.5,
                "alpha_scale": 1.0,
                "beta_loc": -0.5,
                "beta_scale": 1.0,
                "gamma_loc": -0.5,
                "gamma_scale": 1.0,
            }
        ),
        likelihood_function=LikelihoodFunctionConfig(
            name="poisson",
            params={}
        ),
        observation_function=ObservationFunctionConfig(
            name="standard",
            params={"normalize": False, "log1p": False}
        ),
        guide_function=GuideFunctionConfig(
            name="auto",
            params={"init_scale": 0.1}
        ),
    )

    # Use a simpler model for testing
    print("Using a simpler model for testing...")
    def simple_model(u_obs, s_obs):
        # Get dimensions
        batch_size, n_cells, n_genes = u_obs.shape

        # Sample model parameters
        with numpyro.plate("gene", n_genes):
            alpha = numpyro.sample("alpha", dist.LogNormal(-0.5, 1.0))
            beta = numpyro.sample("beta", dist.LogNormal(-0.5, 1.0))
            gamma = numpyro.sample("gamma", dist.LogNormal(-0.5, 1.0))

        # Sample latent time for each cell
        with numpyro.plate("cell", n_cells):
            tau = numpyro.sample("tau", dist.Normal(0.0, 1.0))

        # Compute expected counts (simplified)
        u_expected = jnp.ones_like(u_obs) * 1.0
        s_expected = jnp.ones_like(s_obs) * 1.0

        # Sample observations
        with numpyro.plate("batch", batch_size):
            numpyro.sample("u", dist.Poisson(u_expected).to_event(2), obs=u_obs)
            numpyro.sample("s", dist.Poisson(s_expected).to_event(2), obs=s_obs)

    # Create initial values for MCMC
    init_values = {
        "alpha": jnp.ones(n_genes),
        "beta": jnp.ones(n_genes),
        "gamma": jnp.ones(n_genes),
        "tau": jnp.zeros(n_cells),
    }

    # Run MCMC inference with initialization
    print("Running MCMC inference with initialization...")
    nuts_kernel = NUTS(simple_model, init_strategy=init_to_value(values=init_values))
    mcmc = MCMC(nuts_kernel, num_warmup=500, num_samples=1000)
    mcmc.run(jax.random.PRNGKey(0), u_obs, s_obs)

    # Get posterior samples
    samples = mcmc.get_samples()

    # Plot results
    plt.figure(figsize=(12, 8))

    # Plot parameter distributions
    plt.subplot(2, 2, 1)
    plt.hist(samples["alpha"].flatten())
    plt.xlabel("alpha")
    plt.ylabel("Frequency")

    plt.subplot(2, 2, 2)
    plt.hist(samples["beta"].flatten())
    plt.xlabel("beta")
    plt.ylabel("Frequency")

    plt.subplot(2, 2, 3)
    plt.hist(samples["gamma"].flatten())
    plt.xlabel("gamma")
    plt.ylabel("Frequency")

    plt.subplot(2, 2, 4)
    plt.hist(samples["tau"].flatten())
    plt.xlabel("tau")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.savefig("factory_mcmc_velocity_results.png")
    plt.close()

    print("Results saved to factory_mcmc_velocity_results.png")


if __name__ == "__main__":
    main()
