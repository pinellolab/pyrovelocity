"""
Basic end-to-end example of RNA velocity analysis using PyroVelocity JAX/NumPyro implementation.

This example demonstrates:
1. Loading and preprocessing data
2. Creating a velocity model
3. Performing SVI inference
4. Analyzing and visualizing results

The JAX/NumPyro implementation of PyroVelocity provides several advantages:
1. JIT compilation for faster execution
2. Automatic vectorization for better hardware utilization
3. Functional programming approach for composability
4. Immutable state containers for thread safety
5. Automatic differentiation for gradient-based inference

This script shows how to use the JAX implementation for a basic RNA velocity
analysis workflow, from data loading to visualization. The JAX implementation
follows a functional programming paradigm, where functions are composed to
create the model and perform inference.
"""

import jax
import jax.numpy as jnp
import scanpy as sc
import numpy as np
import scvelo as scv
import matplotlib.pyplot as plt
from importlib.resources import files

# Import core components
from pyrovelocity.models.jax.core.utils import create_key
from pyrovelocity.models.jax.core.state import InferenceConfig

# Import model creation
from pyrovelocity.models.jax.factory.factory import create_model

# Import data processing
from pyrovelocity.models.jax.data.anndata import prepare_anndata

# Import inference
from pyrovelocity.models.jax.inference.unified import run_inference

# Import analysis
from pyrovelocity.models.jax.inference.posterior import compute_velocity

# Import visualization
from pyrovelocity.models.jax.data.anndata import store_results as format_anndata_output

from pyrovelocity.io.serialization import load_anndata_from_json

# Fixture hash for data validation
FIXTURE_HASH = "95c80131694f2c6449a48a56513ef79cdc56eae75204ec69abde0d81a18722ae"

def load_test_data():
    """Load test data from the fixtures."""
    fixture_file_path = (
        files("pyrovelocity.tests.data") / "preprocessed_pancreas_50_7.json"
    )
    return load_anndata_from_json(
        filename=str(fixture_file_path),
        expected_hash=FIXTURE_HASH,
    )

def main():
    """Run a complete RNA velocity analysis workflow using the JAX implementation."""
    # Set random seed for reproducibility
    # In JAX, randomness is explicitly managed through PRNGKeys
    key = create_key(0)

    # 1. Load test data
    # We use a small test dataset for demonstration purposes
    print("Loading test data...")
    adata = load_test_data()
    print(f"Loaded AnnData: {adata.shape[0]} cells, {adata.shape[1]} genes")

    # 2. Prepare data for velocity model
    # The prepare_anndata function extracts data from AnnData and formats it for the model
    print("Preparing data for velocity model...")
    data_dict = prepare_anndata(
        adata,
        spliced_layer="spliced",      # Name of the spliced layer in adata.layers
        unspliced_layer="unspliced",  # Name of the unspliced layer in adata.layers
    )

    # 3. Create model configuration
    # The JAX implementation uses a configuration-based factory system
    print("Creating model configuration...")
    # Import the necessary configuration classes
    from pyrovelocity.models.jax.factory.config import (
        ModelConfig,
        DynamicsFunctionConfig,
        PriorFunctionConfig,
        LikelihoodFunctionConfig,
        ObservationFunctionConfig,
        GuideFunctionConfig,
    )

    # Create the model configuration
    # Each component is configured separately
    model_config = ModelConfig(
        dynamics_function=DynamicsFunctionConfig(name="standard"),     # Standard ODE dynamics
        prior_function=PriorFunctionConfig(name="lognormal"),          # LogNormal prior
        likelihood_function=LikelihoodFunctionConfig(name="poisson"),  # Poisson likelihood
        observation_function=ObservationFunctionConfig(name="standard"), # Standard observation
        guide_function=GuideFunctionConfig(name="auto"),               # Auto guide
    )

    # 4. Create inference configuration
    # This specifies how inference will be performed
    print("Creating inference configuration...")
    inference_config = InferenceConfig(
        num_warmup=100,               # Number of warmup steps for MCMC (not used for SVI)
        num_samples=200,              # Number of samples for MCMC (not used for SVI)
        num_chains=1,                 # Number of MCMC chains (not used for SVI)
        method="svi",                 # Inference method: "svi" or "mcmc"
        optimizer="adam",             # Optimizer for SVI
        learning_rate=0.001,          # Learning rate for the optimizer
        num_epochs=500,               # Number of training epochs
        guide_type="auto_normal",     # Type of guide for SVI
    )

    # 5. Create model
    # The factory system creates the model with the specified components
    print("Creating model...")
    model = create_model(model_config)

    # 6. Run inference
    # This performs SVI to approximate the posterior distribution
    print("Running SVI inference...")
    key, subkey = jax.random.split(key)  # Split the key for randomness

    # Extract the data from data_dict and map to the expected parameter names
    u_obs = data_dict["X_unspliced"]     # Unspliced RNA counts
    s_obs = data_dict["X_spliced"]       # Spliced RNA counts
    u_log_library = jnp.log(data_dict["u_lib_size"])  # Log library size for unspliced
    s_log_library = jnp.log(data_dict["s_lib_size"])  # Log library size for spliced

    # Add batch dimension for the model
    # The model expects inputs with shape [batch, cell, gene]
    u_obs = u_obs[jnp.newaxis, :, :]
    s_obs = s_obs[jnp.newaxis, :, :]
    u_log_library = u_log_library[jnp.newaxis, :]
    s_log_library = s_log_library[jnp.newaxis, :]

    # Run inference
    # The run_inference function is a unified interface for SVI and MCMC
    # It returns a tuple of (inference_object, inference_state)
    _, inference_state = run_inference(
        model=model,                  # The model function
        args=(),                      # Empty tuple for positional args
        kwargs={                      # Keyword arguments for the model
            "u_obs": u_obs,
            "s_obs": s_obs,
            "u_log_library": u_log_library,
            "s_log_library": s_log_library
        },
        config=inference_config,      # Inference configuration
        key=subkey,                   # Random key for reproducibility
    )

    # 7. Analyze results
    # Extract posterior samples and compute velocity
    print("Analyzing results...")
    posterior_samples = inference_state.posterior_samples

    # Print the keys in posterior_samples to see what's available
    print("Keys in posterior_samples:", list(posterior_samples.keys()))

    # Compute velocity from posterior samples
    # This uses the standard dynamics model by default
    velocity = compute_velocity(posterior_samples)

    # 8. Store results in AnnData
    # Format results for storage in AnnData
    print("Storing results in AnnData...")

    # Flatten the velocity dictionary
    results = {}
    for key, value in velocity.items():
        results[key] = value

    # Add mean parameters
    # These are the average values of the parameters across posterior samples
    results["alpha"] = jnp.mean(posterior_samples["alpha"], axis=0)  # Transcription rate
    results["beta"] = jnp.mean(posterior_samples["beta"], axis=0)    # Splicing rate
    results["gamma"] = jnp.mean(posterior_samples["gamma"], axis=0)  # Degradation rate

    # Add latent time if available
    # Latent time represents the progression of cells along a trajectory
    if "tau" in posterior_samples:
        results["latent_time"] = jnp.mean(posterior_samples["tau"], axis=0)

    # Store results in AnnData
    adata_out = format_anndata_output(adata, results)

    # Print the columns in the AnnData object to see what's available
    print("Columns in adata_out.obs:", list(adata_out.obs.columns))
    print("Keys in adata_out.uns:", list(adata_out.uns.keys()) if hasattr(adata_out, 'uns') and adata_out.uns is not None else "None")

    # Add latent time directly to the AnnData object if not already added
    if "tau" in posterior_samples and "latent_time" not in adata_out.obs.columns:
        print("Adding latent_time to AnnData object...")
        tau_mean = jnp.mean(posterior_samples["tau"], axis=0)
        # Convert to numpy array for compatibility with AnnData
        tau_mean_np = np.array(tau_mean)
        # Add to AnnData object
        adata_out.obs["latent_time"] = tau_mean_np
        print("Added latent_time column:", "latent_time" in adata_out.obs.columns)

    # 9. Visualize results
    # Create visualizations of the results
    print("Visualizing results...")

    # Plot latent time
    sc.pl.umap(adata_out, color="latent_time", title="Latent Time")
    plt.savefig("jax_velocity_latent_time.png")
    print("Latent time plot saved to jax_velocity_latent_time.png")

    # Plot velocity stream
    # This shows the direction and magnitude of RNA velocity
    if 'clusters' in adata_out.obs.columns:
        scv.pl.velocity_embedding_stream(adata_out, basis="umap", color="clusters", title="RNA Velocity")
    else:
        # Use a default color if 'clusters' doesn't exist
        scv.pl.velocity_embedding_stream(adata_out, basis="umap", title="RNA Velocity")
    plt.savefig("jax_velocity_stream.png")
    print("Velocity stream plot saved to jax_velocity_stream.png")

    # 10. Save results
    # Save the AnnData object with results for later use
    from pathlib import Path
    output_path = Path("velocity_results_jax_svi.h5ad")
    print(f"Saving results to {output_path}...")
    adata_out.write(output_path)
    print("Done!")

if __name__ == "__main__":
    main()