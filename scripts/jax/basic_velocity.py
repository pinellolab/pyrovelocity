"""
Basic end-to-end example of RNA velocity analysis using PyroVelocity JAX/NumPyro implementation.

This example demonstrates:
1. Loading and preprocessing data
2. Creating a velocity model
3. Performing SVI inference
4. Analyzing and visualizing results
"""

import jax
import jax.numpy as jnp
import scanpy as sc
import numpy as np
import scvelo as scv
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
    # Set random seed for reproducibility
    key = create_key(0)

    # 1. Load test data
    print("Loading test data...")
    adata = load_test_data()
    print(f"Loaded AnnData: {adata.shape[0]} cells, {adata.shape[1]} genes")

    # 2. Prepare data for velocity model
    print("Preparing data for velocity model...")
    data_dict = prepare_anndata(
        adata,
        spliced_layer="spliced",
        unspliced_layer="unspliced",
    )

    # 3. Create model configuration
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
    model_config = ModelConfig(
        dynamics_function=DynamicsFunctionConfig(name="standard"),
        prior_function=PriorFunctionConfig(name="lognormal"),
        likelihood_function=LikelihoodFunctionConfig(name="poisson"),
        observation_function=ObservationFunctionConfig(name="standard"),
        guide_function=GuideFunctionConfig(name="auto"),
    )

    # 4. Create inference configuration
    print("Creating inference configuration...")
    inference_config = InferenceConfig(
        num_warmup=100,  # Reduced for example
        num_samples=200,  # Reduced for example
        num_chains=1,
        method="svi",
        optimizer="adam",
        learning_rate=0.001,  # Lower learning rate for more stable training
        num_epochs=500,  # More epochs to ensure convergence
        guide_type="auto_normal",  # Explicitly specify the guide type
    )

    # 5. Create model
    print("Creating model...")
    model = create_model(model_config)

    # 6. Run inference
    print("Running SVI inference...")
    key, subkey = jax.random.split(key)

    # Extract the data from data_dict and map to the expected parameter names
    u_obs = data_dict["X_unspliced"]
    s_obs = data_dict["X_spliced"]
    u_log_library = jnp.log(data_dict["u_lib_size"])
    s_log_library = jnp.log(data_dict["s_lib_size"])

    # Add batch dimension for the model
    u_obs = u_obs[jnp.newaxis, :, :]
    s_obs = s_obs[jnp.newaxis, :, :]
    u_log_library = u_log_library[jnp.newaxis, :]
    s_log_library = s_log_library[jnp.newaxis, :]

    # The run_inference function expects (model, args, kwargs, config, key)
    # It will create the guide internally based on the guide_type in inference_config
    # run_inference returns a tuple of (inference_object, inference_state)
    _, inference_state = run_inference(
        model=model,
        args=(),  # Empty tuple for positional args
        kwargs={
            "u_obs": u_obs,
            "s_obs": s_obs,
            "u_log_library": u_log_library,
            "s_log_library": s_log_library
        },
        config=inference_config,
        key=subkey,
    )

    # 7. Analyze results
    print("Analyzing results...")
    posterior_samples = inference_state.posterior_samples

    # Print the keys in posterior_samples to see what's available
    print("Keys in posterior_samples:", list(posterior_samples.keys()))

    # Use the default dynamics_fn (standard_dynamics_model) by not providing a second argument
    velocity = compute_velocity(posterior_samples)

    # 8. Store results in AnnData
    print("Storing results in AnnData...")
    # Flatten the velocity dictionary
    results = {}
    for key, value in velocity.items():
        results[key] = value

    # Add mean parameters
    results["alpha"] = jnp.mean(posterior_samples["alpha"], axis=0)
    results["beta"] = jnp.mean(posterior_samples["beta"], axis=0)
    results["gamma"] = jnp.mean(posterior_samples["gamma"], axis=0)

    # Add optional parameters if they exist in posterior_samples
    if "tau" in posterior_samples:
        results["latent_time"] = jnp.mean(posterior_samples["tau"], axis=0)

    adata_out = format_anndata_output(adata, results)

    # Print the columns in the AnnData object to see what's available
    print("Columns in adata_out.obs:", list(adata_out.obs.columns))
    print("Keys in adata_out.uns:", list(adata_out.uns.keys()) if hasattr(adata_out, 'uns') and adata_out.uns is not None else "None")

    # Add latent time directly to the AnnData object
    if "tau" in posterior_samples:
        print("Adding latent_time to AnnData object...")
        tau_mean = jnp.mean(posterior_samples["tau"], axis=0)
        # Convert to numpy array for compatibility with AnnData
        tau_mean_np = np.array(tau_mean)
        # Add to AnnData object
        adata_out.obs["latent_time"] = tau_mean_np
        print("Added latent_time column:", "latent_time" in adata_out.obs.columns)

    # 9. Visualize results
    print("Visualizing results...")
    # Use the column we just added
    sc.pl.umap(adata_out, color="latent_time", title="Latent Time")

    # Check if 'clusters' exists in the AnnData object
    if 'clusters' in adata_out.obs.columns:
        scv.pl.velocity_embedding_stream(adata_out, basis="umap", color="clusters")
    else:
        # Use a default color if 'clusters' doesn't exist
        scv.pl.velocity_embedding_stream(adata_out, basis="umap")

    # 10. Save results
    from pathlib import Path
    output_path = Path("velocity_results_svi.h5ad")
    print(f"Saving results to {output_path}...")
    adata_out.write(output_path)
    print("Done!")

if __name__ == "__main__":
    main()