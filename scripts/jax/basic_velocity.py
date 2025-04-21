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
import numpyro
import matplotlib.pyplot as plt
import scanpy as sc
import anndata
from importlib.resources import files

from pyrovelocity.models.jax import (
    # Core components
    create_key,
    ModelConfig,
    InferenceConfig,
    
    # Model creation
    create_model,
    
    # Data processing
    prepare_anndata,
    
    # Inference
    run_inference,
    
    # Training
    create_optimizer_with_schedule,
    
    # Analysis
    compute_velocity,
    analyze_posterior,
    
    # Visualization
    format_anndata_output,
)

from pyrovelocity.io.serialization import load_anndata_from_json

# Fixture hash for data validation
FIXTURE_HASH = "95c80131694f2c6449a48a56513ef79cdc56eae75204ec69abde0d81a18722ae"

def load_test_data():
    """Load test data from the fixtures."""
    fixture_file_path = (
        files("pyrovelocity.tests.data") / "preprocessed_pancreas_50_7.json"
    )
    return load_anndata_from_json(
        filename=fixture_file_path,
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
    model_config = ModelConfig(
        dynamics="standard",  # or "nonlinear" or "ode"
        likelihood="poisson",  # or "negative_binomial"
        latent_time=True,
        include_prior=True,
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
    # Use the default dynamics_fn (standard_dynamics_model) by not providing a second argument
    velocity = compute_velocity(posterior_samples)
    
    # 8. Store results in AnnData
    print("Storing results in AnnData...")
    results = {
        "velocity": velocity,
        "alpha": jnp.mean(posterior_samples["alpha"], axis=0),
        "beta": jnp.mean(posterior_samples["beta"], axis=0),
        "gamma": jnp.mean(posterior_samples["gamma"], axis=0),
    }
    
    # Add optional parameters if they exist in posterior_samples
    if "tau" in posterior_samples:
        results["latent_time"] = jnp.mean(posterior_samples["tau"], axis=0)
    
    adata_out = format_anndata_output(adata, results)
    
    # 9. Visualize results
    print("Visualizing results...")
    sc.pl.umap(adata_out, color="latent_time", title="Latent Time")
    sc.pl.velocity_embedding_stream(adata_out, basis="umap", color="clusters")
    
    # 10. Save results
    output_path = "velocity_results_svi.h5ad"
    print(f"Saving results to {output_path}...")
    adata_out.write(output_path)
    print("Done!")

if __name__ == "__main__":
    main()