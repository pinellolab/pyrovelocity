"""
End-to-end example of RNA velocity analysis using MCMC with PyroVelocity JAX/NumPyro implementation.

This example demonstrates:
1. Loading and preprocessing data
2. Creating a velocity model
3. Performing MCMC inference
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
    
    # Analysis
    compute_velocity,
    analyze_posterior,
    mcmc_diagnostics,
    
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
        dynamics="standard",
        likelihood="poisson",
        latent_time=True,
        include_prior=True,
    )
    
    # 4. Create inference configuration
    print("Creating inference configuration...")
    inference_config = InferenceConfig(
        num_warmup=50,   # Reduced for example
        num_samples=100,  # Reduced for example
        num_chains=2,     # Reduced for example
        method="mcmc",
        # Note: mcmc_method, target_accept_prob, and max_tree_depth are not part of InferenceConfig
        # They would need to be passed separately to the MCMC kernel
    )
    
    # 5. Create model
    print("Creating model...")
    model = create_model(model_config)
    
    # 6. Run inference
    print("Running MCMC inference...")
    key, subkey = jax.random.split(key)
    
    # Extract the data from data_dict and map to the expected parameter names
    u_obs = data_dict["X_unspliced"]
    s_obs = data_dict["X_spliced"]
    u_log_library = jnp.log(data_dict["u_lib_size"])
    s_log_library = jnp.log(data_dict["s_lib_size"])
    
    # The run_inference function expects (model, args, kwargs, config, key)
    inference_state = run_inference(
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
    
    # 7. Check MCMC diagnostics
    print("Checking MCMC diagnostics...")
    diagnostics = mcmc_diagnostics(inference_state)
    print("MCMC Diagnostics:")
    print(f"Number of divergences: {diagnostics['num_divergences']}")
    print(f"Average r_hat: {diagnostics['average_r_hat']}")
    print(f"Minimum ESS: {diagnostics['min_ess']}")
    
    # 8. Analyze results
    print("Analyzing results...")
    posterior_samples = inference_state.posterior_samples
    velocity = compute_velocity(posterior_samples, data_dict)
    
    # 9. Store results in AnnData
    print("Storing results in AnnData...")
    results = {
        "velocity": velocity,
        "alpha": jnp.mean(posterior_samples["alpha"], axis=0),
        "beta": jnp.mean(posterior_samples["beta"], axis=0),
        "gamma": jnp.mean(posterior_samples["gamma"], axis=0),
        "switching": jnp.mean(posterior_samples["switching"], axis=0),
        "latent_time": jnp.mean(posterior_samples["latent_time"], axis=0),
    }
    
    adata_out = format_anndata_output(adata, results)
    
    # 10. Visualize results
    print("Visualizing results...")
    sc.pl.umap(adata_out, color="latent_time", title="Latent Time")
    sc.pl.velocity_embedding_stream(adata_out, basis="umap", color="clusters")
    
    # 11. Save results
    output_path = "velocity_results_mcmc.h5ad"
    print(f"Saving results to {output_path}...")
    adata_out.write(output_path)
    print("Done!")

if __name__ == "__main__":
    main()