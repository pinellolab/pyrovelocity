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

def main():
    # Set random seed for reproducibility
    key = create_key(0)
    
    # 1. Load and preprocess data
    adata = sc.read("path/to/your/data.h5ad")
    sc.pp.filter_genes(adata, min_counts=10)
    sc.pp.normalize_per_cell(adata)
    sc.pp.log1p(adata)
    
    # 2. Prepare data for velocity model
    data_dict = prepare_anndata(
        adata,
        spliced_layer="spliced",
        unspliced_layer="unspliced",
    )
    
    # 3. Create model configuration
    model_config = ModelConfig(
        dynamics="standard",
        likelihood="poisson",
        latent_time=True,
        include_prior=True,
    )
    
    # 4. Create inference configuration
    inference_config = InferenceConfig(
        num_warmup=500,
        num_samples=1000,
        num_chains=4,
        thinning=2,
        method="mcmc",
        mcmc_method="nuts",
        target_accept_prob=0.8,
        max_tree_depth=10,
    )
    
    # 5. Create model
    model = create_model(model_config)
    
    # 6. Run inference
    key, subkey = jax.random.split(key)
    inference_state = run_inference(
        model=model,
        data=data_dict,
        config=inference_config,
        key=subkey,
    )
    
    # 7. Check MCMC diagnostics
    diagnostics = mcmc_diagnostics(inference_state)
    print("MCMC Diagnostics:")
    print(f"Number of divergences: {diagnostics['num_divergences']}")
    print(f"Average r_hat: {diagnostics['average_r_hat']}")
    print(f"Minimum ESS: {diagnostics['min_ess']}")
    
    # 8. Analyze results
    posterior_samples = inference_state.posterior_samples
    velocity = compute_velocity(posterior_samples, data_dict)
    
    # 9. Store results in AnnData
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
    sc.pl.umap(adata_out, color="latent_time", title="Latent Time")
    sc.pl.velocity_embedding_stream(adata_out, basis="umap", color="clusters")
    
    # 11. Save results
    adata_out.write("velocity_results_mcmc.h5ad")

if __name__ == "__main__":
    main()