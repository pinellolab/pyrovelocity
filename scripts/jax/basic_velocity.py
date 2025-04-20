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
    create_guide,
    run_inference,
    
    # Training
    create_optimizer_with_schedule,
    
    # Analysis
    compute_velocity,
    analyze_posterior,
    
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
        dynamics="standard",  # or "nonlinear" or "ode"
        likelihood="poisson",  # or "negative_binomial"
        latent_time=True,
        include_prior=True,
    )
    
    # 4. Create inference configuration
    inference_config = InferenceConfig(
        num_warmup=500,
        num_samples=1000,
        num_chains=1,
        thinning=2,
        method="svi",
        optimizer="adam",
        learning_rate=0.01,
        num_epochs=1000,
    )
    
    # 5. Create model and guide
    model = create_model(model_config)
    guide = create_guide(model, guide_type="auto_normal")
    
    # 6. Run inference
    key, subkey = jax.random.split(key)
    inference_state = run_inference(
        model=model,
        guide=guide,
        data=data_dict,
        config=inference_config,
        key=subkey,
    )
    
    # 7. Analyze results
    posterior_samples = inference_state.posterior_samples
    velocity = compute_velocity(posterior_samples, data_dict)
    
    # 8. Store results in AnnData
    results = {
        "velocity": velocity,
        "alpha": jnp.mean(posterior_samples["alpha"], axis=0),
        "beta": jnp.mean(posterior_samples["beta"], axis=0),
        "gamma": jnp.mean(posterior_samples["gamma"], axis=0),
        "switching": jnp.mean(posterior_samples["switching"], axis=0),
        "latent_time": jnp.mean(posterior_samples["latent_time"], axis=0),
    }
    
    adata_out = format_anndata_output(adata, results)
    
    # 9. Visualize results
    sc.pl.umap(adata_out, color="latent_time", title="Latent Time")
    sc.pl.velocity_embedding_stream(adata_out, basis="umap", color="clusters")
    
    # 10. Save results
    adata_out.write("velocity_results.h5ad")

if __name__ == "__main__":
    main()