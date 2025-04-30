"""
End-to-end example of RNA velocity analysis using PyroVelocity PyTorch/Pyro modular implementation with MCMC.

This example demonstrates:
1. Loading and preprocessing data
2. Creating a velocity model with the modular architecture
3. Performing MCMC inference
4. Analyzing and visualizing results
5. Saving the results

Note: MCMC initialization can be challenging for complex models. If you encounter
initialization errors, consider:
1. Using SVI first to find good initial parameters, then using MCMC
2. Simplifying the model (as done here by disabling latent time)
3. Using more informative priors
4. Increasing the number of initialization attempts
"""

import torch
import pyro
import scanpy as sc
import scvelo as scv
import matplotlib.pyplot as plt
import numpy as np
from importlib.resources import files

# Import model creation and components
from pyrovelocity.models.modular.factory import create_standard_model, create_model, standard_model_config
from pyrovelocity.models.modular.model import PyroVelocityModel

# Import data loading utilities
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
    pyro.set_rng_seed(0)
    torch.manual_seed(0)

    # 1. Load test data
    print("Loading test data...")
    adata = load_test_data()
    print(f"Loaded AnnData: {adata.shape[0]} cells, {adata.shape[1]} genes")

    # 2. Prepare data for velocity model
    print("Preparing data for velocity model...")
    # Set up AnnData using the direct method
    adata = PyroVelocityModel.setup_anndata(adata)

    # 3. Create model with custom configuration for MCMC
    print("Creating model with MCMC-friendly configuration...")

    # Create a standard model configuration
    config = standard_model_config()

    # Customize the configuration for MCMC
    # - Use simpler priors
    # - Disable latent time for simpler model
    config.prior_model.name = "lognormal"
    config.prior_model.params = {"scale_alpha": 1.0, "scale_beta": 1.0, "scale_gamma": 1.0}
    config.dynamics_model.params = {"shared_time": False}  # Disable latent time

    # Create the model
    model = create_model(config)
    print(f"Created model with components:")
    print(f"  - Dynamics: {model.dynamics_model.__class__.__name__}")
    print(f"  - Prior: {model.prior_model.__class__.__name__}")
    print(f"  - Likelihood: {model.likelihood_model.__class__.__name__}")

    # 4. Train the model directly using AnnData with MCMC
    print("\nRunning MCMC inference...")
    model.train(
        adata=adata,
        method="mcmc",
        num_samples=500,  # Number of posterior samples
        num_warmup=200,   # Number of warmup steps
        num_chains=2,     # Number of MCMC chains
        use_gpu=False,
    )

    # 5. Generate posterior samples
    print("\nGenerating posterior samples...")
    posterior_samples = model.generate_posterior_samples(
        adata=adata,
        num_samples=30
    )

    # 6. Store results in AnnData
    print("Storing results in AnnData...")
    adata_out = model.store_results_in_anndata(
        adata=adata,
        posterior_samples=posterior_samples
    )

    # 7. Visualize results
    print("\nVisualizing results...")
    # Use latent time if available
    if 'velocity_model_latent_time' in adata_out.obs.columns:
        sc.pl.umap(adata_out, color="velocity_model_latent_time", title="Latent Time (MCMC)")
        plt.savefig("mcmc_velocity_latent_time.png")
        print("Latent time plot saved to mcmc_velocity_latent_time.png")

    # Check if 'clusters' exists in the AnnData object
    if 'clusters' in adata_out.obs.columns:
        scv.pl.velocity_embedding_stream(adata_out, basis="umap", color="clusters", title="RNA Velocity (MCMC)")
    else:
        # Use a default color if 'clusters' doesn't exist
        scv.pl.velocity_embedding_stream(adata_out, basis="umap", title="RNA Velocity (MCMC)")
    plt.savefig("mcmc_velocity_stream.png")
    print("Velocity stream plot saved to mcmc_velocity_stream.png")

    # 8. Save results
    from pathlib import Path
    output_path = Path("velocity_results_mcmc.h5ad")
    print(f"\nSaving results to {output_path}...")
    adata_out.write(output_path)
    print("Done!")

if __name__ == "__main__":
    main()
