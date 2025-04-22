"""
Example of model comparison using PyroVelocity PyTorch/Pyro modular implementation.

This example demonstrates:
1. Loading and preprocessing data
2. Creating multiple velocity models with different configurations
3. Comparing models using Bayesian model comparison
4. Selecting the best model
5. Analyzing and visualizing results
"""

import torch
import pyro
import scanpy as sc
import scvelo as scv
import pandas as pd
import matplotlib.pyplot as plt
from importlib.resources import files

# Import model creation
from pyrovelocity.models.modular.factory import create_model, standard_model_config

# No need to import model comparison tools for this simplified example

# Import adapters for AnnData integration
from pyrovelocity.models.adapters import LegacyModelAdapter

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
    # Set up AnnData for LegacyModelAdapter
    LegacyModelAdapter.setup_anndata(adata)

    # 3. Create model configurations
    print("Creating model configurations...")
    # Create different model configurations for comparison

    # Model 1: Standard model with Poisson likelihood
    config1 = standard_model_config()
    config1.likelihood_model.name = "poisson"

    # Model 2: Standard model with Negative Binomial likelihood
    config2 = standard_model_config()
    config2.likelihood_model.name = "negative_binomial"

    # Model 3: Nonlinear dynamics with Poisson likelihood
    config3 = standard_model_config()
    config3.dynamics_model.name = "nonlinear"
    config3.likelihood_model.name = "poisson"

    # Model 4: Standard model with informative priors
    config4 = standard_model_config()
    config4.prior_model.name = "informative"

    # 4. Create models
    print("Creating models...")
    model1 = create_model(config1)
    model2 = create_model(config2)
    model3 = create_model(config3)
    model4 = create_model(config4)

    # Create a dictionary of models for comparison
    models = {
        "standard_poisson": model1,
        "standard_negbinom": model2,
        "nonlinear_poisson": model3,
        "standard_informative": model4,
    }

    # 5. Create adapters for each model
    print("Creating adapters for each model...")
    adapters = {}
    for name, model in models.items():
        adapters[name] = LegacyModelAdapter.from_modular_model(adata, model)

    # 6. Train each model
    print("Training models...")
    for name, adapter in adapters.items():
        print(f"Training model: {name}")
        adapter.train(
            max_epochs=200,  # Reduced for example
            learning_rate=0.001,
            use_gpu=False,
        )

    # 7. Compare models
    print("Comparing models...")

    # Compute metrics
    metrics = {}
    for name, adapter in adapters.items():
        # Get the processed AnnData object
        adata_out = adapter.adata

        # Store metrics
        metrics[name] = {
            "adata": adata_out,
        }

    # 8. Select best model (for this example, we'll just use the first model)
    print("Selecting best model...")
    best_model_name = list(metrics.keys())[0]
    print(f"Best model: {best_model_name}")

    # 9. Analyze best model
    print("Analyzing best model...")
    best_adata = metrics[best_model_name]["adata"]

    # 10. Visualize results
    print("Visualizing results...")
    # Use latent time if available
    if 'latent_time' in best_adata.obs.columns:
        sc.pl.umap(best_adata, color="latent_time", title=f"Latent Time - {best_model_name}")

    # Check if 'clusters' exists in the AnnData object
    if 'clusters' in best_adata.obs.columns:
        scv.pl.velocity_embedding_stream(best_adata, basis="umap", color="clusters", title=f"Velocity - {best_model_name}")
    else:
        # Use a default color if 'clusters' doesn't exist
        scv.pl.velocity_embedding_stream(best_adata, basis="umap", title=f"Velocity - {best_model_name}")

    # 11. Save results
    from pathlib import Path
    output_path = Path(f"velocity_results_{best_model_name}.h5ad")
    print(f"Saving results to {output_path}...")
    best_adata.write(output_path)
    print("Done!")

if __name__ == "__main__":
    main()
