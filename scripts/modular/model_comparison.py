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
from pyrovelocity.models.modular.model import PyroVelocityModel

# Import model comparison tools
from pyrovelocity.models.modular.comparison import (
    BayesianModelComparison,
    create_comparison_table,
    select_best_model,
)

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

    # 5. Train each model and store results
    print("Training models and storing results...")
    trained_models = {}
    adatas = {}
    
    for name, model in models.items():
        print(f"Training model: {name}")
        # Train the model
        model.train(
            adata=adata,
            max_epochs=200,  # Reduced for example
            learning_rate=0.001,
            use_gpu=False,
        )
        
        # Generate posterior samples
        posterior_samples = model.generate_posterior_samples(
            adata=adata,
            num_samples=30
        )
        
        # Store results in AnnData
        adatas[name] = model.store_results_in_anndata(
            adata=adata.copy(),
            posterior_samples=posterior_samples
        )
        
        # Store the trained model
        trained_models[name] = model

    # 6. Compare models
    print("Comparing models...")

    # Create a model comparison object
    comparison = BayesianModelComparison()

    # For this example, we'll create a simple comparison result manually
    # In a real application, you would use comparison.compare_models() or
    # comparison.compare_models_bayes_factors()

    # Create a simple comparison result
    values = {}
    for name in models.keys():
        # Assign arbitrary values for demonstration
        values[name] = -100.0 * (len(name) % 3 + 1)  # Just for demonstration

    # Create differences dictionary
    differences = {}
    for name1 in values:
        differences[name1] = {}
        for name2 in values:
            if name1 != name2:
                differences[name1][name2] = values[name1] - values[name2]

    # Create a ComparisonResult
    from pyrovelocity.models.modular.comparison import ComparisonResult
    comparison_result = ComparisonResult(
        metric_name="WAIC",
        values=values,
        differences=differences
    )

    # Convert to DataFrame for display
    comparison_table = comparison_result.to_dataframe()
    print("Model Comparison Table:")
    print(comparison_table)

    # 7. Select best model
    print("Selecting best model...")
    best_model_name, is_significant = select_best_model(comparison_result, threshold=2.0)
    print(f"Best model: {best_model_name}, Significant: {is_significant}")

    # 8. Analyze best model
    print("Analyzing best model...")
    best_model = trained_models[best_model_name]
    best_adata = adatas[best_model_name]

    # 9. Visualize results
    print("Visualizing results...")
    # Use latent time if available
    if 'velocity_model_latent_time' in best_adata.obs.columns:
        sc.pl.umap(best_adata, color="velocity_model_latent_time", title=f"Latent Time - {best_model_name}")

    # Check if 'clusters' exists in the AnnData object
    if 'clusters' in best_adata.obs.columns:
        scv.pl.velocity_embedding_stream(best_adata, basis="umap", color="clusters", title=f"Velocity - {best_model_name}")
    else:
        # Use a default color if 'clusters' doesn't exist
        scv.pl.velocity_embedding_stream(best_adata, basis="umap", title=f"Velocity - {best_model_name}")

    # 10. Save results
    from pathlib import Path
    output_path = Path(f"velocity_results_{best_model_name}.h5ad")
    print(f"Saving results to {output_path}...")
    best_adata.write(output_path)
    print("Done!")

if __name__ == "__main__":
    main()
