"""
Basic end-to-end example of RNA velocity analysis using PyroVelocity PyTorch/Pyro modular implementation.

This example demonstrates:
1. Loading and preprocessing data
2. Creating a velocity model with the modular architecture
3. Performing SVI inference
4. Analyzing and visualizing results
5. Saving the results

The modular implementation of PyroVelocity provides a component-based architecture
where each aspect of the model (dynamics, priors, likelihoods, observations, guides)
is implemented as a separate component. This enables:
- Flexibility: Components can be swapped out independently
- Extensibility: New components can be added without modifying existing code
- Testability: Components can be tested in isolation
- Reusability: Components can be reused across different models

This script shows how to use the modular implementation for a basic RNA velocity
analysis workflow, from data loading to visualization.
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
from pyrovelocity.models.modular.registry import (
    DynamicsModelRegistry,
    PriorModelRegistry,
    LikelihoodModelRegistry,
    ObservationModelRegistry,
    InferenceGuideRegistry,
)
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
    """Run a complete RNA velocity analysis workflow using the modular implementation."""
    # Set random seed for reproducibility
    # This ensures that results are deterministic and reproducible
    pyro.set_rng_seed(0)
    torch.manual_seed(0)

    # 1. Load test data
    # We use a small test dataset for demonstration purposes
    print("Loading test data...")
    adata = load_test_data()
    print(f"Loaded AnnData: {adata.shape[0]} cells, {adata.shape[1]} genes")

    # 2. Prepare data for velocity model
    # The setup_anndata method adds necessary information to the AnnData object:
    # - Computes library sizes for normalization
    # - Adds library size statistics
    # - Adds indices for batch processing
    print("Preparing data for velocity model...")
    adata = PyroVelocityModel.setup_anndata(adata)

    # 3. List available components in each registry
    # The registry system allows for easy discovery of available components
    print("\nAvailable components in registries:")
    print(f"Dynamics Models: {DynamicsModelRegistry.list_available()}")
    print(f"Prior Models: {PriorModelRegistry.list_available()}")
    print(f"Likelihood Models: {LikelihoodModelRegistry.list_available()}")
    print(f"Observation Models: {ObservationModelRegistry.list_available()}")
    print(f"Inference Guides: {InferenceGuideRegistry.list_available()}")

    # 4. Create model with custom configuration
    # There are multiple ways to create a model, from simple to complex
    print("\nCreating model with custom configuration...")

    # Method 1: Create a standard model directly (simplest approach)
    # This uses default components for all aspects of the model
    print("\nMethod 1: Using create_standard_model() (simplest approach)")
    model1 = create_standard_model()
    print(f"Created model with default components")

    # Method 2: Create a model with a custom configuration
    # This allows for more control over the model components
    print("\nMethod 2: Using create_model() with standard_model_config()")
    config = standard_model_config()

    # Customize the configuration using available components from registries
    # Each component can be configured independently
    config.dynamics_model.name = "standard"      # Use standard dynamics model
    config.prior_model.name = "lognormal"        # Use lognormal prior
    config.likelihood_model.name = "poisson"     # Use Poisson likelihood
    config.inference_guide.name = "auto"         # Use auto guide
    config.inference_guide.params = {"guide_type": "AutoNormal", "init_scale": 0.1}

    # Create the model with custom configuration
    # The factory system creates the model with the specified components
    model2 = create_model(config)
    print(f"Created model with components:")
    print(f"  - Dynamics: {model2.dynamics_model.__class__.__name__}")
    print(f"  - Prior: {model2.prior_model.__class__.__name__}")
    print(f"  - Likelihood: {model2.likelihood_model.__class__.__name__}")
    # Note: inference_guide is not directly accessible as an attribute

    # Use the second model for this example
    model = model2

    # 5. Train the model directly using AnnData
    # The train method handles:
    # - Preparing data from AnnData
    # - Setting up the inference configuration
    # - Running SVI (Stochastic Variational Inference)
    # - Storing the inference state in the model state
    print("\nTraining the model...")
    model.train(
        adata=adata,
        max_epochs=200,  # Reduced for example
        learning_rate=0.01,
        use_gpu=False,   # Set to True to use GPU if available
    )

    # 6. Generate posterior samples
    # This samples from the approximate posterior distribution
    # to enable uncertainty quantification
    print("\nGenerating posterior samples...")
    posterior_samples = model.generate_posterior_samples(
        adata=adata,
        num_samples=30   # Number of samples to generate
    )

    # 7. Store results in AnnData
    # This computes velocity from posterior samples and stores all results
    # in the AnnData object for visualization and analysis
    print("Storing results in AnnData...")
    adata_out = model.store_results_in_anndata(
        adata=adata,
        posterior_samples=posterior_samples,
        model_name="velocity_model"  # Prefix for stored results
    )

    # 8. Visualize results
    # We can visualize the results using scanpy and scvelo
    print("\nVisualizing results...")

    # Plot latent time if available
    # Latent time represents the progression of cells along a trajectory
    if 'velocity_model_latent_time' in adata_out.obs.columns:
        sc.pl.umap(adata_out, color="velocity_model_latent_time", title="Latent Time")
        plt.savefig("basic_velocity_latent_time.png")
        print("Latent time plot saved to basic_velocity_latent_time.png")

    # Plot velocity stream
    # This shows the direction and magnitude of RNA velocity
    if 'clusters' in adata_out.obs.columns:
        scv.pl.velocity_embedding_stream(adata_out, basis="umap", color="clusters", title="RNA Velocity")
    else:
        # Use a default color if 'clusters' doesn't exist
        scv.pl.velocity_embedding_stream(adata_out, basis="umap", title="RNA Velocity")
    plt.savefig("basic_velocity_stream.png")
    print("Velocity stream plot saved to basic_velocity_stream.png")

    # 9. Save results
    # We can save the AnnData object with results for later use
    from pathlib import Path
    output_path = Path("velocity_results_modular.h5ad")
    print(f"\nSaving results to {output_path}...")
    adata_out.write(output_path)
    print("Done!")

if __name__ == "__main__":
    main()
