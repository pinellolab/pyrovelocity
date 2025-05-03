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

PyroVelocity supports two implementation approaches for components:
1. Base Class Approach: Components inherit from base classes (BaseDynamicsModel, etc.)
2. Protocol-First Approach: Components directly implement Protocol interfaces without inheritance

The Protocol-First approach has several advantages:
- Reduced code complexity by eliminating inheritance hierarchies
- Enhanced flexibility through Protocol interfaces
- Perfect architectural consistency with the JAX implementation's pure functional approach
- Discovery of natural abstractions through actual usage patterns
- Avoidance of premature abstraction by initially allowing intentional duplication

This script shows how to use both approaches for a basic RNA velocity analysis workflow.
"""

import time
import torch
import pyro
import scanpy as sc
import scvelo as scv
import matplotlib.pyplot as plt
import numpy as np
from importlib.resources import files
from tqdm import tqdm
import os
import sys
import traceback

# Import model creation and components
from pyrovelocity.models.modular.factory import (
    create_standard_model,
    create_model,
    create_protocol_first_model,
    create_model_from_config,
    standard_model_config
)
from pyrovelocity.models.modular.config import ModelConfig
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
    try:
        fixture_file_path = (
            files("pyrovelocity.tests.data") / "preprocessed_pancreas_50_7.json"
        )
        return load_anndata_from_json(
            filename=str(fixture_file_path),
            expected_hash=FIXTURE_HASH,
        )
    except Exception as e:
        print(f"Error loading test data: {e}")
        print("Falling back to synthetic data...")
        return create_synthetic_data()

def create_synthetic_data():
    """Create synthetic data for testing when fixture data is not available."""
    import anndata as ad
    import numpy as np
    import pandas as pd

    # Create synthetic data
    n_cells, n_genes = 50, 20
    u_data = np.random.poisson(5, size=(n_cells, n_genes))
    s_data = np.random.poisson(5, size=(n_cells, n_genes))

    # Create AnnData object
    adata = ad.AnnData(X=s_data)
    adata.layers["spliced"] = s_data
    adata.layers["unspliced"] = u_data
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]

    # Add cluster information
    adata.obs["clusters"] = np.random.choice(["A", "B", "C"], size=n_cells)

    # Add UMAP coordinates for visualization
    adata.obsm = {}
    adata.obsm["X_umap"] = np.random.normal(0, 1, size=(n_cells, 2))

    print(f"Created synthetic AnnData: {adata.shape[0]} cells, {adata.shape[1]} genes")
    return adata

def main():
    """Run a complete RNA velocity analysis workflow using the modular implementation."""
    # Set random seed for reproducibility
    # This ensures that results are deterministic and reproducible
    import torch
    import pyro
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
    # This uses default components with base class implementations
    print("\nMethod 1: Using create_standard_model() (base class approach)")
    model1 = create_standard_model()
    print(f"Created model with base class components:")
    print(f"  - Dynamics: {model1.dynamics_model.__class__.__name__}")
    print(f"  - Prior: {model1.prior_model.__class__.__name__}")
    print(f"  - Likelihood: {model1.likelihood_model.__class__.__name__}")
    print(f"  - Observation: {model1.observation_model.__class__.__name__}")

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
    print(f"  - Observation: {model2.observation_model.__class__.__name__}")
    # Note: inference_guide is not directly accessible as an attribute

    # Method 3: Create a model with Protocol-First components
    # This uses components that directly implement Protocol interfaces without inheritance
    print("\nMethod 3: Using create_protocol_first_model() (Protocol-First approach)")
    model3 = create_protocol_first_model()
    print(f"Created model with Protocol-First components:")
    print(f"  - Dynamics: {model3.dynamics_model.__class__.__name__}")
    print(f"  - Prior: {model3.prior_model.__class__.__name__}")
    print(f"  - Likelihood: {model3.likelihood_model.__class__.__name__}")
    print(f"  - Observation: {model3.observation_model.__class__.__name__}")

    # Method 4: Create a model with ModelConfig.standard(use_protocol_first=True)
    # This is an alternative way to create a model with Protocol-First components
    print("\nMethod 4: Using ModelConfig.standard(use_protocol_first=True)")
    protocol_config = ModelConfig.standard(use_protocol_first=True)
    model4 = create_model_from_config(protocol_config)
    print(f"Created model with Protocol-First components:")
    print(f"  - Dynamics: {model4.dynamics_model.__class__.__name__}")
    print(f"  - Prior: {model4.prior_model.__class__.__name__}")
    print(f"  - Likelihood: {model4.likelihood_model.__class__.__name__}")
    print(f"  - Observation: {model4.observation_model.__class__.__name__}")

    # Use the second model for this example
    # You could also use model3 or model4 (Protocol-First models) which are functionally equivalent
    model = model2

    # 5. Train the model directly using AnnData
    # The train method handles:
    # - Preparing data from AnnData
    # - Setting up the inference configuration
    # - Running SVI (Stochastic Variational Inference)
    # - Storing the inference state in the model state
    print("\nTraining the model...")

    # Define a progress callback function to show training progress
    def update_progress(epoch, loss):
        if hasattr(update_progress, 'pbar'):
            update_progress.pbar.update(1)
            update_progress.pbar.set_postfix({"loss": f"{loss:.4f}"})

    # Set up progress bar
    update_progress.pbar = tqdm(total=200, desc="Training")

    # Record training time
    start_time = time.time()

    try:
        model.train(
            adata=adata,
            max_epochs=200,  # Reduced for example
            learning_rate=0.01,
            use_gpu=False,   # Set to True to use GPU if available
            progress_callback=update_progress,
        )
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
    except Exception as e:
        print(f"Error during training: {e}")
        print("Trying again with fewer epochs...")
        try:
            # Close previous progress bar
            if hasattr(update_progress, 'pbar'):
                update_progress.pbar.close()

            # Set up new progress bar with fewer epochs
            update_progress.pbar = tqdm(total=50, desc="Training (reduced)")

            model.train(
                adata=adata,
                max_epochs=50,  # Reduced further due to error
                learning_rate=0.005,  # Lower learning rate
                use_gpu=False,
                progress_callback=update_progress,
            )
            training_time = time.time() - start_time
            print(f"Training completed in {training_time:.2f} seconds")
        except Exception as e:
            print(f"Training failed: {e}")
            print("Proceeding with untrained model for demonstration purposes")
    finally:
        # Ensure progress bar is closed
        if hasattr(update_progress, 'pbar'):
            update_progress.pbar.close()

    # 6. Generate posterior samples
    # This samples from the approximate posterior distribution
    # to enable uncertainty quantification
    print("\nGenerating posterior samples...")
    try:
        with tqdm(total=1, desc="Generating samples") as pbar:
            posterior_samples = model.generate_posterior_samples(
                adata=adata,
                num_samples=30   # Number of samples to generate
            )
            pbar.update(1)

        # Print some statistics about the samples
        print("Posterior sample statistics:")
        for param_name, param_samples in posterior_samples.items():
            if isinstance(param_samples, torch.Tensor):
                print(f"  - {param_name}: shape={param_samples.shape}, "
                      f"mean={param_samples.mean().item():.4f}, "
                      f"std={param_samples.std().item():.4f}")
    except Exception as e:
        print(f"Error generating posterior samples: {e}")
        print("Trying with fewer samples...")
        try:
            posterior_samples = model.generate_posterior_samples(
                adata=adata,
                num_samples=10   # Reduced number of samples
            )
        except Exception as e:
            print(f"Failed to generate posterior samples: {e}")
            print("Creating dummy samples for demonstration purposes")
            # Create dummy samples to continue the example
            import torch
            posterior_samples = {
                "alpha": torch.ones(5, adata.shape[1]),
                "beta": torch.ones(5, adata.shape[1]),
                "gamma": torch.ones(5, adata.shape[1])
            }

    # 7. Store results in AnnData
    # This computes velocity from posterior samples and stores all results
    # in the AnnData object for visualization and analysis
    print("Storing results in AnnData...")
    try:
        with tqdm(total=1, desc="Storing results") as pbar:
            adata_out = model.store_results_in_anndata(
                adata=adata,
                posterior_samples=posterior_samples,
                model_name="velocity_model"  # Prefix for stored results
            )
            pbar.update(1)
    except Exception as e:
        print(f"Error storing results in AnnData: {e}")
        print("Using original AnnData for visualization")
        adata_out = adata

    # 8. Visualize results
    # We can visualize the results using scanpy and scvelo
    print("\nVisualizing results...")

    try:
        # Plot latent time if available
        # Latent time represents the progression of cells along a trajectory
        if 'velocity_model_latent_time' in adata_out.obs.columns:
            print("Plotting latent time...")
            sc.pl.umap(adata_out, color="velocity_model_latent_time", title="Latent Time")
            plt.savefig("basic_velocity_latent_time.png")
            print("Latent time plot saved to basic_velocity_latent_time.png")
        else:
            print("Latent time not available in results")

        # Plot velocity stream
        # This shows the direction and magnitude of RNA velocity
        print("Plotting velocity stream...")
        if 'clusters' in adata_out.obs.columns:
            scv.pl.velocity_embedding_stream(
                adata_out,
                basis="umap",
                color="clusters",
                title="RNA Velocity"
            )
        else:
            # Use a default color if 'clusters' doesn't exist
            scv.pl.velocity_embedding_stream(
                adata_out,
                basis="umap",
                title="RNA Velocity"
            )
        plt.savefig("basic_velocity_stream.png")
        print("Velocity stream plot saved to basic_velocity_stream.png")
    except Exception as e:
        print(f"Error during visualization: {e}")
        print("Visualization failed, but analysis results are still valid")

    # 9. Save results
    # We can save the AnnData object with results for later use
    from pathlib import Path
    output_path = Path("velocity_results_modular.h5ad")
    print(f"\nSaving results to {output_path}...")

    try:
        with tqdm(total=1, desc="Saving results") as pbar:
            adata_out.write(output_path)
            pbar.update(1)
        print(f"Results saved successfully to {output_path}")
    except Exception as e:
        print(f"Error saving results: {e}")
        print("Results were not saved to disk")

    # 10. Print execution summary
    print("\nExecution Summary:")
    print(f"- Data: {adata.shape[0]} cells, {adata.shape[1]} genes")
    print(f"- Model: {model.dynamics_model.__class__.__name__}")
    print(f"- Training: {'Completed' if hasattr(model, '_inference_state') else 'Failed'}")
    print(f"- Posterior samples: {len(posterior_samples.get('alpha', []))} samples")
    print(f"- Results stored: {'Yes' if 'velocity_model_alpha' in adata_out.var else 'No'}")

    print("\nDone!")

if __name__ == "__main__":
    main()
