"""
Basic end-to-end example of RNA velocity analysis using PyroVelocity PyTorch/Pyro modular implementation.

This example demonstrates:
1. Loading and preprocessing data
2. Creating a velocity model
3. Performing SVI inference
4. Analyzing and visualizing results
"""

import torch
import pyro
import scanpy as sc
import scvelo as scv
from importlib.resources import files

# Import model creation
from pyrovelocity.models.modular.factory import create_standard_model

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

    # 3. Create model
    print("Creating model...")
    # Create the standard model directly
    model = create_standard_model()

    # 5. Create a LegacyModelAdapter to use the legacy API for training
    print("Creating LegacyModelAdapter...")
    adapter = LegacyModelAdapter.from_modular_model(adata, model)

    # 6. Train the model
    print("Training the model...")
    adapter.train(
        max_epochs=500,  # More epochs to ensure convergence
        learning_rate=0.001,  # Lower learning rate for more stable training
        use_gpu=False,
    )

    # 7. Generate posterior samples
    print("Generating posterior samples...")
    posterior_samples = adapter.generate_posterior_samples()

    # 8. Compute velocity
    print("Computing velocity...")
    # The velocity is computed internally by the adapter
    adata_out = adapter.adata

    # 9. Visualize results
    print("Visualizing results...")
    # Use latent time if available
    if 'latent_time' in adata_out.obs.columns:
        sc.pl.umap(adata_out, color="latent_time", title="Latent Time")

    # Check if 'clusters' exists in the AnnData object
    if 'clusters' in adata_out.obs.columns:
        scv.pl.velocity_embedding_stream(adata_out, basis="umap", color="clusters")
    else:
        # Use a default color if 'clusters' doesn't exist
        scv.pl.velocity_embedding_stream(adata_out, basis="umap")

    # 10. Save results
    from pathlib import Path
    output_path = Path("velocity_results_modular.h5ad")
    print(f"Saving results to {output_path}...")
    adata_out.write(output_path)
    print("Done!")

if __name__ == "__main__":
    main()
