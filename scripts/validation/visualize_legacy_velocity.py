#!/usr/bin/env python
"""
Visualize Legacy Velocity Script for PyroVelocity.

This script visualizes the velocity field from the legacy model by training the model,
generating posterior samples, computing statistics, and plotting the velocity field.

Example usage:
    python visualize_legacy_velocity.py --max-epochs 5 --num-samples 3
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import pyro
import matplotlib.pyplot as plt
from importlib.resources import files

# Add the src directory to the path to import test fixtures
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from pyrovelocity.io.serialization import load_anndata_from_json
from pyrovelocity.models._velocity import PyroVelocity
import scvelo as scv


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Visualize Legacy Velocity for PyroVelocity"
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=5,
        help="Maximum number of epochs for training",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=3,
        help="Number of posterior samples to generate",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="legacy_visualization_results",
        help="Directory to save visualization results",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    return parser.parse_args()


def load_preprocessed_pancreas_data():
    """Load the preprocessed pancreas data fixture."""
    print("Loading preprocessed pancreas data fixture...")

    # Fixture hash for data validation
    FIXTURE_HASH = "95c80131694f2c6449a48a56513ef79cdc56eae75204ec69abde0d81a18722ae"

    try:
        fixture_file_path = (
            files("pyrovelocity.tests.data") / "preprocessed_pancreas_50_7.json"
        )
        adata = load_anndata_from_json(
            filename=str(fixture_file_path),
            expected_hash=FIXTURE_HASH,
        )
        print(f"Loaded preprocessed pancreas data: {adata.shape[0]} cells, {adata.shape[1]} genes")
        return adata
    except Exception as e:
        print(f"Error loading preprocessed pancreas data: {e}")
        raise


def train_legacy_model(adata, max_epochs, num_samples):
    """Train the legacy model and return results."""
    print("Training legacy model...")

    # Set up AnnData for legacy model
    PyroVelocity.setup_anndata(adata)

    # Create legacy model
    model = PyroVelocity(adata)

    # Train model
    training_start_time = time.time()
    model.train(max_epochs=max_epochs, check_val_every_n_epoch=None)
    training_end_time = time.time()
    training_time = training_end_time - training_start_time

    # Generate posterior samples
    inference_start_time = time.time()
    try:
        posterior_samples = model.generate_posterior_samples(num_samples=num_samples)
    except AttributeError as e:
        if "'NoneType' object has no attribute 'uns'" in str(e):
            # This is a known issue with the legacy model
            # We need to pass the adata explicitly
            posterior_samples = model.generate_posterior_samples(adata=model.adata, num_samples=num_samples)
        else:
            raise

    # Compute statistics from posterior samples (this computes velocity and stores it in adata)
    model.compute_statistics_from_posterior_samples(
        adata=adata,
        posterior_samples=posterior_samples,
        vector_field_basis="umap",
        ncpus_use=1,
        random_seed=99
    )
    inference_end_time = time.time()
    inference_time = inference_end_time - inference_start_time

    # Return results
    return {
        "model": model,
        "adata": adata,
        "posterior_samples": posterior_samples,
        "performance": {
            "training_time": training_time,
            "inference_time": inference_time,
        }
    }


def visualize_velocity_field(results, output_dir):
    """Visualize the velocity field from the legacy model."""
    print("Visualizing velocity field...")

    # Create figure directory
    figure_dir = os.path.join(output_dir, "figures")
    os.makedirs(figure_dir, exist_ok=True)

    # Extract results
    adata = results["adata"]

    # Plot velocity field
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)

    # Instead of using the plot_vector_field_uncertainty function, let's use a simpler approach
    # to visualize the velocity field

    # Plot the velocity field using scvelo's velocity_embedding_stream function
    scv.pl.velocity_embedding_stream(
        adata,
        basis="umap",
        vkey="velocity_pyro",
        title="Legacy Model Velocity Field",
        ax=ax,
        save=False,
        show=False,
        color=None,
        density=1.0,
        arrow_size=5,
        linewidth=1.0,
    )

    # Save the figure
    fig.savefig(
        os.path.join(figure_dir, "velocity_field.png"),
        dpi=300,
        bbox_inches="tight"
    )

    print(f"Velocity field visualization saved to {figure_dir}")


def main():
    """Run the legacy model visualization and save results."""
    # Parse command line arguments
    args = parse_args()

    # Set random seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    pyro.set_rng_seed(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Created output directory: {args.output_dir}")

    # Load preprocessed pancreas data
    try:
        adata = load_preprocessed_pancreas_data()
    except Exception as e:
        print(f"Failed to load data: {e}")
        return

    # Train legacy model
    try:
        legacy_results = train_legacy_model(adata, args.max_epochs, args.num_samples)
        print("Legacy model training successful")
    except Exception as e:
        print(f"Error training legacy model: {e}")
        import traceback
        traceback.print_exc()
        return

    # Visualize velocity field
    try:
        visualize_velocity_field(legacy_results, args.output_dir)
        print(f"Results saved to {args.output_dir}")
    except Exception as e:
        print(f"Error visualizing velocity field: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()
