#!/usr/bin/env python
"""
Validate Legacy Model Script for PyroVelocity.

This script validates the legacy model's workflow by training the model,
generating posterior samples, computing statistics, and extracting velocity
from the AnnData object.

Example usage:
    python validate_legacy_model.py --max-epochs 5 --num-samples 3
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


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Validate Legacy Model for PyroVelocity"
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
        default="legacy_validation_results",
        help="Directory to save validation results",
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

    # Extract velocity from adata.layers["velocity_pyro"]
    velocity = adata.layers["velocity_pyro"] if "velocity_pyro" in adata.layers else None
    
    # Extract uncertainty (using the FDR values as a proxy for uncertainty)
    # In the legacy model, uncertainty is computed during vector_field_uncertainty
    # and stored in the posterior_samples dictionary as 'fdri'
    uncertainty = posterior_samples.get('fdri', None)
    
    if uncertainty is None:
        # If fdri is not available, we can use a simple standard deviation across samples
        # as a proxy for uncertainty
        if ('u_scale' in posterior_samples) and ('s_scale' in posterior_samples):
            scale = posterior_samples["u_scale"] / posterior_samples["s_scale"]
        elif ('u_scale' in posterior_samples) and not ('s_scale' in posterior_samples):
            scale = posterior_samples["u_scale"]
        else:
            scale = 1
            
        velocity_samples = (
            posterior_samples["beta"] * posterior_samples["ut"] / scale
            - posterior_samples["gamma"] * posterior_samples["st"]
        )
        uncertainty = np.std(velocity_samples, axis=0)

    # Return results
    return {
        "model": model,
        "posterior_samples": posterior_samples,
        "velocity": velocity,
        "uncertainty": uncertainty,
        "performance": {
            "training_time": training_time,
            "inference_time": inference_time,
        }
    }


def generate_summary_report(legacy_results, args, output_dir):
    """Generate a summary report of the validation results."""
    report_path = os.path.join(output_dir, "legacy_model_summary.txt")

    with open(report_path, "w") as f:
        f.write("PyroVelocity Legacy Model Validation Summary\n")
        f.write("=========================================\n\n")

        # Write validation parameters
        f.write("Validation Parameters\n")
        f.write("--------------------\n")
        f.write(f"Max epochs: {args.max_epochs}\n")
        f.write(f"Number of posterior samples: {args.num_samples}\n")
        f.write(f"Random seed: {args.seed}\n")
        f.write("\n")

        # Write model information
        f.write("Legacy Model Information\n")
        f.write("----------------------\n")
        f.write(f"Training time: {legacy_results['performance']['training_time']:.2f} seconds\n")
        f.write(f"Inference time: {legacy_results['performance']['inference_time']:.2f} seconds\n")
        
        if legacy_results['velocity'] is not None:
            f.write(f"Velocity shape: {legacy_results['velocity'].shape}\n")
        else:
            f.write("Velocity: None\n")
            
        if legacy_results['uncertainty'] is not None:
            f.write(f"Uncertainty shape: {legacy_results['uncertainty'].shape}\n")
        else:
            f.write("Uncertainty: None\n")
        f.write("\n")

        # Write parameter information
        f.write("Parameter Information\n")
        f.write("--------------------\n")
        for param in ["alpha", "beta", "gamma"]:
            if param in legacy_results["posterior_samples"]:
                param_samples = legacy_results["posterior_samples"][param]
                f.write(f"{param}:\n")
                f.write(f"  Mean: {np.mean(param_samples):.6f}\n")
                f.write(f"  Std: {np.std(param_samples):.6f}\n")
                f.write(f"  Min: {np.min(param_samples):.6f}\n")
                f.write(f"  Max: {np.max(param_samples):.6f}\n")
        f.write("\n")

        # Write validation conclusion
        f.write("Validation Conclusion\n")
        f.write("--------------------\n")
        if legacy_results['velocity'] is not None:
            f.write("The legacy model was trained successfully and produced velocity estimates.\n")
            f.write("The workflow of generate_posterior_samples() followed by compute_statistics_from_posterior_samples() works as expected.\n")
            f.write("Velocity is stored in adata.layers['velocity_pyro'] as expected.\n")
        else:
            f.write("The legacy model was trained successfully but did not produce velocity estimates.\n")
            f.write("Further investigation is needed to understand why velocity estimates were not produced.\n")

    print(f"Summary report saved to {report_path}")
    return report_path


def main():
    """Run the legacy model validation and save results."""
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

    # Generate summary report
    try:
        summary_report = generate_summary_report(legacy_results, args, args.output_dir)
        print(f"Results saved to {args.output_dir}")
    except Exception as e:
        print(f"Error generating summary report: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()
