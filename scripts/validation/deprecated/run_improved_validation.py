#!/usr/bin/env python
"""
Run Improved Validation Script for PyroVelocity.

This script uses the improved validation framework to validate that the
PyTorch/Pyro modular implementation produces results equivalent to the
legacy implementation using the preprocessed pancreas fixture data.

Example usage:
    # Run validation with default settings
    python run_improved_validation.py

    # Run validation with specific parameters
    python run_improved_validation.py --max-epochs 20 --num-samples 10
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
from improved_framework import ImprovedValidationRunner


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Improved Validation Script for PyroVelocity"
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=20,
        help="Maximum number of epochs for training",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of posterior samples to generate",
    )
    parser.add_argument(
        "--use-scalene",
        action="store_true",
        help="Whether to use Scalene for performance profiling",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="improved_validation_results",
        help="Directory to save validation results",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.01,
        help="Learning rate for training",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for training",
    )
    parser.add_argument(
        "--use-gpu",
        type=str,
        default="auto",
        choices=["auto", "true", "false"],
        help="Whether to use GPU for training",
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


def save_results(results, output_dir):
    """Save validation results to files."""
    os.makedirs(output_dir, exist_ok=True)

    # Save summary of results
    with open(os.path.join(output_dir, "validation_summary.txt"), "w") as f:
        f.write("PyroVelocity Validation Summary\n")
        f.write("============================\n\n")

        # Write model information
        f.write("Model Information\n")
        f.write("----------------\n")

        for model_name, model_result in results.items():
            f.write(f"{model_name}:\n")

            # Check if model failed
            if "error" in model_result:
                f.write(f"  Error: {model_result['error']}\n")
                continue

            # Write performance information
            if "performance" in model_result:
                perf = model_result["performance"]
                f.write(f"  Training time: {perf.get('training_time', 'N/A'):.2f} seconds\n")
                f.write(f"  Inference time: {perf.get('inference_time', 'N/A'):.2f} seconds\n")

            # Write velocity shape
            if "velocity" in model_result:
                velocity = model_result["velocity"]
                if hasattr(velocity, "shape"):
                    f.write(f"  Velocity shape: {velocity.shape}\n")
                else:
                    f.write(f"  Velocity type: {type(velocity)}\n")

            f.write("\n")

    print(f"Results saved to {output_dir}")


def main():
    """Run validation and save results."""
    # Parse command line arguments
    args = parse_args()

    # Set random seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    pyro.set_rng_seed(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load preprocessed pancreas data
    try:
        adata = load_preprocessed_pancreas_data()
    except Exception as e:
        print(f"Failed to load data: {e}")
        return

    # Initialize ImprovedValidationRunner
    print("Initializing ImprovedValidationRunner...")
    runner = ImprovedValidationRunner(adata)

    # Set up legacy model
    print("Setting up legacy model...")
    runner.setup_legacy_model()

    # Set up modular model
    print("Setting up modular model...")
    runner.setup_modular_model()

    # Run validation
    print(f"Running validation with max_epochs={args.max_epochs}, num_samples={args.num_samples}...")
    start_time = time.time()

    try:
        # Run validation
        results = runner.run_validation(
            max_epochs=args.max_epochs,
            num_samples=args.num_samples,
            use_scalene=args.use_scalene,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            use_gpu=args.use_gpu,
        )

        # Save results
        save_results(results, args.output_dir)

        # Check if any models failed
        failed_models = []
        for model_name, model_result in results.items():
            if "error" in model_result:
                failed_models.append(model_name)
                print(f"Model {model_name} failed: {model_result['error']}")

        # Compare implementations only if both models succeeded
        if len(failed_models) == 0:
            print("Comparing implementations...")
            comparison = runner.compare_implementations()

            # Save comparison results
            with open(os.path.join(args.output_dir, "comparison_results.txt"), "w") as f:
                f.write("PyroVelocity Comparison Results\n")
                f.write("=============================\n\n")

                # Write parameter comparison
                f.write("Parameter Comparison\n")
                f.write("-------------------\n")
                for param, param_comp in comparison["parameter_comparison"].items():
                    f.write(f"{param}:\n")
                    for comp_name, metrics in param_comp.items():
                        f.write(f"  {comp_name}:\n")
                        for metric_name, value in metrics.items():
                            if isinstance(value, float):
                                f.write(f"    {metric_name}: {value:.6f}\n")
                            else:
                                f.write(f"    {metric_name}: {value}\n")
                    f.write("\n")

                # Write velocity comparison
                f.write("Velocity Comparison\n")
                f.write("------------------\n")
                for comp_name, metrics in comparison["velocity_comparison"].items():
                    f.write(f"{comp_name}:\n")
                    for metric_name, value in metrics.items():
                        if isinstance(value, float):
                            f.write(f"  {metric_name}: {value:.6f}\n")
                        else:
                            f.write(f"  {metric_name}: {value}\n")
                f.write("\n")

                # Write uncertainty comparison
                f.write("Uncertainty Comparison\n")
                f.write("---------------------\n")
                for comp_name, metrics in comparison["uncertainty_comparison"].items():
                    f.write(f"{comp_name}:\n")
                    for metric_name, value in metrics.items():
                        if isinstance(value, float):
                            f.write(f"  {metric_name}: {value:.6f}\n")
                        else:
                            f.write(f"  {metric_name}: {value}\n")
                f.write("\n")

                # Write performance comparison
                f.write("Performance Comparison\n")
                f.write("---------------------\n")
                for comp_name, metrics in comparison["performance_comparison"].items():
                    f.write(f"{comp_name}:\n")
                    for metric_name, value in metrics.items():
                        if isinstance(value, float):
                            f.write(f"  {metric_name}: {value:.6f}\n")
                        else:
                            f.write(f"  {metric_name}: {value}\n")
                f.write("\n")
        else:
            print(f"Skipping comparison due to failed models: {failed_models}")

        elapsed_time = time.time() - start_time
        print(f"Validation completed in {elapsed_time:.2f} seconds")

    except Exception as e:
        print(f"Error during validation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
