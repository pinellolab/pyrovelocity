#!/usr/bin/env python
"""
Basic validation script for PyroVelocity.

This script demonstrates how to use the validation framework to compare
different implementations of PyroVelocity (legacy, modular, and JAX).

The validation framework enables:
1. Running multiple implementations on the same data
2. Comparing parameter estimates across implementations
3. Comparing velocity estimates across implementations
4. Comparing uncertainty estimates across implementations
5. Comparing performance metrics across implementations

This script provides a command-line interface for running validation
with different options and saving the results to disk. It can be used
to validate and compare any combination of the three implementations.

Example usage:
    # Run validation with all implementations
    python basic_validation.py --use-legacy --use-modular --use-jax

    # Run validation with specific implementations and parameters
    python basic_validation.py --use-modular --use-jax --max-epochs 500 --num-samples 50
"""

import os
import sys
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import anndata as ad

from pyrovelocity.io.datasets import pancreas
from pyrovelocity.validation.framework import ValidationRunner, run_validation
from pyrovelocity.validation.visualization import (
    plot_parameter_comparison,
    plot_velocity_comparison,
    plot_uncertainty_comparison,
    plot_performance_comparison,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Basic validation script for PyroVelocity")
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=10,
        help="Maximum number of epochs for training",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=30,
        help="Number of posterior samples to generate",
    )
    parser.add_argument(
        "--use-scalene",
        action="store_true",
        help="Whether to use Scalene for performance profiling",
    )
    parser.add_argument(
        "--use-legacy",
        action="store_true",
        help="Whether to use the legacy implementation",
    )
    parser.add_argument(
        "--use-modular",
        action="store_true",
        help="Whether to use the modular implementation",
    )
    parser.add_argument(
        "--use-jax",
        action="store_true",
        help="Whether to use the JAX implementation",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="validation_results",
        help="Directory to save validation results",
    )
    return parser.parse_args()


def main():
    """Run validation and save results.

    This function:
    1. Parses command line arguments
    2. Loads and prepares data
    3. Runs validation on selected implementations
    4. Generates comparison visualizations
    5. Saves results to disk
    """
    # Parse command line arguments
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    # The pancreas dataset is a small single-cell RNA-seq dataset
    # that is suitable for testing and validation
    print("Loading data...")
    adata = pancreas()
    print(f"Loaded dataset with {adata.shape[0]} cells and {adata.shape[1]} genes")

    # Subset data for faster validation
    # For demonstration purposes, we use a small subset of cells and genes
    print("Subsetting data for faster validation...")
    adata = adata[:100, :100].copy()
    print(f"Subset dataset to {adata.shape[0]} cells and {adata.shape[1]} genes")

    # Check which implementations are selected
    implementations = []
    if args.use_legacy:
        implementations.append("legacy")
    if args.use_modular:
        implementations.append("modular")
    if args.use_jax:
        implementations.append("jax")

    if not implementations:
        print("No implementations selected. Please use at least one of --use-legacy, --use-modular, or --use-jax.")
        return

    print(f"Running validation with implementations: {', '.join(implementations)}")
    print(f"Parameters: max_epochs={args.max_epochs}, num_samples={args.num_samples}")

    # Run validation
    # This will train models, generate posterior samples, and compare results
    print("Running validation...")
    start_time = time.time()
    validation_results = run_validation(
        adata=adata,
        max_epochs=args.max_epochs,
        num_samples=args.num_samples,
        use_scalene=args.use_scalene,
        use_legacy=args.use_legacy,
        use_modular=args.use_modular,
        use_jax=args.use_jax,
    )
    elapsed_time = time.time() - start_time
    print(f"Validation completed in {elapsed_time:.2f} seconds")

    # Extract results and comparison
    # The validation_results dictionary contains two keys:
    # - "results": Dictionary of validation results for each model
    # - "comparison": Dictionary of comparison results across implementations
    model_results = validation_results["results"]
    comparison = validation_results["comparison"]

    # Save model results
    # We save the parameter estimates for each model
    print("Saving model results...")
    for model_name, model_result in model_results.items():
        if "parameters" in model_result:
            params = model_result["parameters"]
            for param_name, param_value in params.items():
                np.save(
                    os.path.join(args.output_dir, f"{model_name}_{param_name}.npy"),
                    param_value
                )

    # Plot parameter comparison
    # This compares alpha, beta, gamma estimates across implementations
    print("Plotting parameter comparison...")
    fig = plot_parameter_comparison(comparison["parameter_comparison"])
    fig.savefig(os.path.join(args.output_dir, "parameter_comparison.png"))
    plt.close(fig)

    # Plot velocity comparison
    # This compares velocity vectors across implementations
    print("Plotting velocity comparison...")
    fig = plot_velocity_comparison(comparison["velocity_comparison"])
    fig.savefig(os.path.join(args.output_dir, "velocity_comparison.png"))
    plt.close(fig)

    # Plot uncertainty comparison
    # This compares uncertainty estimates across implementations
    print("Plotting uncertainty comparison...")
    fig = plot_uncertainty_comparison(comparison["uncertainty_comparison"])
    fig.savefig(os.path.join(args.output_dir, "uncertainty_comparison.png"))
    plt.close(fig)

    # Plot performance comparison
    # This compares training and inference times across implementations
    print("Plotting performance comparison...")
    fig = plot_performance_comparison(comparison["performance_comparison"])
    fig.savefig(os.path.join(args.output_dir, "performance_comparison.png"))
    plt.close(fig)

    # Save a summary of the validation
    with open(os.path.join(args.output_dir, "validation_summary.txt"), "w") as f:
        f.write(f"Validation Summary\n")
        f.write(f"=================\n\n")
        f.write(f"Implementations: {', '.join(implementations)}\n")
        f.write(f"Parameters: max_epochs={args.max_epochs}, num_samples={args.num_samples}\n")
        f.write(f"Dataset: {adata.shape[0]} cells, {adata.shape[1]} genes\n\n")

        f.write(f"Performance Summary\n")
        f.write(f"-----------------\n")
        for model_name in implementations:
            if model_name in model_results and "performance" in model_results[model_name]:
                perf = model_results[model_name]["performance"]
                f.write(f"{model_name}:\n")
                f.write(f"  Training time: {perf.get('training_time', 'N/A'):.2f} seconds\n")
                f.write(f"  Inference time: {perf.get('inference_time', 'N/A'):.2f} seconds\n")

    print(f"Validation results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
