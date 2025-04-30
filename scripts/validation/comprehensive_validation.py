#!/usr/bin/env python
"""
Comprehensive validation script for PyroVelocity.

This script demonstrates advanced usage of the validation framework to perform
a comprehensive comparison of different implementations of PyroVelocity
(legacy, modular, and JAX) with detailed analysis and visualization.

The comprehensive validation includes:
1. Running multiple implementations with custom configurations
2. Comparing parameter estimates across implementations
3. Comparing velocity estimates across implementations
4. Comparing uncertainty estimates across implementations
5. Comparing performance metrics across implementations
6. Performing statistical analysis of differences
7. Detecting outliers and systematic biases
8. Identifying edge cases and potential issues
9. Generating detailed reports and visualizations

This script provides a command-line interface for running comprehensive validation
with different options and saving the results to disk. It demonstrates advanced
analysis techniques for comparing model implementations.

Example usage:
    # Run comprehensive validation with default settings
    python comprehensive_validation.py

    # Run comprehensive validation with specific parameters
    python comprehensive_validation.py --max-epochs 500 --num-samples 50 --use-scalene
"""

import os
import sys
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import anndata as ad
import scanpy as sc

from pyrovelocity.io.datasets import pancreas
from pyrovelocity.validation.framework import ValidationRunner
from pyrovelocity.validation.comparison import (
    compare_parameters,
    compare_velocities,
    compare_uncertainties,
    compare_performance,
    statistical_comparison,
    detect_outliers,
    detect_systematic_bias,
    identify_edge_cases,
)
from pyrovelocity.validation.visualization import (
    plot_parameter_comparison,
    plot_velocity_comparison,
    plot_uncertainty_comparison,
    plot_performance_comparison,
    plot_parameter_distributions,
    plot_velocity_vector_field,
    plot_uncertainty_heatmap,
    plot_performance_radar,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Comprehensive validation script for PyroVelocity"
    )
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
        "--output-dir",
        type=str,
        default="validation_results",
        help="Directory to save validation results",
    )
    return parser.parse_args()


def main():
    """Run comprehensive validation and save results.

    This function:
    1. Parses command line arguments
    2. Loads and preprocesses data
    3. Sets up models with custom configurations
    4. Runs validation on all implementations
    5. Performs detailed analysis and comparison
    6. Generates visualizations and reports
    7. Saves results to disk
    """
    # Parse command line arguments
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    # The pancreas dataset is a small single-cell RNA-seq dataset
    # that is suitable for testing and validation
    print("Loading data...")
    start_time = time.time()
    adata = pancreas()
    print(f"Loaded dataset with {adata.shape[0]} cells and {adata.shape[1]} genes")
    print(f"Loading completed in {time.time() - start_time:.2f} seconds")

    # Subset data for faster validation
    # For demonstration purposes, we use a small subset of cells and genes
    print("Subsetting data...")
    adata = adata[:100, :100].copy()
    print(f"Subset dataset to {adata.shape[0]} cells and {adata.shape[1]} genes")

    # Preprocess data
    # Standard scanpy preprocessing pipeline for single-cell data
    print("Preprocessing data...")
    start_time = time.time()
    sc.pp.normalize_per_cell(adata)  # Normalize by library size
    sc.pp.log1p(adata)               # Log-transform
    sc.pp.pca(adata)                 # Dimensionality reduction with PCA
    sc.pp.neighbors(adata)           # Compute neighbor graph
    sc.tl.umap(adata)                # Compute UMAP embedding
    print(f"Preprocessing completed in {time.time() - start_time:.2f} seconds")

    # Initialize ValidationRunner
    # The ValidationRunner handles the validation process for all implementations
    print("Initializing ValidationRunner...")
    runner = ValidationRunner(adata)

    # Set up models with custom configurations
    # We can customize each model's configuration separately
    print("Setting up models...")

    # Legacy model with deterministic configuration
    # This uses the original PyroVelocity implementation
    print("Setting up legacy model...")
    runner.setup_legacy_model(
        model_type="deterministic",  # Use deterministic model
        latent_time=True,            # Include latent time
        likelihood="Poisson",        # Use Poisson likelihood
    )

    # Modular model with standard configuration
    # This uses the PyTorch/Pyro modular implementation
    print("Setting up modular model...")
    runner.setup_modular_model(
        model_type="standard",       # Use standard dynamics model
        latent_time=True,            # Include latent time
        likelihood="poisson",        # Use Poisson likelihood
        prior="lognormal",           # Use LogNormal prior
        guide="auto",                # Use auto guide
    )

    # JAX model with standard configuration
    # This uses the JAX/NumPyro implementation
    print("Setting up JAX model...")
    runner.setup_jax_model(
        model_type="standard",       # Use standard dynamics model
        latent_time=True,            # Include latent time
        likelihood="poisson",        # Use Poisson likelihood
        prior="lognormal",           # Use LogNormal prior
        guide="auto_normal",         # Use auto normal guide
    )

    # Run validation
    # This will train models, generate posterior samples, and compute metrics
    print("Running validation...")
    start_time = time.time()
    results = runner.run_validation(
        max_epochs=args.max_epochs,
        num_samples=args.num_samples,
        use_scalene=args.use_scalene,
        batch_size=64,               # Use mini-batch training
        learning_rate=0.01,          # Set learning rate
        early_stopping=True,         # Enable early stopping
    )
    print(f"Validation completed in {time.time() - start_time:.2f} seconds")

    # Compare implementations
    print("Comparing implementations...")
    comparison = runner.compare_implementations()

    # Save results
    print("Saving results...")
    np.save(os.path.join(args.output_dir, "results.npy"), results)
    np.save(os.path.join(args.output_dir, "comparison.npy"), comparison)

    # Plot parameter comparison
    print("Plotting parameter comparison...")
    fig = plot_parameter_comparison(comparison["parameter_comparison"])
    fig.savefig(os.path.join(args.output_dir, "parameter_comparison.png"))
    plt.close(fig)

    # Plot parameter distributions
    print("Plotting parameter distributions...")
    for param in ["alpha", "beta", "gamma"]:
        fig = plot_parameter_distributions(results, param)
        fig.savefig(os.path.join(args.output_dir, f"{param}_distributions.png"))
        plt.close(fig)

    # Plot velocity comparison
    print("Plotting velocity comparison...")
    fig = plot_velocity_comparison(comparison["velocity_comparison"])
    fig.savefig(os.path.join(args.output_dir, "velocity_comparison.png"))
    plt.close(fig)

    # Plot velocity vector field
    print("Plotting velocity vector field...")
    coordinates = adata.obsm["X_umap"]
    fig = plot_velocity_vector_field(results, coordinates)
    fig.savefig(os.path.join(args.output_dir, "velocity_vector_field.png"))
    plt.close(fig)

    # Plot uncertainty comparison
    print("Plotting uncertainty comparison...")
    fig = plot_uncertainty_comparison(comparison["uncertainty_comparison"])
    fig.savefig(os.path.join(args.output_dir, "uncertainty_comparison.png"))
    plt.close(fig)

    # Plot uncertainty heatmap
    print("Plotting uncertainty heatmap...")
    fig = plot_uncertainty_heatmap(results)
    fig.savefig(os.path.join(args.output_dir, "uncertainty_heatmap.png"))
    plt.close(fig)

    # Plot performance comparison
    print("Plotting performance comparison...")
    fig = plot_performance_comparison(comparison["performance_comparison"])
    fig.savefig(os.path.join(args.output_dir, "performance_comparison.png"))
    plt.close(fig)

    # Plot performance radar
    print("Plotting performance radar...")
    fig = plot_performance_radar(results)
    fig.savefig(os.path.join(args.output_dir, "performance_radar.png"))
    plt.close(fig)

    # Perform statistical comparison
    print("Performing statistical comparison...")
    statistical_results = {}

    # Compare parameters
    for param in ["alpha", "beta", "gamma"]:
        if param in results["legacy"]["posterior_samples"] and param in results["modular"]["posterior_samples"]:
            legacy_param = np.array(results["legacy"]["posterior_samples"][param])
            modular_param = np.array(results["modular"]["posterior_samples"][param])
            statistical_results[f"{param}_legacy_vs_modular"] = statistical_comparison(
                legacy_param.flatten(), modular_param.flatten()
            )

        if param in results["legacy"]["posterior_samples"] and param in results["jax"]["posterior_samples"]:
            legacy_param = np.array(results["legacy"]["posterior_samples"][param])
            jax_param = np.array(results["jax"]["posterior_samples"][param])
            statistical_results[f"{param}_legacy_vs_jax"] = statistical_comparison(
                legacy_param.flatten(), jax_param.flatten()
            )

        if param in results["modular"]["posterior_samples"] and param in results["jax"]["posterior_samples"]:
            modular_param = np.array(results["modular"]["posterior_samples"][param])
            jax_param = np.array(results["jax"]["posterior_samples"][param])
            statistical_results[f"{param}_modular_vs_jax"] = statistical_comparison(
                modular_param.flatten(), jax_param.flatten()
            )

    # Save statistical results
    with open(os.path.join(args.output_dir, "statistical_results.txt"), "w") as f:
        for key, value in statistical_results.items():
            f.write(f"{key}:\n")
            for test, results in value.items():
                f.write(f"  {test}:\n")
                for metric, val in results.items():
                    f.write(f"    {metric}: {val}\n")
            f.write("\n")

    # Detect outliers
    print("Detecting outliers...")
    outlier_results = {}

    # Detect outliers in parameters
    for param in ["alpha", "beta", "gamma"]:
        for impl in ["legacy", "modular", "jax"]:
            if param in results[impl]["posterior_samples"]:
                param_values = np.array(results[impl]["posterior_samples"][param])
                outliers = detect_outliers(param_values.flatten())
                outlier_results[f"{param}_{impl}_outliers"] = outliers

    # Save outlier results
    with open(os.path.join(args.output_dir, "outlier_results.txt"), "w") as f:
        for key, value in outlier_results.items():
            f.write(f"{key}: {value}\n")

    # Detect systematic bias
    print("Detecting systematic bias...")
    bias_results = {}

    # Detect bias in parameters
    for param in ["alpha", "beta", "gamma"]:
        if param in results["legacy"]["posterior_samples"] and param in results["modular"]["posterior_samples"]:
            legacy_param = np.array(results["legacy"]["posterior_samples"][param])
            modular_param = np.array(results["modular"]["posterior_samples"][param])
            bias_results[f"{param}_legacy_vs_modular"] = detect_systematic_bias(
                legacy_param.flatten(), modular_param.flatten()
            )

        if param in results["legacy"]["posterior_samples"] and param in results["jax"]["posterior_samples"]:
            legacy_param = np.array(results["legacy"]["posterior_samples"][param])
            jax_param = np.array(results["jax"]["posterior_samples"][param])
            bias_results[f"{param}_legacy_vs_jax"] = detect_systematic_bias(
                legacy_param.flatten(), jax_param.flatten()
            )

        if param in results["modular"]["posterior_samples"] and param in results["jax"]["posterior_samples"]:
            modular_param = np.array(results["modular"]["posterior_samples"][param])
            jax_param = np.array(results["jax"]["posterior_samples"][param])
            bias_results[f"{param}_modular_vs_jax"] = detect_systematic_bias(
                modular_param.flatten(), jax_param.flatten()
            )

    # Save bias results
    with open(os.path.join(args.output_dir, "bias_results.txt"), "w") as f:
        for key, value in bias_results.items():
            f.write(f"{key}:\n")
            for metric, val in value.items():
                f.write(f"  {metric}: {val}\n")
            f.write("\n")

    # Identify edge cases
    print("Identifying edge cases...")
    edge_case_results = {}

    # Identify edge cases in parameters
    for param in ["alpha", "beta", "gamma"]:
        for impl in ["legacy", "modular", "jax"]:
            if param in results[impl]["posterior_samples"]:
                param_values = np.array(results[impl]["posterior_samples"][param])
                edge_cases = identify_edge_cases(param_values)
                edge_case_results[f"{param}_{impl}_edge_cases"] = edge_cases

    # Save edge case results
    with open(os.path.join(args.output_dir, "edge_case_results.txt"), "w") as f:
        for key, value in edge_case_results.items():
            f.write(f"{key}:\n")
            for metric, val in value.items():
                if metric != "extreme_values":
                    f.write(f"  {metric}: {val}\n")
                else:
                    f.write(f"  {metric}: {len(val)} values\n")
            f.write("\n")

    print(f"Validation results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
