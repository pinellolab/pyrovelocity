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
from tqdm import tqdm

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


def create_synthetic_data():
    """Create synthetic data for testing when fixture data is not available."""
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

    # Add library size information
    adata.obs["u_lib_size_raw"] = np.sum(u_data, axis=1)
    adata.obs["s_lib_size_raw"] = np.sum(s_data, axis=1)
    # Add small epsilon to avoid log(0)
    adata.obs["u_lib_size"] = np.log(adata.obs["u_lib_size_raw"] + 1e-6)
    adata.obs["s_lib_size"] = np.log(adata.obs["s_lib_size_raw"] + 1e-6)
    adata.obs["u_lib_size_mean"] = np.mean(adata.obs["u_lib_size"])
    adata.obs["s_lib_size_mean"] = np.mean(adata.obs["s_lib_size"])
    adata.obs["u_lib_size_scale"] = np.std(adata.obs["u_lib_size"])
    adata.obs["s_lib_size_scale"] = np.std(adata.obs["s_lib_size"])
    adata.obs["ind_x"] = np.arange(n_cells)

    print(f"Created synthetic AnnData: {adata.shape[0]} cells, {adata.shape[1]} genes")
    return adata


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
    try:
        with tqdm(total=1, desc="Loading data") as pbar:
            adata = pancreas()
            pbar.update(1)
        print(f"Loaded dataset with {adata.shape[0]} cells and {adata.shape[1]} genes")
        print(f"Loading completed in {time.time() - start_time:.2f} seconds")

        # Subset data for faster validation
        # For demonstration purposes, we use a small subset of cells and genes
        print("Subsetting data...")
        with tqdm(total=1, desc="Subsetting data") as pbar:
            adata = adata[:100, :100].copy()
            pbar.update(1)
        print(f"Subset dataset to {adata.shape[0]} cells and {adata.shape[1]} genes")
    except Exception as e:
        print(f"Error loading pancreas dataset: {e}")
        print("Falling back to synthetic data...")
        with tqdm(total=1, desc="Creating synthetic data") as pbar:
            adata = create_synthetic_data()
            pbar.update(1)
        print(f"Loading completed in {time.time() - start_time:.2f} seconds")

    # Preprocess data
    # Standard scanpy preprocessing pipeline for single-cell data
    print("Preprocessing data...")
    start_time = time.time()
    try:
        with tqdm(total=5, desc="Preprocessing") as pbar:
            # Step 1: Normalize by library size
            sc.pp.normalize_per_cell(adata)
            pbar.update(1)

            # Step 2: Log-transform
            sc.pp.log1p(adata)
            pbar.update(1)

            # Step 3: Dimensionality reduction with PCA
            sc.pp.pca(adata)
            pbar.update(1)

            # Step 4: Compute neighbor graph
            sc.pp.neighbors(adata)
            pbar.update(1)

            # Step 5: Compute UMAP embedding
            sc.tl.umap(adata)
            pbar.update(1)

        print(f"Preprocessing completed in {time.time() - start_time:.2f} seconds")
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        print("Continuing with minimal preprocessing...")
        # Ensure UMAP coordinates exist for visualization
        if "X_umap" not in adata.obsm:
            adata.obsm["X_umap"] = np.random.normal(0, 1, size=(adata.shape[0], 2))

    # Initialize ValidationRunner
    # The ValidationRunner handles the validation process for all implementations
    print("Initializing ValidationRunner...")
    try:
        runner = ValidationRunner(adata)
    except Exception as e:
        print(f"Error initializing ValidationRunner: {e}")
        print("Please check that the AnnData object has the required fields.")
        return

    # Set up models with custom configurations
    # We can customize each model's configuration separately
    print("Setting up models...")

    try:
        with tqdm(total=3, desc="Setting up models") as pbar:
            # Legacy model with deterministic configuration
            # This uses the original PyroVelocity implementation
            print("Setting up legacy model...")
            runner.setup_legacy_model(
                model_type="deterministic",  # Use deterministic model
                latent_time=True,            # Include latent time
                likelihood="Poisson",        # Use Poisson likelihood
            )
            pbar.update(1)

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
            pbar.update(1)

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
            pbar.update(1)
    except Exception as e:
        print(f"Error setting up models: {e}")
        print("Please check the model configurations.")
        return

    # Run validation
    # This will train models, generate posterior samples, and compute metrics
    print("Running validation...")
    start_time = time.time()
    try:
        results = runner.run_validation(
            max_epochs=args.max_epochs,
            num_samples=args.num_samples,
            use_scalene=args.use_scalene,
            batch_size=64,               # Use mini-batch training
            learning_rate=0.01,          # Set learning rate
            early_stopping=True,         # Enable early stopping
        )
        print(f"Validation completed in {time.time() - start_time:.2f} seconds")
    except Exception as e:
        print(f"Error during validation: {e}")
        print("Trying again with reduced complexity...")
        try:
            # Try with reduced complexity
            results = runner.run_validation(
                max_epochs=max(5, args.max_epochs // 4),  # Reduce epochs
                num_samples=max(5, args.num_samples // 2),  # Reduce samples
                use_scalene=False,  # Disable profiling
                batch_size=32,      # Smaller batch size
                learning_rate=0.005,  # Lower learning rate
                early_stopping=True,
            )
            print(f"Validation completed in {time.time() - start_time:.2f} seconds")
        except Exception as e:
            print(f"Validation failed: {e}")
            print("Cannot proceed without validation results.")
            return

    # Compare implementations
    print("Comparing implementations...")
    try:
        with tqdm(total=1, desc="Comparing implementations") as pbar:
            comparison = runner.compare_implementations()
            pbar.update(1)
    except Exception as e:
        print(f"Error comparing implementations: {e}")
        print("Using minimal comparison results.")
        # Create minimal comparison results to continue
        comparison = {
            "parameter_comparison": {},
            "velocity_comparison": {},
            "uncertainty_comparison": {},
            "performance_comparison": {},
        }

    # Save results
    print("Saving results...")
    try:
        with tqdm(total=2, desc="Saving results") as pbar:
            np.save(os.path.join(args.output_dir, "results.npy"), results)
            pbar.update(1)
            np.save(os.path.join(args.output_dir, "comparison.npy"), comparison)
            pbar.update(1)
    except Exception as e:
        print(f"Error saving results: {e}")

    # Create visualizations with progress tracking
    print("Creating visualizations...")

    # Count total number of plots to create
    total_plots = 8  # Base plots
    params = ["alpha", "beta", "gamma"]
    total_plots += len(params)  # Parameter distribution plots

    with tqdm(total=total_plots, desc="Creating plots") as pbar:
        # Plot parameter comparison
        try:
            print("Plotting parameter comparison...")
            fig = plot_parameter_comparison(comparison.get("parameter_comparison", {}))
            fig.savefig(os.path.join(args.output_dir, "parameter_comparison.png"))
            plt.close(fig)
        except Exception as e:
            print(f"Error plotting parameter comparison: {e}")
        pbar.update(1)

        # Plot parameter distributions
        print("Plotting parameter distributions...")
        for param in params:
            try:
                fig = plot_parameter_distributions(results, param)
                fig.savefig(os.path.join(args.output_dir, f"{param}_distributions.png"))
                plt.close(fig)
            except Exception as e:
                print(f"Error plotting {param} distributions: {e}")
            pbar.update(1)

        # Plot velocity comparison
        try:
            print("Plotting velocity comparison...")
            fig = plot_velocity_comparison(comparison.get("velocity_comparison", {}))
            fig.savefig(os.path.join(args.output_dir, "velocity_comparison.png"))
            plt.close(fig)
        except Exception as e:
            print(f"Error plotting velocity comparison: {e}")
        pbar.update(1)

        # Plot velocity vector field
        try:
            print("Plotting velocity vector field...")
            coordinates = adata.obsm["X_umap"]
            fig = plot_velocity_vector_field(results, coordinates)
            fig.savefig(os.path.join(args.output_dir, "velocity_vector_field.png"))
            plt.close(fig)
        except Exception as e:
            print(f"Error plotting velocity vector field: {e}")
        pbar.update(1)

        # Plot uncertainty comparison
        try:
            print("Plotting uncertainty comparison...")
            fig = plot_uncertainty_comparison(comparison.get("uncertainty_comparison", {}))
            fig.savefig(os.path.join(args.output_dir, "uncertainty_comparison.png"))
            plt.close(fig)
        except Exception as e:
            print(f"Error plotting uncertainty comparison: {e}")
        pbar.update(1)

        # Plot uncertainty heatmap
        try:
            print("Plotting uncertainty heatmap...")
            fig = plot_uncertainty_heatmap(results)
            fig.savefig(os.path.join(args.output_dir, "uncertainty_heatmap.png"))
            plt.close(fig)
        except Exception as e:
            print(f"Error plotting uncertainty heatmap: {e}")
        pbar.update(1)

        # Plot performance comparison
        try:
            print("Plotting performance comparison...")
            fig = plot_performance_comparison(comparison.get("performance_comparison", {}))
            fig.savefig(os.path.join(args.output_dir, "performance_comparison.png"))
            plt.close(fig)
        except Exception as e:
            print(f"Error plotting performance comparison: {e}")
        pbar.update(1)

        # Plot performance radar
        try:
            print("Plotting performance radar...")
            fig = plot_performance_radar(results)
            fig.savefig(os.path.join(args.output_dir, "performance_radar.png"))
            plt.close(fig)
        except Exception as e:
            print(f"Error plotting performance radar: {e}")
        pbar.update(1)

    # Perform statistical analysis with progress tracking
    print("Performing statistical analysis...")

    # Count total number of analysis steps
    params = ["alpha", "beta", "gamma"]
    implementations = ["legacy", "modular", "jax"]
    comparison_pairs = [("legacy", "modular"), ("legacy", "jax"), ("modular", "jax")]

    # Calculate total steps: statistical comparison + outlier detection + bias detection + edge case detection
    total_steps = (len(params) * len(comparison_pairs)) + (len(params) * len(implementations)) * 2 + (len(params) * len(comparison_pairs))

    with tqdm(total=total_steps, desc="Statistical analysis") as pbar:
        # 1. Perform statistical comparison
        print("Performing statistical comparison...")
        statistical_results = {}

        try:
            # Compare parameters
            for param in params:
                for impl1, impl2 in comparison_pairs:
                    try:
                        if (impl1 in results and impl2 in results and
                            "posterior_samples" in results[impl1] and "posterior_samples" in results[impl2] and
                            param in results[impl1]["posterior_samples"] and param in results[impl2]["posterior_samples"]):

                            param1 = np.array(results[impl1]["posterior_samples"][param])
                            param2 = np.array(results[impl2]["posterior_samples"][param])

                            # Ensure arrays are flattened and have the same shape
                            param1_flat = param1.flatten()
                            param2_flat = param2.flatten()

                            # Use the minimum length if they differ
                            min_len = min(len(param1_flat), len(param2_flat))
                            param1_flat = param1_flat[:min_len]
                            param2_flat = param2_flat[:min_len]

                            statistical_results[f"{param}_{impl1}_vs_{impl2}"] = statistical_comparison(
                                param1_flat, param2_flat
                            )
                    except Exception as e:
                        print(f"Error comparing {param} between {impl1} and {impl2}: {e}")
                    pbar.update(1)

            # Save statistical results
            with open(os.path.join(args.output_dir, "statistical_results.txt"), "w") as f:
                for key, value in statistical_results.items():
                    f.write(f"{key}:\n")
                    for test, test_results in value.items():
                        f.write(f"  {test}:\n")
                        for metric, val in test_results.items():
                            f.write(f"    {metric}: {val}\n")
                    f.write("\n")
        except Exception as e:
            print(f"Error in statistical comparison: {e}")

        # 2. Detect outliers
        print("Detecting outliers...")
        outlier_results = {}

        try:
            # Detect outliers in parameters
            for param in params:
                for impl in implementations:
                    try:
                        if (impl in results and "posterior_samples" in results[impl] and
                            param in results[impl]["posterior_samples"]):

                            param_values = np.array(results[impl]["posterior_samples"][param])
                            outliers = detect_outliers(param_values.flatten())
                            outlier_results[f"{param}_{impl}_outliers"] = outliers
                    except Exception as e:
                        print(f"Error detecting outliers for {param} in {impl}: {e}")
                    pbar.update(1)

            # Save outlier results
            with open(os.path.join(args.output_dir, "outlier_results.txt"), "w") as f:
                for key, value in outlier_results.items():
                    f.write(f"{key}: {value}\n")
        except Exception as e:
            print(f"Error in outlier detection: {e}")

        # 3. Detect systematic bias
        print("Detecting systematic bias...")
        bias_results = {}

        try:
            # Detect bias in parameters
            for param in params:
                for impl1, impl2 in comparison_pairs:
                    try:
                        if (impl1 in results and impl2 in results and
                            "posterior_samples" in results[impl1] and "posterior_samples" in results[impl2] and
                            param in results[impl1]["posterior_samples"] and param in results[impl2]["posterior_samples"]):

                            param1 = np.array(results[impl1]["posterior_samples"][param])
                            param2 = np.array(results[impl2]["posterior_samples"][param])

                            # Ensure arrays are flattened and have the same shape
                            param1_flat = param1.flatten()
                            param2_flat = param2.flatten()

                            # Use the minimum length if they differ
                            min_len = min(len(param1_flat), len(param2_flat))
                            param1_flat = param1_flat[:min_len]
                            param2_flat = param2_flat[:min_len]

                            bias_results[f"{param}_{impl1}_vs_{impl2}"] = detect_systematic_bias(
                                param1_flat, param2_flat
                            )
                    except Exception as e:
                        print(f"Error detecting bias for {param} between {impl1} and {impl2}: {e}")
                    pbar.update(1)

            # Save bias results
            with open(os.path.join(args.output_dir, "bias_results.txt"), "w") as f:
                for key, value in bias_results.items():
                    f.write(f"{key}:\n")
                    for metric, val in value.items():
                        f.write(f"  {metric}: {val}\n")
                    f.write("\n")
        except Exception as e:
            print(f"Error in bias detection: {e}")

        # 4. Identify edge cases
        print("Identifying edge cases...")
        edge_case_results = {}

        try:
            # Identify edge cases in parameters
            for param in params:
                for impl in implementations:
                    try:
                        if (impl in results and "posterior_samples" in results[impl] and
                            param in results[impl]["posterior_samples"]):

                            param_values = np.array(results[impl]["posterior_samples"][param])
                            edge_cases = identify_edge_cases(param_values)
                            edge_case_results[f"{param}_{impl}_edge_cases"] = edge_cases
                    except Exception as e:
                        print(f"Error identifying edge cases for {param} in {impl}: {e}")
                    pbar.update(1)

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
        except Exception as e:
            print(f"Error in edge case identification: {e}")

    # Print summary
    print("\nValidation Summary:")
    print(f"- Output directory: {args.output_dir}")
    print(f"- Max epochs: {args.max_epochs}")
    print(f"- Num samples: {args.num_samples}")
    print(f"- Implementations: {', '.join(impl for impl in implementations if impl in results)}")
    print(f"- Parameters analyzed: {', '.join(params)}")
    print(f"- Statistical comparisons: {len(statistical_results)}")
    print(f"- Outliers detected: {len(outlier_results)}")
    print(f"- Bias analyses: {len(bias_results)}")
    print(f"- Edge cases identified: {len(edge_case_results)}")

    print(f"\nValidation results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
