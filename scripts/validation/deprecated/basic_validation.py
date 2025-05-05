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
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import anndata as ad

from pyrovelocity.io.datasets import pancreas
from pyrovelocity.validation.framework import run_validation
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
    parser.add_argument(
        "--normalize-method",
        type=str,
        default="nearest",
        choices=["nearest", "linear", "cubic"],
        help="Method for normalizing shapes when comparing implementations",
    )
    parser.add_argument(
        "--target-strategy",
        type=str,
        default="max",
        choices=["max", "min", "first", "second"],
        help="Strategy for determining target shape when normalizing",
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
    adata.layers["raw_spliced"] = s_data.copy()  # Required by legacy model
    adata.layers["raw_unspliced"] = u_data.copy()  # Required by legacy model
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
    try:
        adata = pancreas()
        print(f"Loaded dataset with {adata.shape[0]} cells and {adata.shape[1]} genes")

        # Subset data for faster validation
        # For demonstration purposes, we use a small subset of cells and genes
        print("Subsetting data for faster validation...")
        adata = adata[:100, :100].copy()
        print(f"Subset dataset to {adata.shape[0]} cells and {adata.shape[1]} genes")

        # Ensure raw layers are present (required by legacy model)
        if "raw_spliced" not in adata.layers:
            adata.layers["raw_spliced"] = adata.layers["spliced"].copy()
        if "raw_unspliced" not in adata.layers:
            adata.layers["raw_unspliced"] = adata.layers["unspliced"].copy()

        # Ensure library size information is present
        if "u_lib_size_raw" not in adata.obs:
            adata.obs["u_lib_size_raw"] = np.sum(adata.layers["unspliced"], axis=1)
        if "s_lib_size_raw" not in adata.obs:
            adata.obs["s_lib_size_raw"] = np.sum(adata.layers["spliced"], axis=1)
        if "u_lib_size" not in adata.obs:
            adata.obs["u_lib_size"] = np.log(adata.obs["u_lib_size_raw"] + 1e-6)
        if "s_lib_size" not in adata.obs:
            adata.obs["s_lib_size"] = np.log(adata.obs["s_lib_size_raw"] + 1e-6)
        if "u_lib_size_mean" not in adata.obs:
            adata.obs["u_lib_size_mean"] = np.mean(adata.obs["u_lib_size"])
        if "s_lib_size_mean" not in adata.obs:
            adata.obs["s_lib_size_mean"] = np.mean(adata.obs["s_lib_size"])
        if "u_lib_size_scale" not in adata.obs:
            adata.obs["u_lib_size_scale"] = np.std(adata.obs["u_lib_size"])
        if "s_lib_size_scale" not in adata.obs:
            adata.obs["s_lib_size_scale"] = np.std(adata.obs["s_lib_size"])
        if "ind_x" not in adata.obs:
            adata.obs["ind_x"] = np.arange(adata.shape[0])
    except Exception as e:
        print(f"Error loading pancreas dataset: {e}")
        print("Falling back to synthetic data...")
        adata = create_synthetic_data()

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
    try:
        # Import comparison functions with shape normalization
        from pyrovelocity.validation.comparison import compare_velocities, compare_uncertainties

        # Run validation
        validation_results = run_validation(
            adata=adata,
            max_epochs=args.max_epochs,
            num_samples=args.num_samples,
            use_scalene=args.use_scalene,
            use_legacy=args.use_legacy,
            use_modular=args.use_modular,
            use_jax=args.use_jax,
        )

        # Extract results
        model_results = validation_results["results"]
        comparison = validation_results["comparison"]

        # Check if we have valid results
        valid_implementations = []
        for name, result in model_results.items():
            if "error" not in result:
                valid_implementations.append(name)

        if len(valid_implementations) < 2:
            print(f"Warning: Less than two valid implementations available for comparison.")
            print(f"Valid implementations: {valid_implementations}")
            print("Skipping detailed comparison.")

            # Save error information
            with open(os.path.join(args.output_dir, "validation_errors.txt"), "w") as f:
                f.write("Validation Errors:\n")
                f.write("=================\n\n")
                for name, result in model_results.items():
                    if "error" in result:
                        f.write(f"{name} implementation:\n")
                        f.write(f"  Error: {result['error']}\n")
                        if "traceback" in result:
                            f.write(f"  Traceback:\n{result['traceback']}\n")
                        f.write("\n")

            print(f"Error information saved to {os.path.join(args.output_dir, 'validation_errors.txt')}")
            return

        # Re-run comparisons with shape normalization
        # This ensures that arrays with different shapes can be compared
        print("Comparing implementations with shape normalization...")

        try:
            # Compare velocities with shape normalization
            velocity_comparison = compare_velocities(
                model_results,
                normalize_method=args.normalize_method,
                target_strategy=args.target_strategy
            )

            # Update comparison results
            comparison["velocity_comparison"] = velocity_comparison
        except Exception as e:
            print(f"Error comparing velocities: {e}")
            comparison["velocity_comparison"] = {"error": str(e)}

        try:
            # Compare uncertainties with shape normalization
            uncertainty_comparison = compare_uncertainties(
                model_results,
                normalize_method=args.normalize_method,
                target_strategy=args.target_strategy
            )

            # Update comparison results
            comparison["uncertainty_comparison"] = uncertainty_comparison
        except Exception as e:
            print(f"Error comparing uncertainties: {e}")
            comparison["uncertainty_comparison"] = {"error": str(e)}

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
        try:
            fig = plot_parameter_comparison(comparison["parameter_comparison"])
            fig.savefig(os.path.join(args.output_dir, "parameter_comparison.png"))
            plt.close(fig)
        except Exception as e:
            print(f"Error plotting parameter comparison: {e}")

        # Plot velocity comparison
        # This compares velocity vectors across implementations
        print("Plotting velocity comparison...")
        try:
            fig = plot_velocity_comparison(comparison["velocity_comparison"])
            fig.savefig(os.path.join(args.output_dir, "velocity_comparison.png"))
            plt.close(fig)

            # Also plot velocity vector field if coordinates are available
            if "X_umap" in adata.obsm:
                from pyrovelocity.validation.visualization import plot_velocity_vector_field

                # Get UMAP coordinates
                coordinates = adata.obsm["X_umap"]

                # Plot velocity vector field
                fig = plot_velocity_vector_field(
                    model_results,
                    coordinates,
                    resample_method=args.normalize_method
                )
                fig.savefig(os.path.join(args.output_dir, "velocity_vector_field.png"))
                plt.close(fig)
        except Exception as e:
            print(f"Error plotting velocity comparison: {e}")
            import traceback
            traceback.print_exc()

        # Plot uncertainty comparison
        # This compares uncertainty estimates across implementations
        print("Plotting uncertainty comparison...")
        try:
            fig = plot_uncertainty_comparison(comparison["uncertainty_comparison"])
            fig.savefig(os.path.join(args.output_dir, "uncertainty_comparison.png"))
            plt.close(fig)

            # Also plot uncertainty heatmap
            from pyrovelocity.validation.visualization import plot_uncertainty_heatmap

            # Plot uncertainty heatmap
            fig = plot_uncertainty_heatmap(
                model_results,
                resample_method=args.normalize_method
            )
            fig.savefig(os.path.join(args.output_dir, "uncertainty_heatmap.png"))
            plt.close(fig)
        except Exception as e:
            print(f"Error plotting uncertainty comparison: {e}")
            import traceback
            traceback.print_exc()

        # Plot performance comparison
        # This compares training and inference times across implementations
        print("Plotting performance comparison...")
        try:
            fig = plot_performance_comparison(comparison["performance_comparison"])
            fig.savefig(os.path.join(args.output_dir, "performance_comparison.png"))
            plt.close(fig)
        except Exception as e:
            print(f"Error plotting performance comparison: {e}")

        # Save a summary of the validation
        with open(os.path.join(args.output_dir, "validation_summary.txt"), "w") as f:
            f.write(f"Validation Summary\n")
            f.write(f"=================\n\n")
            f.write(f"Implementations: {', '.join(implementations)}\n")
            f.write(f"Parameters: max_epochs={args.max_epochs}, num_samples={args.num_samples}\n")
            f.write(f"Dataset: {adata.shape[0]} cells, {adata.shape[1]} genes\n")
            f.write(f"Shape normalization: method={args.normalize_method}, target={args.target_strategy}\n\n")

            # Write model information
            f.write(f"Model Information\n")
            f.write(f"----------------\n")
            for model_name in implementations:
                if model_name in model_results:
                    f.write(f"{model_name}:\n")

                    # Write velocity shape
                    if "velocity" in model_results[model_name]:
                        velocity = model_results[model_name]["velocity"]
                        if isinstance(velocity, np.ndarray):
                            f.write(f"  Velocity shape: {velocity.shape}\n")
                        else:
                            f.write(f"  Velocity shape: {type(velocity)}\n")

                    # Write uncertainty shape
                    if "uncertainty" in model_results[model_name]:
                        uncertainty = model_results[model_name]["uncertainty"]
                        if isinstance(uncertainty, np.ndarray):
                            f.write(f"  Uncertainty shape: {uncertainty.shape}\n")
                        else:
                            f.write(f"  Uncertainty shape: {type(uncertainty)}\n")
            f.write("\n")

            # Write performance information
            f.write(f"Performance Summary\n")
            f.write(f"-----------------\n")
            for model_name in implementations:
                if model_name in model_results and "performance" in model_results[model_name]:
                    perf = model_results[model_name]["performance"]
                    f.write(f"{model_name}:\n")
                    f.write(f"  Training time: {perf.get('training_time', 'N/A'):.2f} seconds\n")
                    f.write(f"  Inference time: {perf.get('inference_time', 'N/A'):.2f} seconds\n")
            f.write("\n")

            # Write comparison information
            f.write(f"Comparison Information\n")
            f.write(f"---------------------\n")

            # Write velocity comparison information
            if "velocity_comparison" in comparison:
                f.write(f"Velocity Comparison:\n")
                for comp_key, comp_value in comparison["velocity_comparison"].items():
                    f.write(f"  {comp_key}:\n")
                    for metric, value in comp_value.items():
                        if metric in ["shape1", "shape2", "shape_mismatch", "error", "velocity1_shape", "velocity2_shape"]:
                            f.write(f"    {metric}: {value}\n")
                        elif isinstance(value, (int, float)):
                            f.write(f"    {metric}: {value:.4f}\n")
                        else:
                            f.write(f"    {metric}: {value}\n")
            f.write("\n")

            # Write uncertainty comparison information
            if "uncertainty_comparison" in comparison:
                f.write(f"Uncertainty Comparison:\n")
                for comp_key, comp_value in comparison["uncertainty_comparison"].items():
                    f.write(f"  {comp_key}:\n")
                    for metric, value in comp_value.items():
                        if metric in ["shape1", "shape2", "shape_mismatch", "error", "uncertainty1_shape", "uncertainty2_shape"]:
                            f.write(f"    {metric}: {value}\n")
                        elif isinstance(value, (int, float)):
                            f.write(f"    {metric}: {value:.4f}\n")
                        else:
                            f.write(f"    {metric}: {value}\n")

        print(f"Validation results saved to {args.output_dir}")

    except Exception as e:
        print(f"Error during validation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
