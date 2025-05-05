#!/usr/bin/env python
"""
Fixture Validation Script for PyroVelocity.

This script validates that the PyTorch/Pyro modular implementation (specifically
the standard model) produces results equivalent to the legacy implementation
using test fixtures from the PyroVelocity test suite.

The validation focuses on:
1. Parameter estimates (alpha, beta, gamma)
2. Velocity estimates
3. Uncertainty estimates
4. Performance metrics

The script uses test fixtures and provides detailed text-based reports of the
validation results.

Example usage:
    # Run validation with default settings
    python fixture_validation.py

    # Run validation with specific parameters
    python fixture_validation.py --max-epochs 500 --num-samples 50
"""

import os
import sys
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import anndata as ad
from tqdm import tqdm

# Add the src directory to the path to import test fixtures
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from pyrovelocity.tests.conftest import default_sample_data
from pyrovelocity.validation.framework import ValidationRunner
from pyrovelocity.validation.comparison import (
    compare_parameters,
    compare_velocities,
    compare_uncertainties,
    compare_performance,
    statistical_comparison,
)
from pyrovelocity.validation.visualization import (
    plot_parameter_comparison,
    plot_velocity_comparison,
    plot_uncertainty_comparison,
    plot_performance_comparison,
    plot_parameter_distributions,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Fixture Validation Script for PyroVelocity"
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=100,
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
        default="fixture_validation_results",
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
        action="store_true",
        help="Whether to use GPU for training",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    return parser.parse_args()


def prepare_fixture_data(seed=42):
    """Load and prepare test fixture data for validation."""
    print("Loading test fixture data...")
    try:
        # Get the default sample data from the test fixtures
        adata = default_sample_data()
        print(f"Loaded test fixture with {adata.shape[0]} cells and {adata.shape[1]} genes")
        
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
        print(f"Error loading test fixture: {e}")
        print("Creating synthetic data instead...")
        
        # Create synthetic data
        n_cells, n_genes = 50, 20
        np.random.seed(seed)
        u_data = np.random.poisson(5, size=(n_cells, n_genes))
        s_data = np.random.poisson(5, size=(n_cells, n_genes))

        # Create AnnData object
        adata = ad.AnnData(X=s_data)
        adata.layers["spliced"] = s_data
        adata.layers["unspliced"] = u_data
        adata.layers["raw_spliced"] = s_data.copy()
        adata.layers["raw_unspliced"] = u_data.copy()
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


def generate_parameter_report(parameter_comparison, output_dir):
    """Generate a detailed report of parameter comparison results."""
    report_path = os.path.join(output_dir, "parameter_comparison_report.txt")
    
    with open(report_path, "w") as f:
        f.write("Parameter Comparison Report\n")
        f.write("==========================\n\n")
        
        for param, comparisons in parameter_comparison.items():
            f.write(f"Parameter: {param}\n")
            f.write("-" * (len(param) + 11) + "\n")
            
            for comp_name, metrics in comparisons.items():
                f.write(f"  {comp_name}:\n")
                
                for metric_name, value in metrics.items():
                    if isinstance(value, float):
                        f.write(f"    {metric_name}: {value:.6f}\n")
                    else:
                        f.write(f"    {metric_name}: {value}\n")
                
                f.write("\n")
            
            f.write("\n")
    
    print(f"Parameter comparison report saved to {report_path}")
    return report_path


def generate_velocity_report(velocity_comparison, output_dir):
    """Generate a detailed report of velocity comparison results."""
    report_path = os.path.join(output_dir, "velocity_comparison_report.txt")
    
    with open(report_path, "w") as f:
        f.write("Velocity Comparison Report\n")
        f.write("=========================\n\n")
        
        for comp_name, metrics in velocity_comparison.items():
            f.write(f"{comp_name}:\n")
            f.write("-" * len(comp_name) + "\n")
            
            for metric_name, value in metrics.items():
                if isinstance(value, float):
                    f.write(f"  {metric_name}: {value:.6f}\n")
                else:
                    f.write(f"  {metric_name}: {value}\n")
            
            f.write("\n")
    
    print(f"Velocity comparison report saved to {report_path}")
    return report_path


def generate_uncertainty_report(uncertainty_comparison, output_dir):
    """Generate a detailed report of uncertainty comparison results."""
    report_path = os.path.join(output_dir, "uncertainty_comparison_report.txt")
    
    with open(report_path, "w") as f:
        f.write("Uncertainty Comparison Report\n")
        f.write("============================\n\n")
        
        for comp_name, metrics in uncertainty_comparison.items():
            f.write(f"{comp_name}:\n")
            f.write("-" * len(comp_name) + "\n")
            
            for metric_name, value in metrics.items():
                if isinstance(value, float):
                    f.write(f"  {metric_name}: {value:.6f}\n")
                else:
                    f.write(f"  {metric_name}: {value}\n")
            
            f.write("\n")
    
    print(f"Uncertainty comparison report saved to {report_path}")
    return report_path


def generate_performance_report(performance_comparison, output_dir):
    """Generate a detailed report of performance comparison results."""
    report_path = os.path.join(output_dir, "performance_comparison_report.txt")
    
    with open(report_path, "w") as f:
        f.write("Performance Comparison Report\n")
        f.write("============================\n\n")
        
        for comp_name, metrics in performance_comparison.items():
            f.write(f"{comp_name}:\n")
            f.write("-" * len(comp_name) + "\n")
            
            for metric_name, value in metrics.items():
                if isinstance(value, float):
                    f.write(f"  {metric_name}: {value:.6f}\n")
                    
                    # Add interpretation for ratios
                    if "ratio" in metric_name:
                        if value < 0.9:
                            f.write(f"    Interpretation: Second implementation is {(1-value)*100:.1f}% faster\n")
                        elif value > 1.1:
                            f.write(f"    Interpretation: Second implementation is {(value-1)*100:.1f}% slower\n")
                        else:
                            f.write(f"    Interpretation: Performance is similar (within 10%)\n")
                else:
                    f.write(f"  {metric_name}: {value}\n")
            
            f.write("\n")
    
    print(f"Performance comparison report saved to {report_path}")
    return report_path


def generate_summary_report(validation_results, args, output_dir):
    """Generate a summary report of all validation results."""
    report_path = os.path.join(output_dir, "validation_summary_report.txt")
    
    with open(report_path, "w") as f:
        f.write("PyroVelocity Fixture Validation Summary Report\n")
        f.write("===========================================\n\n")
        
        # Write validation parameters
        f.write("Validation Parameters\n")
        f.write("--------------------\n")
        f.write(f"Max epochs: {args.max_epochs}\n")
        f.write(f"Number of posterior samples: {args.num_samples}\n")
        f.write(f"Learning rate: {args.learning_rate}\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"Use GPU: {args.use_gpu}\n")
        f.write(f"Random seed: {args.seed}\n")
        f.write(f"Shape normalization: method={args.normalize_method}, target={args.target_strategy}\n")
        f.write("\n")
        
        # Write model information
        model_results = validation_results["results"]
        f.write("Model Information\n")
        f.write("----------------\n")
        
        for model_name, model_result in model_results.items():
            f.write(f"{model_name}:\n")
            
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
            
            # Write uncertainty shape
            if "uncertainty" in model_result:
                uncertainty = model_result["uncertainty"]
                if hasattr(uncertainty, "shape"):
                    f.write(f"  Uncertainty shape: {uncertainty.shape}\n")
                else:
                    f.write(f"  Uncertainty type: {type(uncertainty)}\n")
            
            f.write("\n")
        
        # Write comparison summary
        comparison = validation_results["comparison"]
        f.write("Comparison Summary\n")
        f.write("-----------------\n")
        
        # Parameter comparison summary
        if "parameter_comparison" in comparison:
            f.write("Parameter Comparison:\n")
            for param, param_comp in comparison["parameter_comparison"].items():
                f.write(f"  {param}:\n")
                for comp_name, metrics in param_comp.items():
                    f.write(f"    {comp_name}:\n")
                    for metric_name, value in metrics.items():
                        if isinstance(value, float):
                            f.write(f"      {metric_name}: {value:.6f}\n")
                        else:
                            f.write(f"      {metric_name}: {value}\n")
            f.write("\n")
        
        # Velocity comparison summary
        if "velocity_comparison" in comparison:
            f.write("Velocity Comparison:\n")
            for comp_name, metrics in comparison["velocity_comparison"].items():
                f.write(f"  {comp_name}:\n")
                for metric_name, value in metrics.items():
                    if isinstance(value, float):
                        f.write(f"    {metric_name}: {value:.6f}\n")
                    else:
                        f.write(f"    {metric_name}: {value}\n")
            f.write("\n")
        
        # Uncertainty comparison summary
        if "uncertainty_comparison" in comparison:
            f.write("Uncertainty Comparison:\n")
            for comp_name, metrics in comparison["uncertainty_comparison"].items():
                f.write(f"  {comp_name}:\n")
                for metric_name, value in metrics.items():
                    if isinstance(value, float):
                        f.write(f"    {metric_name}: {value:.6f}\n")
                    else:
                        f.write(f"    {metric_name}: {value}\n")
            f.write("\n")
        
        # Performance comparison summary
        if "performance_comparison" in comparison:
            f.write("Performance Comparison:\n")
            for comp_name, metrics in comparison["performance_comparison"].items():
                f.write(f"  {comp_name}:\n")
                for metric_name, value in metrics.items():
                    if isinstance(value, float):
                        f.write(f"    {metric_name}: {value:.6f}\n")
                        
                        # Add interpretation for ratios
                        if "ratio" in metric_name:
                            if value < 0.9:
                                f.write(f"      Interpretation: Second implementation is {(1-value)*100:.1f}% faster\n")
                            elif value > 1.1:
                                f.write(f"      Interpretation: Second implementation is {(value-1)*100:.1f}% slower\n")
                            else:
                                f.write(f"      Interpretation: Performance is similar (within 10%)\n")
                    else:
                        f.write(f"    {metric_name}: {value}\n")
            f.write("\n")
        
        # Write validation conclusion
        f.write("Validation Conclusion\n")
        f.write("--------------------\n")
        
        # Check if there are any significant differences in parameters
        param_diff = False
        if "parameter_comparison" in comparison:
            for param, param_comp in comparison["parameter_comparison"].items():
                for comp_name, metrics in param_comp.items():
                    if "correlation" in metrics and metrics["correlation"] < 0.9:
                        param_diff = True
                        break
                if param_diff:
                    break
        
        # Check if there are any significant differences in velocities
        vel_diff = False
        if "velocity_comparison" in comparison:
            for comp_name, metrics in comparison["velocity_comparison"].items():
                if "correlation" in metrics and metrics["correlation"] < 0.9:
                    vel_diff = True
                    break
        
        # Write conclusion
        if param_diff or vel_diff:
            f.write("There are significant differences between the legacy and modular implementations.\n")
            if param_diff:
                f.write("- Parameter estimates show low correlation (< 0.9)\n")
            if vel_diff:
                f.write("- Velocity estimates show low correlation (< 0.9)\n")
            f.write("\nRecommendation: Further investigation is needed to understand these differences.\n")
        else:
            f.write("The legacy and modular implementations produce similar results.\n")
            f.write("- Parameter estimates show high correlation (>= 0.9)\n")
            f.write("- Velocity estimates show high correlation (>= 0.9)\n")
            f.write("\nRecommendation: The modular implementation can be considered a valid replacement for the legacy implementation.\n")
    
    print(f"Summary report saved to {report_path}")
    return report_path


def main():
    """Run validation and save results."""
    # Parse command line arguments
    args = parse_args()

    # Set random seed for reproducibility
    np.random.seed(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load and prepare data
    adata = prepare_fixture_data(seed=args.seed)

    # Initialize ValidationRunner
    print("Initializing ValidationRunner...")
    runner = ValidationRunner(adata)

    # Set up models
    print("Setting up legacy model...")
    runner.setup_legacy_model()
    
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
        
        # Compare implementations
        print("Comparing implementations...")
        comparison = runner.compare_implementations()
        
        # Create validation results dictionary
        validation_results = {
            "results": results,
            "comparison": comparison,
        }
        
        elapsed_time = time.time() - start_time
        print(f"Validation completed in {elapsed_time:.2f} seconds")
        
        # Generate reports
        print("Generating reports...")
        
        # Generate parameter comparison report
        parameter_report = generate_parameter_report(
            comparison["parameter_comparison"], args.output_dir
        )
        
        # Generate velocity comparison report
        velocity_report = generate_velocity_report(
            comparison["velocity_comparison"], args.output_dir
        )
        
        # Generate uncertainty comparison report
        uncertainty_report = generate_uncertainty_report(
            comparison["uncertainty_comparison"], args.output_dir
        )
        
        # Generate performance comparison report
        performance_report = generate_performance_report(
            comparison["performance_comparison"], args.output_dir
        )
        
        # Generate summary report
        summary_report = generate_summary_report(
            validation_results, args, args.output_dir
        )
        
        # Generate visualizations
        print("Generating visualizations...")
        
        # Plot parameter comparison
        try:
            fig = plot_parameter_comparison(comparison["parameter_comparison"])
            fig.savefig(os.path.join(args.output_dir, "parameter_comparison.png"))
            plt.close(fig)
        except Exception as e:
            print(f"Error plotting parameter comparison: {e}")
        
        # Plot velocity comparison
        try:
            fig = plot_velocity_comparison(comparison["velocity_comparison"])
            fig.savefig(os.path.join(args.output_dir, "velocity_comparison.png"))
            plt.close(fig)
        except Exception as e:
            print(f"Error plotting velocity comparison: {e}")
        
        # Plot uncertainty comparison
        try:
            fig = plot_uncertainty_comparison(comparison["uncertainty_comparison"])
            fig.savefig(os.path.join(args.output_dir, "uncertainty_comparison.png"))
            plt.close(fig)
        except Exception as e:
            print(f"Error plotting uncertainty comparison: {e}")
        
        # Plot performance comparison
        try:
            fig = plot_performance_comparison(comparison["performance_comparison"])
            fig.savefig(os.path.join(args.output_dir, "performance_comparison.png"))
            plt.close(fig)
        except Exception as e:
            print(f"Error plotting performance comparison: {e}")
        
        # Plot parameter distributions
        try:
            for param in ["alpha", "beta", "gamma"]:
                fig = plot_parameter_distributions(results, param)
                fig.savefig(os.path.join(args.output_dir, f"{param}_distribution.png"))
                plt.close(fig)
        except Exception as e:
            print(f"Error plotting parameter distributions: {e}")
        
        print(f"Validation results saved to {args.output_dir}")
        
    except Exception as e:
        print(f"Error during validation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
