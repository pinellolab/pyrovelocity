#!/usr/bin/env python
"""
Basic validation script for PyroVelocity.

This script demonstrates how to use the validation framework to compare
different implementations of PyroVelocity.
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
    """Run validation and save results."""
    # Parse command line arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print("Loading data...")
    adata = pancreas()
    
    # Subset data for faster validation
    print("Subsetting data...")
    adata = adata[:100, :100].copy()
    
    # Run validation
    print("Running validation...")
    validation_results = run_validation(
        adata=adata,
        max_epochs=args.max_epochs,
        num_samples=args.num_samples,
        use_scalene=args.use_scalene,
        use_legacy=args.use_legacy,
        use_modular=args.use_modular,
        use_jax=args.use_jax,
    )
    
    # Extract results and comparison
    results = validation_results["results"]
    comparison = validation_results["comparison"]
    
    # Plot parameter comparison
    print("Plotting parameter comparison...")
    fig = plot_parameter_comparison(comparison["parameter_comparison"])
    fig.savefig(os.path.join(args.output_dir, "parameter_comparison.png"))
    plt.close(fig)
    
    # Plot velocity comparison
    print("Plotting velocity comparison...")
    fig = plot_velocity_comparison(comparison["velocity_comparison"])
    fig.savefig(os.path.join(args.output_dir, "velocity_comparison.png"))
    plt.close(fig)
    
    # Plot uncertainty comparison
    print("Plotting uncertainty comparison...")
    fig = plot_uncertainty_comparison(comparison["uncertainty_comparison"])
    fig.savefig(os.path.join(args.output_dir, "uncertainty_comparison.png"))
    plt.close(fig)
    
    # Plot performance comparison
    print("Plotting performance comparison...")
    fig = plot_performance_comparison(comparison["performance_comparison"])
    fig.savefig(os.path.join(args.output_dir, "performance_comparison.png"))
    plt.close(fig)
    
    print(f"Validation results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
