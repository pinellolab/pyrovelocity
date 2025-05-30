#!/usr/bin/env python3
"""
Example usage of PyroVelocity predictive check plotting with automatic saving.

This script demonstrates how to use the plot_prior_predictive_checks and
plot_posterior_predictive_checks functions with automatic PNG/PDF saving.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyro
import torch

# PyroVelocity imports
from pyrovelocity.models.modular.factory import (
    create_piecewise_activation_model,
)
from pyrovelocity.plots.predictive_checks import (
    plot_posterior_predictive_checks,
    plot_prior_predictive_checks,
)


def main():
    """Demonstrate predictive check plotting with automatic saving."""
    print("PyroVelocity Predictive Checks Example")
    print("=" * 40)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    pyro.set_rng_seed(42)
    np.random.seed(42)
    
    # Create the piecewise activation model
    model = create_piecewise_activation_model()
    print(f"✅ Created model: {model.name}")
    
    # Generate prior predictive samples with proper parameter-data correspondence
    print("\n📊 Generating prior predictive samples...")

    # STEP 1: Sample parameters from the prior first
    prior_parameters = model.sample_system_parameters(
        num_samples=1000,
        constrain_to_pattern=False,
        n_genes=10,
        n_cells=200
    )

    # STEP 2: Generate data using a specific parameter sample
    # Take first parameter sample for data generation
    single_param_sample = {key: value[:1] for key, value in prior_parameters.items()}

    prior_adata = model.generate_predictive_samples(
        num_cells=200,
        num_genes=10,
        samples=single_param_sample,  # Use specific parameters
        return_format="anndata"
    )
    
    print(f"✅ Generated prior samples: {prior_adata.shape}")
    print(f"✅ Generated prior parameters: {len(prior_parameters)} types")
    
    # Create prior predictive checks with automatic saving
    print("\n🎨 Creating prior predictive check plots...")
    fig_prior = plot_prior_predictive_checks(
        model=model,
        prior_adata=prior_adata,
        prior_parameters=prior_parameters,
        figsize=(15, 10),
        save_path="reports/docs/prior_predictive",
        figure_name="piecewise_activation_prior_checks"
    )
    
    plt.suptitle("Prior Predictive Checks: Piecewise Activation Model", fontsize=16)
    plt.tight_layout()
    
    # For demonstration, also show how to use posterior predictive checks
    print("\n🔬 Demonstrating posterior predictive checks...")
    
    # For this example, we'll use the same data as "posterior" samples
    # In practice, these would come from actual model fitting
    fig_posterior = plot_posterior_predictive_checks(
        model=model,
        posterior_adata=prior_adata,  # Using prior data as example
        posterior_parameters=prior_parameters,  # Using prior params as example
        figsize=(15, 10),
        save_path="reports/docs/posterior_predictive",
        figure_name="piecewise_activation_posterior_checks"
    )
    
    plt.suptitle("Posterior Predictive Checks: Piecewise Activation Model", fontsize=16)
    plt.tight_layout()
    
    # Show parameter summary
    print("\n📈 Parameter Summary:")
    for param_name, samples in prior_parameters.items():
        if param_name.startswith(('alpha_off', 'alpha_on', 't_on', 'delta', 'gamma_star')):
            print(f"  {param_name}:")
            print(f"    Range: [{samples.min():.3f}, {samples.max():.3f}]")
            print(f"    Mean ± Std: {samples.mean():.3f} ± {samples.std():.3f}")
    
    print("\n✅ Example complete!")
    print("📁 Check the following directories for saved plots:")
    print("   - reports/docs/prior_predictive/")
    print("   - reports/docs/posterior_predictive/")
    print("📄 Each directory contains both PNG and PDF versions")
    
    # Clean up
    plt.close('all')

if __name__ == "__main__":
    main()
