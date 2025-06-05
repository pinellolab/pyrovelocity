import torch
import pyro
import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc

from pyrovelocity.models.modular.factory import create_piecewise_activation_model
from pyrovelocity.plots.predictive_checks import (
    plot_prior_predictive_checks,
)

# Set random seeds for reproducibility
torch.manual_seed(42)
pyro.set_rng_seed(42)
np.random.seed(42)

# Create the piecewise activation model
model = create_piecewise_activation_model()

print(f"Created model: {model}")
print(f"Model components:")
print(f"  Dynamics: {model.dynamics_model}")
print(f"  Prior: {model.prior_model}")
print(f"  Likelihood: {model.likelihood_model}")
print(f"  Guide: {model.guide_model}")

# Generate prior predictive samples with proper parameter-data correspondence
print("Performing prior predictive checks...")

# Set seeds for reproducible prior sampling
torch.manual_seed(0)
pyro.set_rng_seed(0)

prior_predictive_adata = model.generate_predictive_samples(
    num_cells=200,
    num_genes=10,
    num_samples=100,  # Generate data from 100 different parameter sets
    return_format="anndata"
)

# Sample parameters from the prior
prior_parameter_samples = model.sample_system_parameters(
    num_samples=1000,
    constrain_to_pattern=False,  # Use full prior ranges
    n_genes=10,
    n_cells=200
)

# Add UMAP and clustering though these will be uninformative in the context of prior predictive checks
sc.pp.neighbors(prior_predictive_adata, n_neighbors=10, random_state=42)
sc.tl.umap(prior_predictive_adata, random_state=42)
sc.tl.leiden(prior_predictive_adata, random_state=42)

# Plot randomly sampled parameters and data
fig_prior = plot_prior_predictive_checks(
    model=model,
    prior_adata=prior_predictive_adata,
    prior_parameters=prior_parameter_samples,
    figsize=(15, 10),
    save_path="reports/docs/prior_predictive",
    combine_individual_pdfs=True,
)

plt.suptitle("Prior Predictive Checks: Piecewise Activation Model", fontsize=16)
plt.tight_layout()
plt.show()

# Validate prior parameter ranges
print("\nPrior parameter range validation:")
for param_name, samples in prior_parameter_samples.items():
    if param_name.startswith(('alpha_off', 'alpha_on', 't_on', 'delta', 'gamma_star')):
        print(f"{param_name}:")
        print(f"  Range: [{samples.min():.3f}, {samples.max():.3f}]")
        print(f"  Mean ± Std: {samples.mean():.3f} ± {samples.std():.3f}")
