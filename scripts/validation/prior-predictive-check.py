import torch
import pyro
import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc

from pyrovelocity.models.modular.factory import create_piecewise_activation_model
from pyrovelocity.plots.predictive_checks import (
    plot_prior_predictive_checks,
)

# Configurable random seed for reproducibility and variability analysis
# Change this value to generate different datasets and see variability in priors
# The seed will be included in the output filename: combined_prior_predictive_checks_{RANDOM_SEED}.pdf
RANDOM_SEED = 42

# Set random seeds for reproducibility
torch.manual_seed(RANDOM_SEED)
pyro.set_rng_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Create the piecewise activation model
model = create_piecewise_activation_model()

print(f"Created model: {model}")
print(f"Model components:")
print(f"  Dynamics: {model.dynamics_model}")
print(f"  Prior: {model.prior_model}")
print(f"  Likelihood: {model.likelihood_model}")
print(f"  Guide: {model.guide_model}")

# Generate prior predictive samples with proper parameter-data correspondence
print(f"Performing prior predictive checks with random seed: {RANDOM_SEED}")
print(f"Output will be saved as: combined_prior_predictive_checks_{RANDOM_SEED}.pdf")

# Set seeds for reproducible prior sampling
torch.manual_seed(RANDOM_SEED)
pyro.set_rng_seed(RANDOM_SEED)

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
sc.pp.neighbors(prior_predictive_adata, n_neighbors=10, random_state=RANDOM_SEED)
sc.tl.umap(prior_predictive_adata, random_state=RANDOM_SEED)
sc.tl.leiden(prior_predictive_adata, random_state=RANDOM_SEED)

# Plot randomly sampled parameters and data
fig_prior = plot_prior_predictive_checks(
    model=model,
    prior_adata=prior_predictive_adata,
    prior_parameters=prior_parameter_samples,
    figsize=(15, 10),
    save_path="reports/docs/prior_predictive",
    figure_name=f"piecewise_activation_prior_checks_{RANDOM_SEED}",
    combine_individual_pdfs=True,
)

plt.suptitle("Prior Predictive Checks: Piecewise Activation Model", fontsize=16)
plt.tight_layout()
plt.show()

# Validate prior parameter ranges (updated for hierarchical temporal parameterization)
print("\nPrior parameter range validation:")
for param_name, samples in prior_parameter_samples.items():
    if param_name.startswith(('R_on', 'tilde_t_on', 'tilde_delta', 't_on', 'delta', 'gamma_star', 'T_M_star')):
        print(f"{param_name}:")
        print(f"  Range: [{samples.min():.3f}, {samples.max():.3f}]")
        print(f"  Mean ± Std: {samples.mean():.3f} ± {samples.std():.3f}")

        # Show mode for LogNormal parameters
        if param_name in ['gamma_star', 'R_on', 'tilde_delta_star']:
            # For LogNormal, mode = exp(μ - σ²)
            if param_name == 'gamma_star':
                mode = torch.exp(torch.tensor(-0.405) - torch.tensor(0.5)**2)
                print(f"  Expected mode (γ*): {mode:.3f}")
            elif param_name == 'R_on':
                mode = torch.exp(torch.tensor(0.693) - torch.tensor(0.35)**2)
                print(f"  Expected mode (R_on): {mode:.3f}")
            elif param_name == 'tilde_delta_star':
                mode = torch.exp(torch.tensor(-0.8) - torch.tensor(0.45)**2)
                print(f"  Expected mode (tilde_δ*): {mode:.3f}")
