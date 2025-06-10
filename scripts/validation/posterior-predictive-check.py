import torch
import pyro
import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc

from pyrovelocity.models.modular.factory import create_piecewise_activation_model
from pyrovelocity.plots.predictive_checks import (
    plot_posterior_predictive_checks,
)
from pyrovelocity.utils import print_anndata


RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)
pyro.set_rng_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

print("🚀 Posterior Predictive Check Workflow")
print("=" * 50)

# Step 1: Create model
model = create_piecewise_activation_model()

print(f"✅ Created model: {model}")
print(f"Model components:")
print(f"  Dynamics: {model.dynamics_model}")
print(f"  Prior: {model.prior_model}")
print(f"  Likelihood: {model.likelihood_model}")
print(f"  Guide: {model.guide_model}")

# Step 2: Generate prior predictive data with known true parameters
print(f"\n📊 Generating prior predictive data (seed: {RANDOM_SEED})...")
prior_predictive_adata = model.generate_predictive_samples(
    num_cells=200,
    num_genes=100,
    num_samples=1,   # Single dataset with parameters stored in AnnData
    return_format="anndata"
)

print("✅ Prior predictive data generated:")
print_anndata(prior_predictive_adata)

# Verify true parameters are stored in AnnData
print("\n🔍 Verifying true parameters in AnnData:")
if "true_parameters" in prior_predictive_adata.uns:
    print(f"  Found {len(prior_predictive_adata.uns['true_parameters'])} parameters in true_parameters")
    for key in sorted(prior_predictive_adata.uns['true_parameters'].keys())[:5]:  # Show first 5
        print(f"    {key}")
    if len(prior_predictive_adata.uns['true_parameters']) > 5:
        print(f"    ... and {len(prior_predictive_adata.uns['true_parameters']) - 5} more")
else:
    print("  ❌ No parameters found in adata.uns['true_parameters']")
    raise ValueError("Prior predictive data must contain true_parameters for training")

# Step 3: Train model on prior predictive data (clean API)
print(f"\n🎯 Training model on prior predictive data...")
print("  Using clean train() API without hard-coded assumptions")

# Clean training API - no repetitive logic from validate_parameter_recovery
trained_model = model.train(
    adata=prior_predictive_adata,
    max_epochs=1000,
    learning_rate=0.01,
    early_stopping=True,
    early_stopping_patience=10,
    use_gpu="auto",
    seed=RANDOM_SEED
)

print("✅ Model training completed")

# Step 4: Generate posterior samples from trained model
print(f"\n🔬 Generating posterior samples from trained model...")

# Generate posterior samples using the trained model
posterior_samples = trained_model.generate_posterior_samples(
    adata=prior_predictive_adata,
    num_samples=1000,  # Generate sufficient samples for good posterior approximation
    seed=RANDOM_SEED
)

print(f"✅ Generated {len(posterior_samples)} types of posterior parameters")
for key in sorted(posterior_samples.keys())[:5]:  # Show first 5
    param_shape = posterior_samples[key].shape if hasattr(posterior_samples[key], 'shape') else len(posterior_samples[key])
    print(f"    {key}: {param_shape}")
if len(posterior_samples) > 5:
    print(f"    ... and {len(posterior_samples) - 5} more")

# Step 5: Generate posterior predictive data using posterior samples
print(f"\n📊 Generating posterior predictive data...")
print("  Using posterior samples to generate synthetic data")

# Convert numpy arrays back to tensors for generate_predictive_samples
posterior_samples_tensors = {}
for key, value in posterior_samples.items():
    if isinstance(value, np.ndarray):
        posterior_samples_tensors[key] = torch.tensor(value)
    else:
        posterior_samples_tensors[key] = value

# Generate posterior predictive data
posterior_predictive_adata = trained_model.generate_predictive_samples(
    num_cells=prior_predictive_adata.n_obs,
    num_genes=prior_predictive_adata.n_vars,
    samples=posterior_samples_tensors,  # Use posterior samples
    return_format="anndata"
)

print("✅ Posterior predictive data generated:")
print_anndata(posterior_predictive_adata)

# Store fit parameters in the posterior predictive AnnData for plotting
print("\n� Storing fit parameters in AnnData...")
fit_parameters = {}
for key, value in posterior_samples.items():
    # Take mean of posterior samples for plotting
    if isinstance(value, np.ndarray) and value.ndim > 1:
        fit_parameters[key] = value.mean(axis=0)  # Average over samples
    else:
        fit_parameters[key] = value

# Store in AnnData
posterior_predictive_adata.uns["fit_parameters"] = fit_parameters

print(f"✅ Stored {len(fit_parameters)} fit parameters in AnnData")

# Extract parameters for plotting functions
posterior_parameter_samples = {}
for key, value in fit_parameters.items():
    posterior_parameter_samples[key] = torch.tensor(value) if not isinstance(value, torch.Tensor) else value

# Add UMAP and clustering for visualization
print(f"\n🗺️ Computing UMAP and clustering for visualization...")
sc.pp.neighbors(posterior_predictive_adata, n_neighbors=10, random_state=RANDOM_SEED)
sc.tl.umap(posterior_predictive_adata, random_state=RANDOM_SEED)
sc.tl.leiden(posterior_predictive_adata, random_state=RANDOM_SEED)

# Step 5: Generate posterior predictive check plots (clean API)
print(f"\n🎨 Creating posterior predictive check plots...")
print("  Using plot_posterior_predictive_checks() with appropriate arguments")

fig_posterior = plot_posterior_predictive_checks(
    model=trained_model,
    posterior_adata=posterior_predictive_adata,
    posterior_parameters=posterior_parameter_samples,
    figsize=(7.5, 5.0),
    save_path="reports/docs/posterior_predictive",
    figure_name=f"piecewise_activation_posterior_checks_{RANDOM_SEED}",
    combine_individual_pdfs=True,
)

plt.suptitle("Posterior Predictive Checks: Piecewise Activation Model", fontsize=16)
plt.tight_layout()
plt.show()

print(f"\n✅ Posterior predictive check workflow completed!")
print(f"📁 Plots saved to: reports/docs/posterior_predictive/")
print(f"🎯 Random seed used: {RANDOM_SEED}")
