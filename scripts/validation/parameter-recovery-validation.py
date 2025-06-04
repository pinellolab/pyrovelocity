import torch
import pyro
import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc

from pyrovelocity.models.modular.factory import create_piecewise_activation_model
from pyrovelocity.models.modular.validation import (
    create_validation_summary,
    display_validation_table,
    plot_validation_summary
)
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

# # Generate validation datasets for all patterns
print("📊 Generating validation datasets...")

validation_datasets = model.generate_validation_datasets(
    # patterns=['activation', 'decay', 'transient', 'sustained'],
    patterns=['activation'],
    n_genes=5,
    n_cells=200,
    n_parameter_sets=2,  # 2 parameter sets per pattern = 8 total datasets
    seed=42
)

print(f"✅ Generated {len(validation_datasets)} validation datasets")

# Display dataset information
for dataset_key, dataset_info in validation_datasets.items():
    adata = dataset_info['adata']
    pattern = dataset_info['pattern']
    true_params = dataset_info['true_parameters']

    print(f"\n{dataset_key}:")
    print(f"  Data shape: {adata.shape}")
    print(f"  Pattern: {pattern}")
    print(f"  True parameters: {list(true_params.keys())}")
    print(f"  Metadata: {dataset_info.get('metadata', {})}")

print(f"\n🎯 Ready for parameter recovery validation!")

print("🚀 Starting Parameter Recovery Validation...")
print("=" * 60)

# Run comprehensive parameter recovery validation
validation_results = model.validate_parameter_recovery(
    validation_datasets=validation_datasets,
    max_epochs=1000,
    learning_rate=0.01,
    success_threshold=0.9,
    num_posterior_samples=1000,
    adaptive_guide_rank=True,
    create_predictive_plots=True,
    save_plots_path="reports/docs/validation",
    seed=42
)

print("✅ Parameter recovery validation complete!")
print(f"📊 Validated {len(validation_results)} datasets")

# Display training summary
for dataset_key, result in validation_results.items():
    if 'training_result' in result:
        training = result['training_result']
        metrics = result['recovery_metrics']
        print(f"\n{dataset_key} ({result['pattern']} pattern):")
        print(f"  Training: {training['num_epochs']} epochs, ELBO: {training['final_elbo']:.2f}")
        print(f"  Recovery: {metrics['overall_correlation']:.3f} correlation, {metrics['success_rate']:.1%} success")
        print(f"  Plots: {result.get('plot_paths', {})}")

# Display validation results summary
print("📊 PARAMETER RECOVERY VALIDATION RESULTS")
print("=" * 80)

# Get overall summary
summary = create_validation_summary(validation_results)
print(f"\n🎯 Overall Success Rate: {summary['overall_success_rate']:.1%}")
print(f"📊 Mean Correlation: {summary['mean_correlation']:.3f}")
print(f"📉 Mean Error: {summary['mean_error']:.3f}")

# Pattern-wise performance
print(f"\n📈 Pattern-wise Performance:")
for pattern, metrics in summary['by_pattern'].items():
    print(f"  {pattern.upper()}: {metrics['correlation']:.3f} correlation, {metrics['success_rate']:.1%} success")

# Detailed results table
print(f"\n📋 Detailed Results:")
display_validation_table(validation_results)

# Generate comprehensive validation summary plot
plot_validation_summary(
    validation_results,
    save_path='reports/docs/validation/validation_summary.png',
    figsize=(15, 12),
    show_success_threshold=True
)
