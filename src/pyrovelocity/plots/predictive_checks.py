"""
Predictive check plotting functions for PyroVelocity models.

This module provides flexible plotting functions for both prior and posterior
predictive checks, designed to validate model behavior and biological plausibility.
"""

from typing import Any, Dict, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from anndata import AnnData
from beartype import beartype

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


@beartype
def plot_prior_predictive_checks(
    model: Any,
    prior_adata: AnnData,
    prior_parameters: Dict[str, torch.Tensor],
    figsize: Tuple[int, int] = (15, 10),
    check_type: str = "prior"
) -> plt.Figure:
    """
    Generate comprehensive predictive check plots for PyroVelocity models.
    
    This function creates a multi-panel figure showing parameter distributions,
    expression data validation, temporal dynamics, and biological plausibility checks.
    Can be used for both prior and posterior predictive checks.
    
    Args:
        model: PyroVelocity model instance
        prior_adata: AnnData object with predictive samples
        prior_parameters: Dictionary of parameter samples
        figsize: Figure size (width, height)
        check_type: Type of check ("prior" or "posterior")
    
    Returns:
        matplotlib Figure object
    """
    fig = plt.figure(figsize=figsize)
    
    # Create subplot grid: 3 rows × 4 columns
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # Row 1: Parameter Distribution Plots
    ax1 = fig.add_subplot(gs[0, 0])
    _plot_parameter_marginals(prior_parameters, ax1, check_type)
    
    ax2 = fig.add_subplot(gs[0, 1])
    _plot_parameter_correlations(prior_parameters, ax2, check_type)
    
    ax3 = fig.add_subplot(gs[0, 2])
    _plot_fold_change_distribution(prior_parameters, ax3, check_type)
    
    ax4 = fig.add_subplot(gs[0, 3])
    _plot_activation_timing(prior_parameters, ax4, check_type)
    
    # Row 2: Expression Data Validation
    ax5 = fig.add_subplot(gs[1, 0])
    _plot_count_distributions(prior_adata, ax5, check_type)
    
    ax6 = fig.add_subplot(gs[1, 1])
    _plot_expression_relationships(prior_adata, ax6, check_type)
    
    ax7 = fig.add_subplot(gs[1, 2])
    _plot_library_sizes(prior_adata, ax7, check_type)
    
    ax8 = fig.add_subplot(gs[1, 3])
    _plot_expression_ranges(prior_adata, ax8, check_type)
    
    # Row 3: Temporal Dynamics and Biological Validation
    ax9 = fig.add_subplot(gs[2, 0])
    _plot_phase_portrait(prior_adata, ax9, check_type)
    
    ax10 = fig.add_subplot(gs[2, 1])
    _plot_velocity_magnitudes(prior_adata, ax10, check_type)
    
    ax11 = fig.add_subplot(gs[2, 2])
    _plot_pattern_proportions(prior_adata, prior_parameters, ax11, check_type)
    
    ax12 = fig.add_subplot(gs[2, 3])
    _plot_correlation_structure(prior_adata, ax12, check_type)
    
    return fig


def _plot_parameter_marginals(
    parameters: Dict[str, torch.Tensor], 
    ax: plt.Axes, 
    check_type: str
) -> None:
    """Plot marginal distributions of key parameters."""
    # Focus on piecewise activation parameters
    key_params = ['alpha_off', 'alpha_on', 't_on_star', 'delta_star']
    
    colors = sns.color_palette("husl", len(key_params))
    
    for i, param_name in enumerate(key_params):
        if param_name in parameters:
            values = parameters[param_name].flatten().numpy()
            ax.hist(values, bins=30, alpha=0.6, color=colors[i], 
                   label=f'${param_name.replace("_", "^*_")}$', density=True)
    
    ax.set_xlabel('Parameter Value')
    ax.set_ylabel('Density')
    ax.set_title(f'{check_type.title()} Parameter Marginals')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def _plot_parameter_correlations(
    parameters: Dict[str, torch.Tensor], 
    ax: plt.Axes, 
    check_type: str
) -> None:
    """Plot parameter correlation matrix."""
    # Extract key parameters for correlation analysis
    key_params = ['alpha_off', 'alpha_on', 't_on_star', 'delta_star']
    
    param_data = {}
    for param_name in key_params:
        if param_name in parameters:
            param_data[param_name.replace('_', '*_')] = parameters[param_name].flatten().numpy()
    
    if len(param_data) > 1:
        df = pd.DataFrame(param_data)
        corr_matrix = df.corr()
        
        sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0,
                   square=True, ax=ax, cbar_kws={'shrink': 0.8})
        ax.set_title(f'{check_type.title()} Parameter Correlations')
    else:
        ax.text(0.5, 0.5, 'Insufficient parameters\nfor correlation analysis', 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'{check_type.title()} Parameter Correlations')


def _plot_fold_change_distribution(
    parameters: Dict[str, torch.Tensor], 
    ax: plt.Axes, 
    check_type: str
) -> None:
    """Plot fold-change distribution."""
    if 'alpha_off' in parameters and 'alpha_on' in parameters:
        alpha_off = parameters['alpha_off'].flatten()
        alpha_on = parameters['alpha_on'].flatten()
        fold_change = (alpha_on / alpha_off).numpy()
        
        ax.hist(fold_change, bins=50, alpha=0.7, color='skyblue', density=True)
        ax.axvline(fold_change.mean(), color='red', linestyle='--', 
                  label=f'Mean: {fold_change.mean():.1f}')
        ax.axvline(3.3, color='orange', linestyle=':', label='Min threshold: 3.3')
        ax.axvline(7.5, color='green', linestyle=':', label='Activation threshold: 7.5')
        
        ax.set_xlabel('Fold-change (α*_on / α*_off)')
        ax.set_ylabel('Density')
        ax.set_title(f'{check_type.title()} Fold-change Distribution')
        ax.legend(fontsize=8)
        ax.set_xlim(0, min(100, fold_change.max()))
    else:
        ax.text(0.5, 0.5, 'Alpha parameters\nnot available', 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'{check_type.title()} Fold-change Distribution')
    
    ax.grid(True, alpha=0.3)


def _plot_activation_timing(
    parameters: Dict[str, torch.Tensor], 
    ax: plt.Axes, 
    check_type: str
) -> None:
    """Plot activation timing and duration distributions."""
    if 't_on_star' in parameters and 'delta_star' in parameters:
        t_on = parameters['t_on_star'].flatten().numpy()
        delta = parameters['delta_star'].flatten().numpy()
        
        ax.scatter(t_on, delta, alpha=0.6, s=20, color='purple')
        ax.set_xlabel('Activation Onset (t*_on)')
        ax.set_ylabel('Activation Duration (δ*)')
        ax.set_title(f'{check_type.title()} Activation Timing')
        
        # Add pattern boundaries
        ax.axhline(0.35, color='red', linestyle='--', alpha=0.7, 
                  label='Transient/Sustained boundary')
        ax.axvline(0.3, color='orange', linestyle='--', alpha=0.7,
                  label='Early/Late activation')
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, 'Timing parameters\nnot available', 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'{check_type.title()} Activation Timing')
    
    ax.grid(True, alpha=0.3)


def _plot_count_distributions(adata: AnnData, ax: plt.Axes, check_type: str) -> None:
    """Plot unspliced and spliced count distributions."""
    if 'unspliced' in adata.layers and 'spliced' in adata.layers:
        unspliced = adata.layers['unspliced'].flatten()
        spliced = adata.layers['spliced'].flatten()
        
        # Remove zeros for log scale
        unspliced_nz = unspliced[unspliced > 0]
        spliced_nz = spliced[spliced > 0]
        
        ax.hist(np.log1p(unspliced_nz), bins=50, alpha=0.6, 
               label='Unspliced', color='red', density=True)
        ax.hist(np.log1p(spliced_nz), bins=50, alpha=0.6, 
               label='Spliced', color='blue', density=True)
        
        ax.set_xlabel('log(count + 1)')
        ax.set_ylabel('Density')
        ax.set_title(f'{check_type.title()} Count Distributions')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'Count data\nnot available', 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'{check_type.title()} Count Distributions')
    
    ax.grid(True, alpha=0.3)


def _plot_expression_relationships(adata: AnnData, ax: plt.Axes, check_type: str) -> None:
    """Plot unspliced vs spliced expression relationships."""
    if 'unspliced' in adata.layers and 'spliced' in adata.layers:
        # Sample subset for visualization
        n_sample = min(1000, adata.n_obs * adata.n_vars)
        unspliced = adata.layers['unspliced'].flatten()
        spliced = adata.layers['spliced'].flatten()
        
        # Random sample for plotting
        idx = np.random.choice(len(unspliced), n_sample, replace=False)
        u_sample = unspliced[idx]
        s_sample = spliced[idx]
        
        ax.scatter(np.log1p(s_sample), np.log1p(u_sample), 
                  alpha=0.5, s=1, color='purple')
        ax.set_xlabel('log(Spliced + 1)')
        ax.set_ylabel('log(Unspliced + 1)')
        ax.set_title(f'{check_type.title()} U vs S Relationship')
        
        # Add diagonal reference
        max_val = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='U = S')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'Expression data\nnot available', 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'{check_type.title()} U vs S Relationship')
    
    ax.grid(True, alpha=0.3)


def _plot_library_sizes(adata: AnnData, ax: plt.Axes, check_type: str) -> None:
    """Plot library size distributions."""
    if 'unspliced' in adata.layers and 'spliced' in adata.layers:
        total_counts = adata.layers['unspliced'].sum(axis=1) + adata.layers['spliced'].sum(axis=1)

        ax.hist(total_counts, bins=50, alpha=0.7, color='green', density=True)
        ax.axvline(total_counts.mean(), color='red', linestyle='--',
                  label=f'Mean: {total_counts.mean():.0f}')

        ax.set_xlabel('Total Counts per Cell')
        ax.set_ylabel('Density')
        ax.set_title(f'{check_type.title()} Library Sizes')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'Count data\nnot available',
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'{check_type.title()} Library Sizes')

    ax.grid(True, alpha=0.3)


def _plot_expression_ranges(adata: AnnData, ax: plt.Axes, check_type: str) -> None:
    """Plot expression range validation."""
    if 'unspliced' in adata.layers and 'spliced' in adata.layers:
        # Calculate expression ranges per gene
        u_ranges = np.ptp(adata.layers['unspliced'], axis=0)  # peak-to-peak
        s_ranges = np.ptp(adata.layers['spliced'], axis=0)

        ax.scatter(s_ranges, u_ranges, alpha=0.7, s=50, color='orange')
        ax.set_xlabel('Spliced Expression Range')
        ax.set_ylabel('Unspliced Expression Range')
        ax.set_title(f'{check_type.title()} Expression Ranges')

        # Add diagonal reference
        max_val = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='U range = S range')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'Expression data\nnot available',
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'{check_type.title()} Expression Ranges')

    ax.grid(True, alpha=0.3)


def _plot_phase_portrait(adata: AnnData, ax: plt.Axes, check_type: str) -> None:
    """Plot phase portrait (unspliced vs spliced trajectories)."""
    if 'unspliced' in adata.layers and 'spliced' in adata.layers:
        # Sample genes for visualization
        n_genes_plot = min(3, adata.n_vars)
        gene_indices = np.random.choice(adata.n_vars, n_genes_plot, replace=False)

        colors = sns.color_palette("husl", n_genes_plot)

        for i, gene_idx in enumerate(gene_indices):
            u_gene = adata.layers['unspliced'][:, gene_idx]
            s_gene = adata.layers['spliced'][:, gene_idx]

            ax.scatter(s_gene, u_gene, alpha=0.6, s=20, color=colors[i],
                      label=f'Gene {gene_idx}')

        ax.set_xlabel('Spliced Expression')
        ax.set_ylabel('Unspliced Expression')
        ax.set_title(f'{check_type.title()} Phase Portrait')
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, 'Expression data\nnot available',
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'{check_type.title()} Phase Portrait')

    ax.grid(True, alpha=0.3)


def _plot_velocity_magnitudes(adata: AnnData, ax: plt.Axes, check_type: str) -> None:
    """Plot RNA velocity magnitude distributions."""
    # Check if velocity has been computed
    velocity_layers = [key for key in adata.layers.keys() if 'velocity' in key.lower()]

    if velocity_layers:
        # Use the first velocity layer found
        velocity = adata.layers[velocity_layers[0]]
        velocity_magnitudes = np.linalg.norm(velocity, axis=1)

        ax.hist(velocity_magnitudes, bins=50, alpha=0.7, color='teal', density=True)
        ax.axvline(velocity_magnitudes.mean(), color='red', linestyle='--',
                  label=f'Mean: {velocity_magnitudes.mean():.3f}')

        ax.set_xlabel('Velocity Magnitude')
        ax.set_ylabel('Density')
        ax.set_title(f'{check_type.title()} Velocity Magnitudes')
        ax.legend()
    else:
        # Compute simple velocity approximation from U/S ratio changes
        if 'unspliced' in adata.layers and 'spliced' in adata.layers:
            u = adata.layers['unspliced']
            s = adata.layers['spliced']

            # Simple velocity approximation: du/dt ≈ α - γu (assuming steady-state splicing)
            velocity_approx = np.mean(u - s, axis=1)  # Simplified

            ax.hist(velocity_approx, bins=50, alpha=0.7, color='teal', density=True)
            ax.axvline(velocity_approx.mean(), color='red', linestyle='--',
                      label=f'Mean: {velocity_approx.mean():.3f}')

            ax.set_xlabel('Velocity Approximation')
            ax.set_ylabel('Density')
            ax.set_title(f'{check_type.title()} Velocity Approximation')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'Velocity data\nnot available',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{check_type.title()} Velocity Magnitudes')

    ax.grid(True, alpha=0.3)


def _plot_pattern_proportions(
    adata: AnnData,
    parameters: Dict[str, torch.Tensor],
    ax: plt.Axes,
    check_type: str
) -> None:
    """Plot proportion of expression patterns."""
    # Try to get pattern information from AnnData
    if 'pattern' in adata.uns:
        pattern = adata.uns['pattern']
        pattern_counts = {pattern: 1}
    else:
        # Classify patterns based on parameters if available
        pattern_counts = _classify_patterns_from_parameters(parameters)

    if pattern_counts:
        patterns = list(pattern_counts.keys())
        counts = list(pattern_counts.values())
        colors = sns.color_palette("Set2", len(patterns))

        wedges, texts, autotexts = ax.pie(counts, labels=patterns, colors=colors,
                                         autopct='%1.1f%%', startangle=90)
        ax.set_title(f'{check_type.title()} Pattern Proportions')
    else:
        ax.text(0.5, 0.5, 'Pattern information\nnot available',
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'{check_type.title()} Pattern Proportions')


def _plot_correlation_structure(adata: AnnData, ax: plt.Axes, check_type: str) -> None:
    """Plot gene-gene correlation structure."""
    if 'spliced' in adata.layers:
        # Calculate gene-gene correlations
        expr_data = adata.layers['spliced']

        # Sample genes if too many
        n_genes_plot = min(10, adata.n_vars)
        if adata.n_vars > n_genes_plot:
            gene_indices = np.random.choice(adata.n_vars, n_genes_plot, replace=False)
            expr_subset = expr_data[:, gene_indices]
        else:
            expr_subset = expr_data

        # Calculate correlation matrix
        corr_matrix = np.corrcoef(expr_subset.T)

        sns.heatmap(corr_matrix, annot=False, cmap='RdBu_r', center=0,
                   square=True, ax=ax, cbar_kws={'shrink': 0.8})
        ax.set_title(f'{check_type.title()} Gene Correlations')
        ax.set_xlabel('Gene Index')
        ax.set_ylabel('Gene Index')
    else:
        ax.text(0.5, 0.5, 'Expression data\nnot available',
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'{check_type.title()} Gene Correlations')


def _classify_patterns_from_parameters(parameters: Dict[str, torch.Tensor]) -> Dict[str, int]:
    """Classify expression patterns from parameter samples."""
    if not all(key in parameters for key in ['alpha_off', 'alpha_on', 't_on_star', 'delta_star']):
        return {}

    alpha_off = parameters['alpha_off'].flatten()
    alpha_on = parameters['alpha_on'].flatten()
    t_on_star = parameters['t_on_star'].flatten()
    delta_star = parameters['delta_star'].flatten()
    fold_change = alpha_on / alpha_off

    pattern_counts = {'activation': 0, 'decay': 0, 'transient': 0, 'sustained': 0, 'unknown': 0}

    for i in range(len(alpha_off)):
        # Apply same classification logic as in model
        if (alpha_off[i] < 0.15 and alpha_on[i] > 1.5 and
            t_on_star[i] < 0.4 and delta_star[i] > 0.4 and fold_change[i] > 7.5):
            pattern_counts['activation'] += 1
        elif alpha_off[i] > 0.08 and t_on_star[i] > 0.35:
            pattern_counts['decay'] += 1
        elif (alpha_off[i] < 0.3 and alpha_on[i] > 1.0 and
              t_on_star[i] < 0.5 and delta_star[i] < 0.35 and fold_change[i] > 3.3):
            pattern_counts['transient'] += 1
        elif (alpha_off[i] < 0.3 and alpha_on[i] > 1.0 and
              t_on_star[i] < 0.3 and delta_star[i] > 0.35 and fold_change[i] > 3.3):
            pattern_counts['sustained'] += 1
        else:
            pattern_counts['unknown'] += 1

    # Remove zero counts
    return {k: v for k, v in pattern_counts.items() if v > 0}


# Alias for posterior predictive checks (same function, different check_type)
@beartype
def plot_posterior_predictive_checks(
    model: Any,
    posterior_adata: AnnData,
    posterior_parameters: Dict[str, torch.Tensor],
    figsize: Tuple[int, int] = (15, 10)
) -> plt.Figure:
    """
    Generate posterior predictive check plots.

    This is an alias for plot_prior_predictive_checks with check_type="posterior".

    Args:
        model: PyroVelocity model instance
        posterior_adata: AnnData object with posterior predictive samples
        posterior_parameters: Dictionary of posterior parameter samples
        figsize: Figure size (width, height)

    Returns:
        matplotlib Figure object
    """
    return plot_prior_predictive_checks(
        model=model,
        prior_adata=posterior_adata,
        prior_parameters=posterior_parameters,
        figsize=figsize,
        check_type="posterior"
    )
