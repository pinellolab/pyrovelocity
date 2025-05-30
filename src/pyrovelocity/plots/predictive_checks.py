"""
Predictive check plotting functions for PyroVelocity models.

This module provides flexible plotting functions for both prior and posterior
predictive checks, designed to validate model behavior and biological plausibility.
The module is organized into modular plotting functions that can be used individually
or combined into comprehensive overview plots.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from anndata import AnnData
from beartype import beartype

from pyrovelocity.styles import configure_matplotlib_style

# Try to import UMAP, fall back gracefully if not available
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

configure_matplotlib_style()


def _format_parameter_name(param_name: str) -> str:
    """
    Format parameter names for LaTeX rendering in plots.

    Args:
        param_name: Raw parameter name (e.g., 'alpha_off', 't_on_star')

    Returns:
        LaTeX-formatted parameter name
    """
    # Greek letter replacements
    greek_replacements = {
        'alpha': r'\alpha',
        'beta': r'\beta',
        'gamma': r'\gamma',
        'delta': r'\delta',
        'epsilon': r'\epsilon',
        'theta': r'\theta',
        'lambda': r'\lambda',
        'mu': r'\mu',
        'sigma': r'\sigma',
        'tau': r'\tau',
        'phi': r'\phi',
        'psi': r'\psi',
        'omega': r'\omega'
    }

    # Subscript replacements for common suffixes
    subscript_replacements = {
        '_loc': r'_{loc}',
        '_scale': r'_{scl}',
        '_on': r'_{on}',
        '_off': r'_{off}',
        '_0i': r'_{0i}',
        '_star': r'^*',
        '_max': r'_{max}',
        '_min': r'_{min}',
        '_rate': r'_{rate}',
        '_conc': r'_{conc}',
        '_shape': r'_{shape}',
        '_mean': r'_{mean}',
        '_std': r'_{std}',
        '_var': r'_{var}',
        '_j': r'_j'
    }

    # Start with the original name
    formatted = param_name

    # Handle special cases first
    special_cases = {
        'T_M_star': r'T^*_M',
        't_star': r't^*',
        'lambda_j': r'\lambda_j',
        'U_0i': r'U_{0i}'
    }

    if param_name in special_cases:
        return f'${special_cases[param_name]}$'

    # Apply subscript/superscript replacements
    for suffix, latex_suffix in subscript_replacements.items():
        formatted = formatted.replace(suffix, latex_suffix)

    # Replace Greek letters at the beginning
    for greek, latex in greek_replacements.items():
        if formatted.startswith(greek):
            formatted = formatted.replace(greek, latex, 1)
            break

    # Wrap in math mode
    return f'${formatted}$'


def _save_figure(
    fig: plt.Figure,
    save_path: str,
    figure_name: str,
    formats: List[str] = ["png", "pdf"]
) -> None:
    """
    Save figure in multiple formats with consistent naming.

    Args:
        fig: matplotlib Figure object
        save_path: Directory path to save figures
        figure_name: Base name for the figure files
        formats: List of file formats to save
    """
    if save_path is not None:
        output_dir = Path(save_path)
        os.makedirs(output_dir, exist_ok=True)

        for ext in formats:
            save_file = output_dir / f"{figure_name}.{ext}"
            fig.savefig(save_file, dpi=300, bbox_inches='tight')
            print(f"Saved figure: {save_file}")


def _get_available_parameters(
    parameters: Dict[str, torch.Tensor],
    exclude_params: Optional[List[str]] = None
) -> List[str]:
    """
    Get list of available parameters, optionally excluding specified ones.

    Args:
        parameters: Dictionary of parameter tensors
        exclude_params: Optional list of parameter names to exclude

    Returns:
        List of available parameter names
    """
    exclude_params = exclude_params or []
    return [name for name in parameters.keys() if name not in exclude_params]

@beartype
def plot_parameter_marginals(
    parameters: Dict[str, torch.Tensor],
    check_type: str = "prior",
    exclude_params: Optional[List[str]] = None,
    figsize: Optional[Tuple[int, int]] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot individual histograms for all parameter marginal distributions.

    Args:
        parameters: Dictionary of parameter tensors
        check_type: Type of check ("prior" or "posterior")
        exclude_params: Optional list of parameter names to exclude
        figsize: Optional figure size (auto-calculated if None)
        save_path: Optional directory path to save figures

    Returns:
        matplotlib Figure object
    """
    available_params = _get_available_parameters(parameters, exclude_params)

    if not available_params:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, 'No parameters available',
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'{check_type.title()} Parameter Marginals')
        return fig

    # Auto-calculate figure size based on number of parameters
    n_params = len(available_params)
    if figsize is None:
        cols = min(4, n_params)
        rows = (n_params + cols - 1) // cols
        figsize = (4 * cols, 3 * rows)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if n_params == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()

    colors = sns.color_palette("husl", n_params)

    for i, param_name in enumerate(available_params):
        ax = axes[i]
        values = parameters[param_name].flatten().numpy()

        # Use relative frequency instead of density for consistent y-axis interpretation
        counts, bins, _ = ax.hist(values, bins=30, alpha=0.7, color=colors[i], density=False)

        # Normalize to relative frequency (0-1 scale)
        total_count = len(values)
        relative_freq = counts / total_count

        # Clear and replot with relative frequency
        ax.clear()
        ax.bar(bins[:-1], relative_freq, width=np.diff(bins), alpha=0.7, color=colors[i],
               align='edge', edgecolor='none')

        ax.set_xlabel('Value')
        ax.set_ylabel('Relative Frequency')
        ax.set_title(_format_parameter_name(param_name))
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, max(relative_freq) * 1.1)  # Add some headroom

        # Add summary statistics
        mean_val = values.mean()
        ax.axvline(mean_val, color='red', linestyle='--', alpha=0.7,
                  label=f'Mean: {mean_val:.3f}')
        ax.legend(fontsize=8)

    # Hide unused subplots
    for i in range(n_params, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()

    if save_path is not None:
        _save_figure(fig, save_path, f"{check_type}_parameter_marginals")

    return fig


@beartype
def plot_parameter_relationships(
    parameters: Dict[str, torch.Tensor],
    check_type: str = "prior",
    figsize: Tuple[int, int] = (12, 4),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot parameter relationships: correlations, fold-change, and timing.

    Args:
        parameters: Dictionary of parameter tensors
        check_type: Type of check ("prior" or "posterior")
        figsize: Figure size (width, height)
        save_path: Optional directory path to save figures

    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Hierarchical time structure
    _plot_hierarchical_time_structure(parameters, axes[0], check_type)

    # Fold-change distribution
    _plot_fold_change_distribution(parameters, axes[1], check_type)

    # Activation timing
    _plot_activation_timing(parameters, axes[2], check_type)

    plt.tight_layout()

    if save_path is not None:
        _save_figure(fig, save_path, f"{check_type}_parameter_relationships")

    return fig


@beartype
def plot_expression_validation(
    adata: AnnData,
    check_type: str = "prior",
    figsize: Tuple[int, int] = (10, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot expression data validation: counts, relationships, library sizes, ranges.

    Args:
        adata: AnnData object with expression data
        check_type: Type of check ("prior" or "posterior")
        figsize: Figure size (width, height) - defaults to square aspect ratio
        save_path: Optional directory path to save figures

    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Count distributions (top-left)
    _plot_count_distributions(adata, axes[0, 0], check_type)

    # U vs S relationships (top-right)
    _plot_expression_relationships(adata, axes[0, 1], check_type)

    # Library sizes (bottom-left)
    _plot_library_sizes(adata, axes[1, 0], check_type)

    # Expression ranges (bottom-right)
    _plot_expression_ranges(adata, axes[1, 1], check_type)

    plt.tight_layout()

    if save_path is not None:
        _save_figure(fig, save_path, f"{check_type}_expression_validation")

    return fig


@beartype
def plot_temporal_dynamics(
    adata: AnnData,
    check_type: str = "prior",
    figsize: Tuple[int, int] = (10, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot temporal dynamics: phase portraits and velocity distributions.

    Args:
        adata: AnnData object with expression data
        check_type: Type of check ("prior" or "posterior")
        figsize: Figure size (width, height)
        save_path: Optional directory path to save figures

    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Phase portrait
    _plot_phase_portrait(adata, axes[0], check_type)

    # Velocity magnitudes
    _plot_velocity_magnitudes(adata, axes[1], check_type)

    plt.tight_layout()

    if save_path is not None:
        _save_figure(fig, save_path, f"{check_type}_temporal_dynamics")

    return fig


@beartype
def plot_pattern_analysis(
    adata: AnnData,
    parameters: Dict[str, torch.Tensor],
    check_type: str = "prior",
    figsize: Tuple[int, int] = (10, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot pattern analysis: proportions and gene correlations.

    Args:
        adata: AnnData object with expression data
        parameters: Dictionary of parameter tensors
        check_type: Type of check ("prior" or "posterior")
        figsize: Figure size (width, height)
        save_path: Optional directory path to save figures

    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Pattern proportions
    _plot_pattern_proportions(adata, parameters, axes[0], check_type)

    # Gene correlations
    _plot_correlation_structure(adata, axes[1], check_type)

    plt.tight_layout()

    if save_path is not None:
        _save_figure(fig, save_path, f"{check_type}_pattern_analysis")

    return fig


@beartype
def plot_prior_predictive_checks(
    model: Any,
    prior_adata: AnnData,
    prior_parameters: Dict[str, torch.Tensor],
    figsize: Tuple[int, int] = (15, 10),
    check_type: str = "prior",
    save_path: Optional[str] = None,
    figure_name: Optional[str] = None,
    create_individual_plots: bool = True
) -> plt.Figure:
    """
    Generate comprehensive predictive check plots for PyroVelocity models.

    This function creates a multi-panel figure showing parameter distributions,
    expression data validation, temporal dynamics, and biological plausibility checks.
    Can be used for both prior and posterior predictive checks.

    Optionally creates individual modular plots in addition to the overview.

    Args:
        model: PyroVelocity model instance (unused but kept for compatibility)
        prior_adata: AnnData object with predictive samples
        prior_parameters: Dictionary of parameter samples
        figsize: Figure size (width, height)
        check_type: Type of check ("prior" or "posterior")
        save_path: Optional directory path to save figures (creates if doesn't exist)
        figure_name: Optional figure name (defaults to "{check_type}_predictive_checks")
        create_individual_plots: Whether to create individual modular plots

    Returns:
        matplotlib Figure object

    Example:
        >>> fig = plot_prior_predictive_checks(
        ...     model=model,
        ...     prior_adata=adata,
        ...     prior_parameters=params,
        ...     save_path="reports/docs/prior_predictive",
        ...     figure_name="piecewise_activation_prior_checks"
        ... )
    """
    # Create individual modular plots if requested
    if create_individual_plots and save_path is not None:
        # Process parameters for plotting compatibility (handle batch dimensions)
        processed_parameters = _process_parameters_for_plotting(prior_parameters)

        plot_parameter_marginals(processed_parameters, check_type, save_path=save_path)
        plot_parameter_relationships(processed_parameters, check_type, save_path=save_path)
        plot_expression_validation(prior_adata, check_type, save_path=save_path)
        plot_temporal_dynamics(prior_adata, check_type, save_path=save_path)
        plot_pattern_analysis(prior_adata, processed_parameters, check_type, save_path=save_path)

    # Process parameters for plotting compatibility (handle batch dimensions)
    processed_parameters = _process_parameters_for_plotting(prior_parameters)

    # Create comprehensive overview plot
    fig = plt.figure(figsize=figsize)

    # Create subplot grid: 3 rows × 4 columns
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

    # Row 1: UMAP and Parameter Distribution Plots
    ax1 = fig.add_subplot(gs[0, 0])
    _plot_umap_leiden_clusters(prior_adata, ax1, check_type)

    ax2 = fig.add_subplot(gs[0, 1])
    _plot_umap_time_coordinate(prior_adata, ax2, check_type)

    ax3 = fig.add_subplot(gs[0, 2])
    _plot_fold_change_distribution(processed_parameters, ax3, check_type)

    ax4 = fig.add_subplot(gs[0, 3])
    _plot_activation_timing(processed_parameters, ax4, check_type)

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
    _plot_pattern_proportions(prior_adata, processed_parameters, ax11, check_type)

    ax12 = fig.add_subplot(gs[2, 3])
    _plot_correlation_structure(prior_adata, ax12, check_type)

    # Save comprehensive figure if path is provided
    if save_path is not None:
        # Use provided name or default
        name = figure_name if figure_name is not None else f"{check_type}_predictive_checks"
        _save_figure(fig, save_path, name)

    return fig


def _plot_umap_leiden_clusters(adata: AnnData, ax: plt.Axes, check_type: str) -> None:
    """Plot UMAP embedding colored by Leiden clusters."""
    if 'X_umap' in adata.obsm and 'leiden' in adata.obs:
        umap_coords = adata.obsm['X_umap']
        clusters = adata.obs['leiden']

        # Get unique clusters and assign colors
        unique_clusters = clusters.unique()
        colors = sns.color_palette("tab10", len(unique_clusters))

        for i, cluster in enumerate(unique_clusters):
            mask = clusters == cluster
            ax.scatter(umap_coords[mask, 0], umap_coords[mask, 1],
                      c=[colors[i]], label=f'Cluster {cluster}',
                      alpha=0.7, s=20)

        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.set_title(f'{check_type.title()} UMAP (Leiden Clusters)')

    elif 'X_umap' in adata.obsm:
        # Plot UMAP without cluster information
        umap_coords = adata.obsm['X_umap']
        ax.scatter(umap_coords[:, 0], umap_coords[:, 1],
                  alpha=0.7, s=20, c='gray')
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.set_title(f'{check_type.title()} UMAP (No Clusters)')

    else:
        # Compute UMAP if not available and UMAP is installed
        if UMAP_AVAILABLE and 'X_pca' in adata.obsm:
            try:
                reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
                embedding = reducer.fit_transform(adata.obsm['X_pca'][:, :50])  # Use first 50 PCs

                ax.scatter(embedding[:, 0], embedding[:, 1],
                          alpha=0.7, s=20, c='gray')
                ax.set_xlabel('UMAP 1')
                ax.set_ylabel('UMAP 2')
                ax.set_title(f'{check_type.title()} UMAP (Computed)')

            except Exception as e:
                ax.text(0.5, 0.5, f'UMAP computation failed:\n{str(e)[:50]}...',
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{check_type.title()} UMAP (Failed)')
        else:
            ax.text(0.5, 0.5, 'UMAP data not available\nor UMAP not installed',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{check_type.title()} UMAP (Clusters)')


def _plot_umap_time_coordinate(adata: AnnData, ax: plt.Axes, check_type: str) -> None:
    """Plot UMAP embedding colored by time coordinate."""
    if 'X_umap' in adata.obsm:
        umap_coords = adata.obsm['X_umap']

        # Look for time coordinate in various possible locations
        time_coord = None
        time_label = 'Time'

        # Check common time coordinate names
        time_keys = ['latent_time', 'velocity_pseudotime', 'dpt_pseudotime',
                    'pseudotime', 'time', 't', 'shared_time']

        for key in time_keys:
            if key in adata.obs:
                time_coord = adata.obs[key].values
                time_label = _format_parameter_name(key.replace('_', '_'))
                break

        if time_coord is not None:
            # Create scatter plot colored by time
            scatter = ax.scatter(umap_coords[:, 0], umap_coords[:, 1],
                               c=time_coord, cmap='viridis', alpha=0.7, s=20)

            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
            cbar.set_label(time_label, fontsize=10)

            ax.set_xlabel('UMAP 1')
            ax.set_ylabel('UMAP 2')
            ax.set_title(f'{check_type.title()} UMAP (Time Coordinate)')

        else:
            # No time coordinate found, use a simple gradient based on position
            gradient = np.arange(len(umap_coords))
            scatter = ax.scatter(umap_coords[:, 0], umap_coords[:, 1],
                               c=gradient, cmap='viridis', alpha=0.7, s=20)

            cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
            cbar.set_label('Cell Index', fontsize=10)

            ax.set_xlabel('UMAP 1')
            ax.set_ylabel('UMAP 2')
            ax.set_title(f'{check_type.title()} UMAP (Cell Index)')

    else:
        # Compute UMAP if not available and UMAP is installed
        if UMAP_AVAILABLE and 'X_pca' in adata.obsm:
            try:
                reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
                embedding = reducer.fit_transform(adata.obsm['X_pca'][:, :50])

                # Use cell index as pseudo-time
                gradient = np.arange(len(embedding))
                scatter = ax.scatter(embedding[:, 0], embedding[:, 1],
                                   c=gradient, cmap='viridis', alpha=0.7, s=20)

                cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
                cbar.set_label('Cell Index', fontsize=10)

                ax.set_xlabel('UMAP 1')
                ax.set_ylabel('UMAP 2')
                ax.set_title(f'{check_type.title()} UMAP (Computed)')

            except Exception as e:
                ax.text(0.5, 0.5, f'UMAP computation failed:\n{str(e)[:50]}...',
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{check_type.title()} UMAP (Failed)')
        else:
            ax.text(0.5, 0.5, 'UMAP data not available\nor UMAP not installed',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{check_type.title()} UMAP (Time Coordinate)')


def _process_parameters_for_plotting(
    parameters: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """
    Process parameters to make them compatible with plotting functions.

    The plotting functions expect parameters to be flattened 1D arrays, but posterior
    samples from SVI have batch dimensions. This method handles the tensor reshaping
    to make the samples compatible with the plotting code.

    Args:
        parameters: Raw parameters with potential batch dimensions

    Returns:
        Processed parameters suitable for plotting functions
    """
    processed_parameters = {}

    for key, value in parameters.items():
        if isinstance(value, torch.Tensor):
            # Handle different tensor shapes
            if value.ndim == 1:
                # Already 1D, use as-is
                processed_parameters[key] = value
            elif value.ndim == 2:
                # 2D tensor: [num_samples, param_dim] or [batch_size, param_dim]
                # Flatten to 1D for plotting
                processed_parameters[key] = value.flatten()
            elif value.ndim == 3:
                # 3D tensor: [batch_size, num_samples, param_dim]
                # Remove batch dimension and flatten
                if value.shape[0] == 1:
                    # Remove batch dimension: [1, num_samples, param_dim] -> [num_samples, param_dim]
                    squeezed = value.squeeze(0)
                    processed_parameters[key] = squeezed.flatten()
                else:
                    # Multiple batches: flatten everything
                    processed_parameters[key] = value.flatten()
            else:
                # Higher dimensions: flatten everything
                processed_parameters[key] = value.flatten()
        else:
            # Non-tensor values: keep as-is
            processed_parameters[key] = value

    return processed_parameters


def _plot_parameter_marginals_summary(
    parameters: Dict[str, torch.Tensor],
    ax: plt.Axes,
    check_type: str
) -> None:
    """Plot marginal distributions of key parameters (summary version for overview)."""
    # Focus on piecewise activation parameters
    key_params = ['alpha_off', 'alpha_on', 't_on_star', 'delta_star']

    colors = sns.color_palette("husl", len(key_params))

    for i, param_name in enumerate(key_params):
        if param_name in parameters:
            values = parameters[param_name].flatten().numpy()
            # Use relative frequency for consistency with main marginals plot
            ax.hist(values, bins=30, alpha=0.6, color=colors[i],
                   label=_format_parameter_name(param_name), density=False,
                   weights=np.ones(len(values)) / len(values))

    ax.set_xlabel('Parameter Value')
    ax.set_ylabel('Relative Frequency')
    ax.set_title(f'{check_type.title()} Parameter Marginals')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)



def _plot_hierarchical_time_structure(
    parameters: Dict[str, torch.Tensor],
    ax: plt.Axes,
    check_type: str
) -> None:
    """Plot hierarchical time parameter relationships."""
    # Check for hierarchical time parameters
    time_params = ['T_M_star', 't_loc', 't_scale']
    available_time_params = [p for p in time_params if p in parameters]

    if len(available_time_params) >= 2:
        # Plot T_M_star vs population time spread (t_scale)
        if 'T_M_star' in parameters and 't_scale' in parameters:
            T_M = parameters['T_M_star'].flatten().numpy()
            t_scale = parameters['t_scale'].flatten().numpy()

            ax.scatter(T_M, t_scale, alpha=0.6, s=20, color='purple')
            ax.set_xlabel(f'Global Time Scale ({_format_parameter_name("T_M_star")})')
            ax.set_ylabel(f'Population Time Spread ({_format_parameter_name("t_scale")})')
            ax.set_title(f'{check_type.title()} Hierarchical Time Structure')

            # Add interpretation guidelines
            ax.axhline(0.2, color='orange', linestyle='--', alpha=0.7,
                      label='Tight temporal clustering')
            ax.axvline(5.0, color='green', linestyle='--', alpha=0.7,
                      label='Typical process duration')
            ax.legend(fontsize=8)

        # Alternative: t_loc vs t_scale if T_M_star not available
        elif 't_loc' in parameters and 't_scale' in parameters:
            t_loc = parameters['t_loc'].flatten().numpy()
            t_scale = parameters['t_scale'].flatten().numpy()

            ax.scatter(t_loc, t_scale, alpha=0.6, s=20, color='purple')
            ax.set_xlabel(f'Population Time Location ({_format_parameter_name("t_loc")})')
            ax.set_ylabel(f'Population Time Spread ({_format_parameter_name("t_scale")})')
            ax.set_title(f'{check_type.title()} Population Time Parameters')
            ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, 'Hierarchical time parameters\nnot available',
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'{check_type.title()} Hierarchical Time Structure')

    ax.grid(True, alpha=0.3)


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
        
        # Use relative frequency for consistency
        ax.hist(fold_change, bins=50, alpha=0.7, color='skyblue', density=False,
               weights=np.ones(len(fold_change)) / len(fold_change))
        ax.axvline(fold_change.mean(), color='red', linestyle='--',
                  label=f'Mean: {fold_change.mean():.1f}')
        ax.axvline(3.3, color='orange', linestyle=':', label='Min threshold: 3.3')
        ax.axvline(7.5, color='green', linestyle=':', label='Activation threshold: 7.5')

        ax.set_xlabel(f'Fold-change ({_format_parameter_name("alpha_on")} / {_format_parameter_name("alpha_off")})')
        ax.set_ylabel('Relative Frequency')
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
        ax.set_xlabel(f'Activation Onset ({_format_parameter_name("t_on_star")})')
        ax.set_ylabel(f'Activation Duration ({_format_parameter_name("delta_star")})')
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
        
        # Use relative frequency for consistency
        ax.hist(np.log1p(unspliced_nz), bins=50, alpha=0.6,
               label='Unspliced', color='red', density=False,
               weights=np.ones(len(unspliced_nz)) / len(unspliced_nz))
        ax.hist(np.log1p(spliced_nz), bins=50, alpha=0.6,
               label='Spliced', color='blue', density=False,
               weights=np.ones(len(spliced_nz)) / len(spliced_nz))

        ax.set_xlabel('log(count + 1)')
        ax.set_ylabel('Relative Frequency')
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

        # Use relative frequency for consistency
        ax.hist(total_counts, bins=50, alpha=0.7, color='green', density=False,
               weights=np.ones(len(total_counts)) / len(total_counts))
        ax.axvline(total_counts.mean(), color='red', linestyle='--',
                  label=f'Mean: {total_counts.mean():.0f}')

        ax.set_xlabel('Total Counts per Cell')
        ax.set_ylabel('Relative Frequency')
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

        # Use relative frequency for consistency
        ax.hist(velocity_magnitudes, bins=50, alpha=0.7, color='teal', density=False,
               weights=np.ones(len(velocity_magnitudes)) / len(velocity_magnitudes))
        ax.axvline(velocity_magnitudes.mean(), color='red', linestyle='--',
                  label=f'Mean: {velocity_magnitudes.mean():.3f}')

        ax.set_xlabel('Velocity Magnitude')
        ax.set_ylabel('Relative Frequency')
        ax.set_title(f'{check_type.title()} Velocity Magnitudes')
        ax.legend()
    else:
        # Compute simple velocity approximation from U/S ratio changes
        if 'unspliced' in adata.layers and 'spliced' in adata.layers:
            u = adata.layers['unspliced']
            s = adata.layers['spliced']

            # Simple velocity approximation: du/dt ≈ α - γu (assuming steady-state splicing)
            velocity_approx = np.mean(u - s, axis=1)  # Simplified

            # Use relative frequency for consistency
            ax.hist(velocity_approx, bins=50, alpha=0.7, color='teal', density=False,
                   weights=np.ones(len(velocity_approx)) / len(velocity_approx))
            ax.axvline(velocity_approx.mean(), color='red', linestyle='--',
                      label=f'Mean: {velocity_approx.mean():.3f}')

            ax.set_xlabel('Velocity Approximation')
            ax.set_ylabel('Relative Frequency')
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

        ax.pie(counts, labels=patterns, colors=colors,
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
    figsize: Tuple[int, int] = (15, 10),
    save_path: Optional[str] = None,
    figure_name: Optional[str] = None
) -> plt.Figure:
    """
    Generate posterior predictive check plots.

    This is an alias for plot_prior_predictive_checks with check_type="posterior".

    Args:
        model: PyroVelocity model instance
        posterior_adata: AnnData object with posterior predictive samples
        posterior_parameters: Dictionary of posterior parameter samples
        figsize: Figure size (width, height)
        save_path: Optional directory path to save figures (creates if doesn't exist)
        figure_name: Optional figure name (defaults to "posterior_predictive_checks")

    Returns:
        matplotlib Figure object

    Example:
        >>> fig = plot_posterior_predictive_checks(
        ...     model=model,
        ...     posterior_adata=adata,
        ...     posterior_parameters=params,
        ...     save_path="reports/docs/posterior_predictive",
        ...     figure_name="piecewise_activation_posterior_checks"
        ... )
    """
    return plot_prior_predictive_checks(
        model=model,
        prior_adata=posterior_adata,
        prior_parameters=posterior_parameters,
        figsize=figsize,
        check_type="posterior",
        save_path=save_path,
        figure_name=figure_name
    )
