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


def _latex_safe_text(text: str) -> str:
    """
    Make text safe for LaTeX rendering by escaping special characters.

    Args:
        text: Input text that may contain LaTeX special characters

    Returns:
        LaTeX-safe text with special characters escaped
    """
    # Escape underscores for LaTeX
    text = text.replace('_', r'\_')

    # Replace other problematic characters
    replacements = {
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '^': r'\^{}',
        '~': r'\~{}',
        '{': r'\{',
        '}': r'\}',
    }

    for char, replacement in replacements.items():
        text = text.replace(char, replacement)

    return text


def _format_pattern_name(pattern_name: str) -> str:
    """
    Format pattern names for display in legends and titles.

    Args:
        pattern_name: Raw pattern name (e.g., 'pre_activation', 'transient')

    Returns:
        Formatted pattern name suitable for LaTeX rendering
    """
    # Pattern name mappings for better display
    pattern_mappings = {
        'pre_activation': 'Pre-activation',
        'transient': 'Transient',
        'sustained': 'Sustained'
    }

    formatted = pattern_mappings.get(pattern_name, pattern_name.replace('_', ' ').title())
    return _latex_safe_text(formatted)


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
        'tilde_t_on_star': r'\tilde{t}^*_{on}',
        'tilde_delta_star': r'\tilde{\delta}^*',
        't_on_star': r't^*_{on}',
        'delta_star': r'\delta^*',
        'lambda_j': r'\lambda_j',
        'U_0i': r'U_{0i}',
        'latent_time': r't_{latent}',
        'velocity_pseudotime': r't_{velocity}',
        'dpt_pseudotime': r't_{dpt}',
        'pseudotime': r't_{pseudo}',
        'shared_time': r't_{shared}'
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


@beartype
def combine_pdfs(
    pdf_directory: str,
    output_filename: str = "combined_predictive_checks.pdf",
    pdf_pattern: str = "*.pdf",
    exclude_patterns: Optional[List[str]] = None
) -> None:
    """
    Combine multiple PDF files into a single PDF using only Python libraries.

    This function uses PyPDF2/pypdf to merge PDF files without requiring system calls.
    It's designed to work with the output from plot_prior_predictive_checks.

    Args:
        pdf_directory: Directory containing PDF files to combine
        output_filename: Name of the combined output PDF file
        pdf_pattern: Glob pattern to match PDF files (default: "*.pdf")
        exclude_patterns: Optional list of filename patterns to exclude from combination

    Example:
        >>> # Combine all PDFs from prior predictive checks
        >>> combine_pdfs(
        ...     pdf_directory="reports/docs/prior_predictive",
        ...     output_filename="combined_prior_checks.pdf",
        ...     exclude_patterns=["combined_*.pdf"]  # Don't include previous combined files
        ... )
    """
    from pypdf import PdfReader, PdfWriter

    pdf_dir = Path(pdf_directory)
    if not pdf_dir.exists():
        raise FileNotFoundError(f"Directory {pdf_directory} does not exist")

    # Find all PDF files matching the pattern
    pdf_files = list(pdf_dir.glob(pdf_pattern))

    # Apply exclusion patterns
    if exclude_patterns:
        filtered_files = []
        for pdf_file in pdf_files:
            exclude = False
            for pattern in exclude_patterns:
                if pdf_file.match(pattern):
                    exclude = True
                    break
            if not exclude:
                filtered_files.append(pdf_file)
        pdf_files = filtered_files

    if not pdf_files:
        print(f"No PDF files found in {pdf_directory} matching pattern '{pdf_pattern}'")
        return

    # Sort files for consistent ordering
    pdf_files.sort()

    # Create PDF writer object
    pdf_writer = PdfWriter()

    # Add each PDF to the writer
    for pdf_file in pdf_files:
        try:
            pdf_reader = PdfReader(str(pdf_file))
            for page in pdf_reader.pages:
                pdf_writer.add_page(page)
            print(f"Added {pdf_file.name} ({len(pdf_reader.pages)} pages)")
        except Exception as e:
            print(f"Warning: Could not read {pdf_file.name}: {e}")
            continue

    # Write combined PDF
    output_path = pdf_dir / output_filename
    with open(output_path, 'wb') as output_file:
        pdf_writer.write(output_file)

    print(f"Combined {len(pdf_files)} PDFs into {output_path}")
    print(f"Total pages: {len(pdf_writer.pages)}")


def _get_available_parameters(
    parameters: Dict[str, torch.Tensor],
    exclude_params: Optional[List[str]] = None,
    model: Optional[Any] = None
) -> List[str]:
    """
    Get list of available parameters, optionally excluding specified ones.

    Parameters are ordered by plot_order from metadata if available, otherwise alphabetically.

    Args:
        parameters: Dictionary of parameter tensors
        exclude_params: Optional list of parameter names to exclude
        model: Optional PyroVelocity model instance for parameter metadata

    Returns:
        List of available parameter names in proper order
    """
    # Default exclusions for deprecated parameters in corrected parameterization
    default_exclude = ['alpha_off', 'alpha_on']  # alpha_off fixed at 1.0, alpha_on computed from R_on
    exclude_params = (exclude_params or []) + default_exclude

    available_params = [name for name in parameters.keys() if name not in exclude_params]

    # Try to order by metadata plot_order if available
    try:
        from pyrovelocity.plots.parameter_metadata import (
            get_model_parameter_metadata,
        )

        if model is not None:
            metadata = get_model_parameter_metadata(model)
            if metadata is not None:
                # Create ordering based on plot_order
                param_order = {}
                for param_name in available_params:
                    if param_name in metadata.parameters:
                        plot_order = metadata.parameters[param_name].plot_order
                        param_order[param_name] = plot_order if plot_order is not None else 999
                    else:
                        param_order[param_name] = 999  # Put unordered params at end

                # Sort by plot_order, then alphabetically for ties
                available_params.sort(key=lambda x: (param_order.get(x, 999), x))
                return available_params
    except Exception:
        pass  # Fall back to alphabetical ordering

    # Fall back to alphabetical ordering
    return sorted(available_params)

@beartype
def plot_parameter_marginals(
    parameters: Dict[str, torch.Tensor],
    check_type: str = "prior",
    exclude_params: Optional[List[str]] = None,
    figsize: Optional[Tuple[int, int]] = None,
    save_path: Optional[str] = None,
    file_prefix: str = "",
    model: Optional[Any] = None
) -> plt.Figure:
    """
    Plot individual histograms for all parameter marginal distributions.

    Args:
        parameters: Dictionary of parameter tensors
        check_type: Type of check ("prior" or "posterior")
        exclude_params: Optional list of parameter names to exclude
        figsize: Optional figure size (auto-calculated if None)
        save_path: Optional directory path to save figures
        file_prefix: Prefix for saved file names
        model: Optional PyroVelocity model instance for parameter metadata

    Returns:
        matplotlib Figure object
    """
    available_params = _get_available_parameters(parameters, exclude_params, model)

    if not available_params:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, 'No parameters available',
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'{check_type.title()} Parameter Marginals')
        return fig

    # Auto-calculate figure size based on number of parameters
    # Use 7.5" width to fit 8.5x11" page with 0.5" margins
    n_params = len(available_params)
    cols = min(4, n_params)
    rows = (n_params + cols - 1) // cols

    if figsize is None:
        width = 7.5  # Standard width for 8.5x11" with margins
        height = width * (rows / cols) * 0.75  # Maintain reasonable aspect ratio
        figsize = (width, height)

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

        # Get parameter label using the new metadata system
        from pyrovelocity.plots.parameter_metadata import (
            get_parameter_label,
            infer_component_name_from_parameters,
        )

        # Try to infer component name from all parameters if model not provided
        component_name = None
        if model is None:
            component_name = infer_component_name_from_parameters(parameters)

        # Get short label for x-axis and display name for title
        short_label = get_parameter_label(
            param_name=param_name,
            label_type="short",
            model=model,
            component_name=component_name,
            fallback_to_legacy=True
        )
        display_name = get_parameter_label(
            param_name=param_name,
            label_type="display",
            model=model,
            component_name=component_name,
            fallback_to_legacy=True
        )

        ax.set_xlabel(short_label, fontsize=7)
        ax.set_ylabel('Freq.', fontsize=7)
        ax.set_title(display_name, fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, max(relative_freq) * 1.1)  # Add some headroom
        ax.tick_params(labelsize=6)  # Reduce tick label size

        # Add median line in light gray instead of mean line with legend
        median_val = np.median(values)
        ax.axvline(median_val, color='lightgray', linestyle='--', alpha=0.7)

    # Hide unused subplots
    for i in range(n_params, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()

    if save_path is not None:
        prefix = f"{file_prefix}_" if file_prefix else ""
        _save_figure(fig, save_path, f"{prefix}{check_type}_parameter_marginals")

    return fig


@beartype
def plot_parameter_relationships(
    parameters: Dict[str, torch.Tensor],
    check_type: str = "prior",
    figsize: Tuple[int, int] = (7.5, 2.5),  # Standard width, appropriate height
    save_path: Optional[str] = None,
    file_prefix: str = "",
    model: Optional[Any] = None
) -> plt.Figure:
    """
    Plot parameter relationships: correlations, fold-change, and timing.

    Args:
        parameters: Dictionary of parameter tensors
        check_type: Type of check ("prior" or "posterior")
        figsize: Figure size (width, height)
        save_path: Optional directory path to save figures
        file_prefix: Prefix for saved file names
        model: Optional PyroVelocity model instance for parameter metadata

    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Hierarchical time structure
    _plot_hierarchical_time_structure(parameters, axes[0], check_type, model=model)

    # Fold-change distribution
    _plot_fold_change_distribution(parameters, axes[1], check_type, model=model)

    # Activation timing
    _plot_activation_timing(parameters, axes[2], check_type, model=model)

    plt.tight_layout()

    if save_path is not None:
        prefix = f"{file_prefix}_" if file_prefix else ""
        _save_figure(fig, save_path, f"{prefix}{check_type}_parameter_relationships")

    return fig


@beartype
def plot_expression_validation(
    adata: AnnData,
    check_type: str = "prior",
    figsize: Tuple[int, int] = (7.5, 7.5),  # Square aspect ratio with standard width
    save_path: Optional[str] = None,
    file_prefix: str = ""
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
        prefix = f"{file_prefix}_" if file_prefix else ""
        _save_figure(fig, save_path, f"{prefix}{check_type}_expression_validation")

    return fig


@beartype
def plot_temporal_dynamics(
    adata: AnnData,
    check_type: str = "prior",
    figsize: Optional[Tuple[int, int]] = None,
    save_path: Optional[str] = None,
    num_genes: int = 6,
    basis: str = "umap",
    default_fontsize: int = 7,
    file_prefix: str = ""
) -> plt.Figure:
    """
    Plot temporal dynamics: multi-gene visualization with phase portraits,
    spliced dynamics, predictive and observed expression in UMAP space.

    Creates a rainbow-plot style visualization with one row per gene showing:
    - (u,s) phase space scatter plot
    - Spliced dynamics over time
    - Predictive spliced expression in UMAP space
    - Observed log spliced expression in UMAP space

    Uses the same gridspec layout as the rainbow plot for proper spacing and formatting.

    Args:
        adata: AnnData object with expression data
        check_type: Type of check ("prior" or "posterior")
        figsize: Figure size (width, height). If None, auto-calculated based on num_genes
        save_path: Optional directory path to save figures
        num_genes: Number of genes to display (default: 6)
        basis: Embedding basis for spatial plots (default: "umap")
        default_fontsize: Default font size for labels and titles (default: 7)

    Returns:
        matplotlib Figure object
    """
    from matplotlib.gridspec import GridSpec

    from pyrovelocity.plots._common import set_colorbar, set_font_size

    # Set font size
    set_font_size(default_fontsize)

    # Select genes to plot
    available_genes = min(num_genes, adata.n_vars)
    if available_genes < num_genes:
        print(f"Warning: Only {available_genes} genes available, plotting all")

    gene_indices = np.random.choice(adata.n_vars, available_genes, replace=False)
    gene_names = [adata.var_names[i] for i in gene_indices]

    # Create figure and gridspec using rainbow plot pattern
    # Use standard width if figsize not provided
    if figsize is None:
        width = 7.5  # Standard width for 8.5x11" with margins
        height = width * (available_genes / 4) * 0.6  # Reasonable height based on gene count
        figsize = (width, height)

    fig, axes_dict = _create_temporal_dynamics_figure(available_genes, figsize)

    # Plot each gene
    for n, (gene_idx, gene_name) in enumerate(zip(gene_indices, gene_names)):
        # Phase portrait
        _plot_gene_phase_portrait_rainbow(adata, axes_dict, n, gene_idx, gene_name, check_type, available_genes)

        # Spliced dynamics
        _plot_gene_spliced_dynamics_rainbow(adata, axes_dict, n, gene_idx, gene_name, check_type, available_genes)

        # Predictive spliced in UMAP
        _plot_gene_predictive_umap_rainbow(adata, axes_dict, n, gene_idx, gene_name, check_type, basis)

        # Observed spliced in UMAP
        _plot_gene_observed_umap_rainbow(adata, axes_dict, n, gene_idx, gene_name, check_type, basis)

        # Set labels and formatting
        _set_temporal_dynamics_labels(axes_dict, n, gene_name, available_genes, default_fontsize)

    # Set aspect ratios like rainbow plot
    _set_temporal_dynamics_aspect(axes_dict)

    fig.tight_layout()

    if save_path is not None:
        prefix = f"{file_prefix}_" if file_prefix else ""
        _save_figure(fig, save_path, f"{prefix}{check_type}_temporal_dynamics")

    return fig


@beartype
def plot_temporal_trajectories(
    parameters: Dict[str, torch.Tensor],
    check_type: str = "prior",
    figsize: Optional[Tuple[int, int]] = None,
    save_path: Optional[str] = None,
    file_prefix: str = "",
    n_examples: int = 10,
    n_time_points: int = 300,
    buffer_factor: float = 1.2
) -> plt.Figure:
    """
    Plot temporal trajectories showing underlying continuous dynamics.

    This function creates trajectory plots similar to those in prior-hyperparameter-calibration.py,
    showing the continuous temporal dynamics that underlie the discrete count observations.
    Plots are organized by pattern type (pre-activation, transient, sustained) with multiple
    examples per pattern.

    Args:
        parameters: Dictionary of parameter tensors from prior/posterior samples
        check_type: Type of check ("prior" or "posterior")
        figsize: Figure size (width, height). If None, auto-calculated
        save_path: Optional directory path to save figures
        file_prefix: Prefix for saved file names
        n_examples: Number of examples to show per pattern (default: 10)
        n_time_points: Number of time points for trajectory evaluation (default: 300)
        buffer_factor: Multiplicative buffer beyond T_M_star for time range (default: 1.2)

    Returns:
        matplotlib Figure object
    """
    # Classify parameters into patterns and select examples
    pattern_examples = _classify_parameters_into_patterns(parameters, n_examples)

    if not pattern_examples:
        # Create empty figure if no patterns found
        fig, ax = plt.subplots(figsize=(7.5, 4))
        ax.text(0.5, 0.5, 'No pattern examples available',
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'{check_type.title()} Temporal Trajectories')
        return fig

    n_patterns = len(pattern_examples)

    # Auto-calculate figure size if not provided
    if figsize is None:
        width = 15  # Wide enough for 3 columns
        height = 4 * n_patterns  # 4 inches per pattern row
        figsize = (width, height)

    fig, axes = plt.subplots(n_patterns, 3, figsize=figsize)

    if n_patterns == 1:
        axes = axes.reshape(1, -1)

    # Compute adaptive time range based on T_M_star values
    t_star = _compute_adaptive_time_range(pattern_examples, buffer_factor, n_time_points)

    pattern_colors = ['red', 'blue', 'green', 'orange', 'purple']

    for pattern_idx, (pattern_name, examples) in enumerate(pattern_examples.items()):
        if not examples:
            continue

        color = pattern_colors[pattern_idx % len(pattern_colors)]
        formatted_pattern = _format_pattern_name(pattern_name)

        # Plot multiple examples for this pattern
        for example_idx, params in enumerate(examples):
            # Compute time courses using piecewise dynamics
            u_star, s_star = _compute_time_course(t_star, params)

            # Apply log2 transformation to show fold changes more clearly
            u_star_log2 = torch.log2(u_star)
            s_star_log2 = torch.log2(s_star)

            # Plot unspliced (log2 scale)
            axes[pattern_idx, 0].plot(
                t_star.numpy(), u_star_log2.numpy(),
                color=color, alpha=0.7, linewidth=2,
                label=f'Example {example_idx+1}' if example_idx < 3 else None
            )

            # Plot spliced (log2 scale)
            axes[pattern_idx, 1].plot(
                t_star.numpy(), s_star_log2.numpy(),
                color=color, alpha=0.7, linewidth=2,
                label=f'Example {example_idx+1}' if example_idx < 3 else None
            )

            # Plot phase portrait (both axes log2)
            axes[pattern_idx, 2].plot(
                u_star_log2.numpy(), s_star_log2.numpy(),
                color=color, alpha=0.7, linewidth=2,
                label=f'Example {example_idx+1}' if example_idx < 3 else None
            )

        # Format axes with LaTeX-safe labels (log2 scale for fold changes)
        axes[pattern_idx, 0].set_xlabel(_latex_safe_text('Dimensionless Time (t*)'))
        axes[pattern_idx, 0].set_ylabel(_latex_safe_text('log2(Unspliced) (u*)'))
        axes[pattern_idx, 0].set_title(f'{formatted_pattern}: Unspliced')
        axes[pattern_idx, 0].grid(True, alpha=0.3)
        axes[pattern_idx, 0].legend()

        axes[pattern_idx, 1].set_xlabel(_latex_safe_text('Dimensionless Time (t*)'))
        axes[pattern_idx, 1].set_ylabel(_latex_safe_text('log2(Spliced) (s*)'))
        axes[pattern_idx, 1].set_title(f'{formatted_pattern}: Spliced')
        axes[pattern_idx, 1].grid(True, alpha=0.3)

        axes[pattern_idx, 2].set_xlabel(_latex_safe_text('log2(Unspliced) (u*)'))
        axes[pattern_idx, 2].set_ylabel(_latex_safe_text('log2(Spliced) (s*)'))
        axes[pattern_idx, 2].set_title(f'{formatted_pattern}: Phase Portrait')
        axes[pattern_idx, 2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        prefix = f"{file_prefix}_" if file_prefix else ""
        _save_figure(fig, save_path, f"{prefix}{check_type}_temporal_trajectories")

    return fig


@beartype
def plot_pattern_analysis(
    adata: AnnData,
    parameters: Dict[str, torch.Tensor],
    check_type: str = "prior",
    figsize: Tuple[int, int] = (7.5, 3.75),  # Standard width, half height
    save_path: Optional[str] = None,
    file_prefix: str = ""
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
        prefix = f"{file_prefix}_" if file_prefix else ""
        _save_figure(fig, save_path, f"{prefix}{check_type}_pattern_analysis")

    return fig


def _classify_parameters_into_patterns(
    parameters: Dict[str, torch.Tensor],
    n_examples: int = 10
) -> Dict[str, List[Dict[str, torch.Tensor]]]:
    """
    Classify parameter samples into patterns and select examples.

    Args:
        parameters: Dictionary of parameter tensors
        n_examples: Number of examples to select per pattern

    Returns:
        Dictionary mapping pattern names to lists of parameter dictionaries
    """
    # Check if we have the required parameters
    required_params = ['R_on', 't_on_star', 'delta_star', 'gamma_star']
    if not all(param in parameters for param in required_params):
        print(f"Warning: Missing required parameters for pattern classification: {required_params}")
        return {}

    # Get parameter values (flatten to 1D if needed)
    R_on = parameters['R_on'].flatten()
    t_on_star = parameters['t_on_star'].flatten()
    delta_star = parameters['delta_star'].flatten()
    gamma_star = parameters['gamma_star'].flatten()

    # Ensure all parameters have the same length
    min_length = min(len(R_on), len(t_on_star), len(delta_star), len(gamma_star))
    R_on = R_on[:min_length]
    t_on_star = t_on_star[:min_length]
    delta_star = delta_star[:min_length]
    gamma_star = gamma_star[:min_length]

    # Compute alpha_off (fixed at 1.0) and alpha_on from R_on
    alpha_off = torch.ones_like(R_on)
    alpha_on = R_on * alpha_off  # R_on = alpha_on / alpha_off

    pattern_examples = {
        'pre_activation': [],
        'transient': [],
        'sustained': []
    }

    # Classify each parameter set
    for i in range(min_length):
        # Create parameter dictionary for this sample
        params = {
            'alpha_off': alpha_off[i],
            'alpha_on': alpha_on[i],
            'gamma_star': gamma_star[i],
            't_on_star': t_on_star[i],
            'delta_star': delta_star[i],
            'R_on': R_on[i]
        }

        # Pattern classification logic (simplified from hyperparameter calibration)
        if t_on_star[i] < 0:
            # Negative onset time -> pre-activation pattern
            if len(pattern_examples['pre_activation']) < n_examples:
                pattern_examples['pre_activation'].append(params)
        elif R_on[i] > 2.0 and t_on_star[i] > 0:
            # Positive onset with significant fold change
            if delta_star[i] < 0.4:
                # Short duration -> transient pattern
                if len(pattern_examples['transient']) < n_examples:
                    pattern_examples['transient'].append(params)
            else:
                # Long duration -> sustained pattern
                if len(pattern_examples['sustained']) < n_examples:
                    pattern_examples['sustained'].append(params)

    # Remove empty patterns
    pattern_examples = {k: v for k, v in pattern_examples.items() if v}

    return pattern_examples


def _compute_adaptive_time_range(
    pattern_examples: Dict[str, List[Dict[str, torch.Tensor]]],
    buffer_factor: float = 1.2,
    n_points: int = 300
) -> torch.Tensor:
    """
    Compute adaptive time range based on T_M_star values in pattern examples.

    Args:
        pattern_examples: Dictionary with parameter sets for each pattern
        buffer_factor: Multiplicative buffer beyond T_M_star
        n_points: Number of time points to generate

    Returns:
        Time tensor for trajectory evaluation
    """
    # Find the maximum time scale from all examples
    max_time_scale = 0.0

    for pattern_name, examples in pattern_examples.items():
        for params in examples:
            # Estimate time scale from onset time and duration
            t_on = params['t_on_star'].item()
            delta = params['delta_star'].item()
            gamma = params['gamma_star'].item()

            # Estimate when trajectory returns to baseline (3 time constants)
            decay_time = 3.0 / gamma if gamma > 0 else 10.0

            # Total time scale includes onset, duration, and decay
            total_time = max(0, t_on) + delta + decay_time
            max_time_scale = max(max_time_scale, total_time)

    # Use buffer factor and ensure minimum reasonable range
    time_range_max = max(max_time_scale * buffer_factor, 10.0)

    return torch.linspace(0, time_range_max, n_points)


def _compute_time_course(
    t_star: torch.Tensor,
    params: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute time course using piecewise activation dynamics.

    Args:
        t_star: Time points to evaluate
        params: Parameter dictionary

    Returns:
        Tuple of (u_star, s_star) time courses
    """
    # Extract parameters
    alpha_off = params['alpha_off']
    alpha_on = params['alpha_on']
    gamma_star = params['gamma_star']
    t_on_star = params['t_on_star']
    delta_star = params['delta_star']

    # Initialize output tensors
    u_star = torch.zeros_like(t_star)
    s_star = torch.zeros_like(t_star)

    # Phase 1: Off state (t* < t*_on)
    phase1_mask = t_star < t_on_star
    u_star[phase1_mask] = 1.0  # Fixed reference state
    s_star[phase1_mask] = 1.0 / gamma_star

    # Phase 2: On state (t*_on ≤ t* < t*_on + δ*)
    phase2_mask = (t_star >= t_on_star) & (t_star < t_on_star + delta_star)
    if phase2_mask.any():
        tau_on = t_star[phase2_mask] - t_on_star
        u_on, s_on = _compute_on_phase_solution(tau_on, alpha_on, gamma_star)
        u_star[phase2_mask] = u_on
        s_star[phase2_mask] = s_on

    # Phase 3: Return to off state (t* ≥ t*_on + δ*)
    phase3_mask = t_star >= t_on_star + delta_star
    if phase3_mask.any():
        tau_off = t_star[phase3_mask] - (t_on_star + delta_star)
        u_off, s_off = _compute_off_phase_solution(
            tau_off, alpha_off, alpha_on, gamma_star, delta_star
        )
        u_star[phase3_mask] = u_off
        s_star[phase3_mask] = s_off

    return u_star, s_star


def _compute_on_phase_solution(
    tau_on: torch.Tensor,
    alpha_on: torch.Tensor,
    gamma_star: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute analytical solution for ON phase."""
    # Initial conditions: u*_0 = 1.0, s*_0 = 1.0/γ*
    u_0 = 1.0
    s_0 = 1.0 / gamma_star

    # Analytical solution for ON phase
    exp_tau = torch.exp(-tau_on)
    exp_gamma_tau = torch.exp(-gamma_star * tau_on)

    u_on = alpha_on + (u_0 - alpha_on) * exp_tau

    s_on = (alpha_on / gamma_star +
            (s_0 - alpha_on / gamma_star) * exp_gamma_tau +
            (alpha_on - u_0) * (exp_tau - exp_gamma_tau) / (gamma_star - 1))

    return u_on, s_on


def _compute_off_phase_solution(
    tau_off: torch.Tensor,
    alpha_off: torch.Tensor,
    alpha_on: torch.Tensor,
    gamma_star: torch.Tensor,
    delta_star: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute analytical solution for return to OFF phase."""
    # Initial conditions: endpoint values from ON phase
    u_end, s_end = _compute_on_phase_solution(
        delta_star, alpha_on, gamma_star
    )

    # Analytical solution for OFF phase
    exp_tau = torch.exp(-tau_off)
    exp_gamma_tau = torch.exp(-gamma_star * tau_off)

    u_off = alpha_off + (u_end - alpha_off) * exp_tau

    s_off = (alpha_off / gamma_star +
            (s_end - alpha_off / gamma_star) * exp_gamma_tau +
            (alpha_off - u_end) * (exp_tau - exp_gamma_tau) / (gamma_star - 1))

    return u_off, s_off


@beartype
def plot_prior_predictive_checks(
    model: Any,
    prior_adata: AnnData,
    prior_parameters: Dict[str, torch.Tensor],
    figsize: Tuple[int, int] = (7.5, 5.0),  # Standard width for 8.5x11" with margins
    check_type: str = "prior",
    save_path: Optional[str] = None,
    figure_name: Optional[str] = None,
    create_individual_plots: bool = True,
    combine_individual_pdfs: bool = False,
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
        combine_individual_pdfs: Whether to combine individual PDF plots into a single file

    Returns:
        matplotlib Figure object

    Example:
        >>> fig = plot_prior_predictive_checks(
        ...     model=model,
        ...     prior_adata=adata,
        ...     prior_parameters=params,
        ...     save_path="reports/docs/prior_predictive",
        ...     figure_name="piecewise_activation_prior_checks",
        ...     combine_individual_pdfs=True
        ... )
    """
    # Create individual modular plots if requested
    if create_individual_plots and save_path is not None:
        # Process parameters for plotting compatibility (handle batch dimensions)
        processed_parameters = _process_parameters_for_plotting(prior_parameters)

        # Create plots in logical order with numbered prefixes for proper PDF combination ordering
        plot_parameter_marginals(processed_parameters, check_type, save_path=save_path, file_prefix="02", model=model)
        plot_parameter_relationships(processed_parameters, check_type, save_path=save_path, file_prefix="03", model=model)
        plot_temporal_trajectories(processed_parameters, check_type, save_path=save_path, file_prefix="04")
        plot_temporal_dynamics(prior_adata, check_type, save_path=save_path, file_prefix="05")
        plot_expression_validation(prior_adata, check_type, save_path=save_path, file_prefix="06")
        plot_pattern_analysis(prior_adata, processed_parameters, check_type, save_path=save_path, file_prefix="07")

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
    _plot_umap_time_coordinate(prior_adata, ax2, check_type, model=model)

    ax3 = fig.add_subplot(gs[0, 2])
    _plot_fold_change_distribution(processed_parameters, ax3, check_type, model=model)

    ax4 = fig.add_subplot(gs[0, 3])
    _plot_activation_timing(processed_parameters, ax4, check_type, model=model)

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
        # Use provided name or default, with "01" prefix for proper ordering
        if figure_name is not None:
            name = f"01_{figure_name}"
        else:
            name = f"01_{check_type}_predictive_checks"
        _save_figure(fig, save_path, name)

        # Combine individual PDFs if requested
        if combine_individual_pdfs and create_individual_plots:
            try:
                # Extract seed from figure_name if present, otherwise use default filename
                if figure_name is not None and '_' in figure_name:
                    # Try to extract seed from figure_name (e.g., "piecewise_activation_prior_checks_42")
                    parts = figure_name.split('_')
                    if parts[-1].isdigit():
                        seed_suffix = f"_{parts[-1]}"
                    else:
                        seed_suffix = ""
                else:
                    seed_suffix = ""

                combine_pdfs(
                    pdf_directory=save_path,
                    output_filename=f"combined_{check_type}_predictive_checks{seed_suffix}.pdf",
                    exclude_patterns=["combined_*.pdf"]  # Don't include previous combined files
                )
            except Exception as e:
                print(f"Warning: Could not combine PDFs: {e}")
                print("Individual PDF files are still available in the directory.")

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


def _plot_umap_time_coordinate(adata: AnnData, ax: plt.Axes, check_type: str, model: Optional[Any] = None) -> None:
    """Plot UMAP embedding colored by time coordinate."""

    if 'X_umap' in adata.obsm:
        umap_coords = adata.obsm['X_umap']

        # Look for time coordinate in various possible locations
        time_coord = None
        time_label = 'Time'

        # Check common time coordinate names
        time_keys = ['latent_time', 'velocity_pseudotime', 'dpt_pseudotime',
                    'pseudotime', 'time', 't', 'shared_time']

        # Define appropriate labels for time coordinates (non-LaTeX for colorbar)
        time_coordinate_labels = {
            'latent_time': 'Latent Time',
            'velocity_pseudotime': 'Velocity Pseudotime',
            'dpt_pseudotime': 'DPT Pseudotime',
            'pseudotime': 'Pseudotime',
            'time': 'Time',
            't': 'Time',
            'shared_time': 'Shared Time'
        }

        for key in time_keys:
            if key in adata.obs:
                time_coord = adata.obs[key].values

                # Use appropriate label for colorbar (non-LaTeX)
                time_label = time_coordinate_labels.get(key, 'Time Coordinate')
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

    Also filters out guide-specific parameters that shouldn't be displayed.

    Args:
        parameters: Raw parameters with potential batch dimensions

    Returns:
        Processed parameters suitable for plotting functions
    """
    processed_parameters = {}

    # Define patterns for guide parameters to exclude
    # Be specific to avoid filtering legitimate model parameters like t_loc, t_scale
    guide_param_patterns = [
        'AutoLowRankMultivariateNormal',
        'AutoNormal',
        'AutoDelta',
        'AutoGuide',
        '_latent',
        'auto_',
        'guide_'
    ]

    for key, value in parameters.items():
        # Skip guide-specific parameters
        if any(pattern in key for pattern in guide_param_patterns):
            continue

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

    # Compute t_star if missing but required components are available
    # t_star = T_M_star * max(tilde_t, epsilon) where tilde_t ~ Normal(t_loc, t_scale)
    if ('t_star' not in processed_parameters and
        'T_M_star' in processed_parameters and
        'tilde_t' in processed_parameters):

        T_M_star = processed_parameters['T_M_star']
        tilde_t = processed_parameters['tilde_t']

        # Use same epsilon as in the prior model (default 1e-6)
        t_epsilon = 1e-6

        # Compute t_star: T_M_star * max(tilde_t, epsilon)
        # Handle broadcasting: T_M_star is [num_samples], tilde_t is [num_samples * num_cells]
        if T_M_star.ndim == 1 and tilde_t.ndim == 1:
            # Determine number of cells from tilde_t length
            num_samples = len(T_M_star)
            num_cells = len(tilde_t) // num_samples

            if len(tilde_t) % num_samples == 0:  # Valid division
                # Reshape tilde_t to [num_samples, num_cells] for broadcasting
                tilde_t_reshaped = tilde_t.view(num_samples, num_cells)
                t_star_computed = T_M_star.unsqueeze(-1) * torch.clamp(tilde_t_reshaped, min=t_epsilon)
                # Flatten back to 1D for consistency with other parameters
                processed_parameters['t_star'] = t_star_computed.flatten()

                print(f"ℹ️  Computed missing t_star from T_M_star and tilde_t (shape: {t_star_computed.shape} -> flattened)")
            else:
                print(f"⚠️  Cannot compute t_star: tilde_t length {len(tilde_t)} not divisible by T_M_star length {num_samples}")
        else:
            print(f"⚠️  Cannot compute t_star: unexpected tensor dimensions T_M_star: {T_M_star.shape}, tilde_t: {tilde_t.shape}")

    # Compute t_on_star if missing but required components are available
    # t_on_star = T_M_star * tilde_t_on_star
    if ('t_on_star' not in processed_parameters and
        'T_M_star' in processed_parameters and
        'tilde_t_on_star' in processed_parameters):

        T_M_star = processed_parameters['T_M_star']
        tilde_t_on_star = processed_parameters['tilde_t_on_star']

        # Ensure both tensors are 1D for consistent processing
        if T_M_star.dim() > 1:
            T_M_star = T_M_star.flatten()
        if tilde_t_on_star.dim() > 1:
            tilde_t_on_star = tilde_t_on_star.flatten()

        # Compute t_on_star = T_M_star * tilde_t_on_star
        # Handle broadcasting for gene-level parameters
        num_samples = len(T_M_star)
        num_genes = len(tilde_t_on_star) // num_samples if len(tilde_t_on_star) % num_samples == 0 else None

        if num_genes is not None:
            # Reshape tilde_t_on_star to [num_samples, num_genes] for proper broadcasting
            tilde_t_on_star_reshaped = tilde_t_on_star.view(num_samples, num_genes)
            # Broadcast T_M_star to match: [num_samples, 1] -> [num_samples, num_genes]
            T_M_star_expanded = T_M_star.unsqueeze(-1).expand(-1, num_genes)
            # Element-wise multiplication
            t_on_star_computed = T_M_star_expanded * tilde_t_on_star_reshaped
            # Flatten back to 1D for consistency
            t_on_star_computed = t_on_star_computed.flatten()
        elif len(T_M_star) == len(tilde_t_on_star):
            # Same length - element-wise multiplication
            t_on_star_computed = T_M_star * tilde_t_on_star
        elif len(T_M_star) == 1:
            # Broadcast single T_M_star to all genes
            t_on_star_computed = T_M_star.item() * tilde_t_on_star
        elif len(tilde_t_on_star) == 1:
            # Broadcast single tilde_t_on_star to all samples
            t_on_star_computed = T_M_star * tilde_t_on_star.item()
        else:
            print(f"⚠️  Cannot compute t_on_star: incompatible shapes T_M_star: {T_M_star.shape}, tilde_t_on_star: {tilde_t_on_star.shape}")
            t_on_star_computed = None

        if t_on_star_computed is not None:
            processed_parameters['t_on_star'] = t_on_star_computed
            print(f"ℹ️  Computed missing t_on_star from T_M_star and tilde_t_on_star (shape: {t_on_star_computed.shape})")

    # Compute delta_star if missing but required components are available
    # delta_star = T_M_star * tilde_delta_star
    if ('delta_star' not in processed_parameters and
        'T_M_star' in processed_parameters and
        'tilde_delta_star' in processed_parameters):

        T_M_star = processed_parameters['T_M_star']
        tilde_delta_star = processed_parameters['tilde_delta_star']

        # Ensure both tensors are 1D for consistent processing
        if T_M_star.dim() > 1:
            T_M_star = T_M_star.flatten()
        if tilde_delta_star.dim() > 1:
            tilde_delta_star = tilde_delta_star.flatten()

        # Compute delta_star = T_M_star * tilde_delta_star
        # Handle broadcasting for gene-level parameters
        num_samples = len(T_M_star)
        num_genes = len(tilde_delta_star) // num_samples if len(tilde_delta_star) % num_samples == 0 else None

        if num_genes is not None:
            # Reshape tilde_delta_star to [num_samples, num_genes] for proper broadcasting
            tilde_delta_star_reshaped = tilde_delta_star.view(num_samples, num_genes)
            # Broadcast T_M_star to match: [num_samples, 1] -> [num_samples, num_genes]
            T_M_star_expanded = T_M_star.unsqueeze(-1).expand(-1, num_genes)
            # Element-wise multiplication
            delta_star_computed = T_M_star_expanded * tilde_delta_star_reshaped
            # Flatten back to 1D for consistency
            delta_star_computed = delta_star_computed.flatten()
        elif len(T_M_star) == len(tilde_delta_star):
            # Same length - element-wise multiplication
            delta_star_computed = T_M_star * tilde_delta_star
        elif len(T_M_star) == 1:
            # Broadcast single T_M_star to all genes
            delta_star_computed = T_M_star.item() * tilde_delta_star
        elif len(tilde_delta_star) == 1:
            # Broadcast single tilde_delta_star to all samples
            delta_star_computed = T_M_star * tilde_delta_star.item()
        else:
            print(f"⚠️  Cannot compute delta_star: incompatible shapes T_M_star: {T_M_star.shape}, tilde_delta_star: {tilde_delta_star.shape}")
            delta_star_computed = None

        if delta_star_computed is not None:
            processed_parameters['delta_star'] = delta_star_computed
            print(f"ℹ️  Computed missing delta_star from T_M_star and tilde_delta_star (shape: {delta_star_computed.shape})")

    return processed_parameters


def _plot_parameter_marginals_summary(
    parameters: Dict[str, torch.Tensor],
    ax: plt.Axes,
    check_type: str,
    model: Optional[Any] = None
) -> None:
    """Plot marginal distributions of key parameters (summary version for overview)."""
    from pyrovelocity.plots.parameter_metadata import (
        get_parameter_label,
        infer_component_name_from_parameters,
    )

    # Try to infer component name from all parameters if model not provided
    component_name = None
    if model is None:
        component_name = infer_component_name_from_parameters(parameters)

    # Focus on piecewise activation parameters (use R_on instead of deprecated alpha_on)
    key_params = ['R_on', 't_on_star', 'delta_star', 'gamma_star']

    colors = sns.color_palette("husl", len(key_params))

    for i, param_name in enumerate(key_params):
        if param_name in parameters:
            values = parameters[param_name].flatten().numpy()

            # Get parameter label using new metadata system
            param_label = get_parameter_label(
                param_name=param_name,
                label_type="display",
                model=model,
                component_name=component_name,
                fallback_to_legacy=True
            )

            # Use relative frequency for consistency with main marginals plot
            ax.hist(values, bins=30, alpha=0.6, color=colors[i],
                   label=param_label, density=False,
                   weights=np.ones(len(values)) / len(values))

    ax.set_xlabel('Parameter Value')
    ax.set_ylabel('Relative Frequency')
    ax.set_title(f'{check_type.title()} Parameter Marginals')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)



def _plot_hierarchical_time_structure(
    parameters: Dict[str, torch.Tensor],
    ax: plt.Axes,
    check_type: str,
    model: Optional[Any] = None
) -> None:
    """Plot hierarchical time parameter relationships."""
    from pyrovelocity.plots.parameter_metadata import (
        get_parameter_label,
        infer_component_name_from_parameters,
    )

    # Try to infer component name from all parameters if model not provided
    component_name = None
    if model is None:
        component_name = infer_component_name_from_parameters(parameters)

    # Check for hierarchical time parameters
    time_params = ['T_M_star', 't_loc', 't_scale']
    available_time_params = [p for p in time_params if p in parameters]

    if len(available_time_params) >= 2:
        # Plot T_M_star vs population time spread (t_scale)
        if 'T_M_star' in parameters and 't_scale' in parameters:
            T_M = parameters['T_M_star'].flatten().numpy()
            t_scale = parameters['t_scale'].flatten().numpy()

            ax.scatter(T_M, t_scale, alpha=0.6, s=20, color='purple')

            # Get parameter labels using new metadata system
            T_M_label = get_parameter_label(
                param_name="T_M_star",
                label_type="display",
                model=model,
                component_name=component_name,
                fallback_to_legacy=True
            )
            t_scale_label = get_parameter_label(
                param_name="t_scale",
                label_type="display",
                model=model,
                component_name=component_name,
                fallback_to_legacy=True
            )

            ax.set_xlabel(f'Global Time Scale ({T_M_label})', fontsize=7)
            ax.set_ylabel(f'Population Time Spread ({t_scale_label})', fontsize=7)
            ax.set_title(f'{check_type.title()} Hierarchical Time Structure', fontsize=8)
            ax.tick_params(labelsize=6)  # Reduce tick label size

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

            # Get parameter labels using new metadata system
            t_loc_label = get_parameter_label(
                param_name="t_loc",
                label_type="display",
                model=model,
                component_name=component_name,
                fallback_to_legacy=True
            )
            t_scale_label = get_parameter_label(
                param_name="t_scale",
                label_type="display",
                model=model,
                component_name=component_name,
                fallback_to_legacy=True
            )

            ax.set_xlabel(f'Population Time Location ({t_loc_label})', fontsize=7)
            ax.set_ylabel(f'Population Time Spread ({t_scale_label})', fontsize=7)
            ax.set_title(f'{check_type.title()} Population Time Parameters', fontsize=8)
            ax.tick_params(labelsize=6)  # Reduce tick label size
            ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, 'Hierarchical time parameters\nnot available',
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'{check_type.title()} Hierarchical Time Structure', fontsize=8)

    ax.grid(True, alpha=0.3)


def _plot_fold_change_distribution(
    parameters: Dict[str, torch.Tensor],
    ax: plt.Axes,
    check_type: str,
    model: Optional[Any] = None
) -> None:
    """Plot fold-change distribution using R_on parameter."""
    from pyrovelocity.plots.parameter_metadata import (
        get_parameter_label,
        infer_component_name_from_parameters,
    )

    # Try to infer component name from all parameters if model not provided
    component_name = None
    if model is None:
        component_name = infer_component_name_from_parameters(parameters)

    # Use R_on directly (preferred) or fall back to alpha_on/alpha_off ratio
    if 'R_on' in parameters:
        fold_change = parameters['R_on'].flatten().numpy()
        param_source = "R_on"
    elif 'alpha_off' in parameters and 'alpha_on' in parameters:
        alpha_off = parameters['alpha_off'].flatten()
        alpha_on = parameters['alpha_on'].flatten()
        fold_change = (alpha_on / alpha_off).numpy()
        param_source = "alpha_ratio"
    else:
        ax.text(0.5, 0.5, 'Fold-change parameters\nnot available',
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'{check_type.title()} Fold-change Distribution', fontsize=8)
        ax.grid(True, alpha=0.3)
        return

    # Use relative frequency for consistency
    ax.hist(fold_change, bins=50, alpha=0.7, color='skyblue', density=False,
           weights=np.ones(len(fold_change)) / len(fold_change))
    ax.axvline(fold_change.mean(), color='red', linestyle='--',
              label=f'Mean: {fold_change.mean():.1f}')
    ax.axvline(3.3, color='orange', linestyle=':', label='Min threshold: 3.3')
    ax.axvline(7.5, color='green', linestyle=':', label='Activation threshold: 7.5')

    # Get parameter label using new metadata system
    if param_source == "R_on":
        param_label = get_parameter_label(
            param_name="R_on",
            label_type="display",
            model=model,
            component_name=component_name,
            fallback_to_legacy=True
        )
        xlabel = f'Fold-change ({param_label})'
    else:
        # Legacy fallback for alpha_on/alpha_off ratio
        alpha_on_label = get_parameter_label(
            param_name="alpha_on",
            label_type="display",
            model=model,
            component_name=component_name,
            fallback_to_legacy=True
        )
        alpha_off_label = get_parameter_label(
            param_name="alpha_off",
            label_type="display",
            model=model,
            component_name=component_name,
            fallback_to_legacy=True
        )
        xlabel = f'Fold-change ({alpha_on_label} / {alpha_off_label})'

    ax.set_xlabel(xlabel, fontsize=7)
    ax.set_ylabel('Relative Frequency', fontsize=7)
    ax.set_title(f'{check_type.title()} Fold-change Distribution', fontsize=8)
    ax.tick_params(labelsize=6)  # Reduce tick label size
    ax.legend(fontsize=8)
    ax.set_xlim(0, min(100, fold_change.max()))
    ax.grid(True, alpha=0.3)


def _plot_activation_timing(
    parameters: Dict[str, torch.Tensor],
    ax: plt.Axes,
    check_type: str,
    model: Optional[Any] = None
) -> None:
    """Plot activation timing and duration distributions."""
    from pyrovelocity.plots.parameter_metadata import (
        get_parameter_label,
        infer_component_name_from_parameters,
    )

    # Try to infer component name from all parameters if model not provided
    component_name = None
    if model is None:
        component_name = infer_component_name_from_parameters(parameters)

    if 't_on_star' in parameters and 'delta_star' in parameters:
        t_on = parameters['t_on_star'].flatten().numpy()
        delta = parameters['delta_star'].flatten().numpy()

        ax.scatter(t_on, delta, alpha=0.6, s=20, color='purple')

        # Get parameter labels using new metadata system
        t_on_label = get_parameter_label(
            param_name="t_on_star",
            label_type="display",
            model=model,
            component_name=component_name,
            fallback_to_legacy=True
        )
        delta_label = get_parameter_label(
            param_name="delta_star",
            label_type="display",
            model=model,
            component_name=component_name,
            fallback_to_legacy=True
        )

        ax.set_xlabel(f'Activation Onset ({t_on_label})', fontsize=7)
        ax.set_ylabel(f'Activation Duration ({delta_label})', fontsize=7)
        ax.set_title(f'{check_type.title()} Activation Timing', fontsize=8)
        ax.tick_params(labelsize=6)  # Reduce tick label size

        # Add pattern boundaries
        ax.axhline(0.35, color='red', linestyle='--', alpha=0.7,
                  label='Transient/Sustained boundary')
        ax.axvline(0.3, color='orange', linestyle='--', alpha=0.7,
                  label='Early/Late activation')
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, 'Timing parameters\nnot available',
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'{check_type.title()} Activation Timing', fontsize=8)

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
        # Compute velocity using the correct piecewise activation model formula
        if 'unspliced' in adata.layers and 'spliced' in adata.layers:
            u = adata.layers['unspliced']
            s = adata.layers['spliced']

            # For the dimensionless piecewise activation model: ds*/dt* = u* - γ*s*
            # We need gamma_star values, but if not available, use a reasonable approximation
            if 'gamma_star' in adata.var:
                gamma_star = adata.var['gamma_star'].values
                # Compute velocity per gene: ds*/dt* = u* - γ*s*
                velocity_per_gene = u - gamma_star[np.newaxis, :] * s
            else:
                # Use a typical gamma_star value (~1.0) as approximation
                gamma_star_approx = 1.0
                velocity_per_gene = u - gamma_star_approx * s

            # Take mean velocity magnitude across genes for each cell
            velocity_magnitudes = np.mean(np.abs(velocity_per_gene), axis=1)

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
            ax.text(0.5, 0.5, 'Velocity data\nnot available',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{check_type.title()} Velocity Magnitudes')

    ax.grid(True, alpha=0.3)





def _create_temporal_dynamics_figure(
    number_of_genes: int,
    figsize: Optional[Tuple[int, int]] = None
) -> Tuple[plt.Figure, Dict[str, plt.Axes]]:
    """Create figure and axes dict using rainbow plot gridspec pattern."""
    from matplotlib.gridspec import GridSpec

    # Define number of horizontal panels
    horizontal_panels = 5  # gene_label, phase, dynamics, predictive, observed

    # Calculate figure size using rainbow plot pattern
    if figsize is None:
        subplot_height = 0.9
        subplot_width = 1.5 * subplot_height * horizontal_panels
        figsize = (subplot_width, subplot_height * number_of_genes)

    fig = plt.figure(figsize=figsize)

    # Create gridspec with proper width ratios and spacing like rainbow plot
    gs = GridSpec(
        nrows=number_of_genes + 1,  # Add extra row for titles
        ncols=horizontal_panels,
        figure=fig,
        width_ratios=[
            0.21,  # Gene label column (same as rainbow plot)
            1,     # Phase portrait
            1,     # Dynamics
            1,     # Predictive UMAP
            1,     # Observed UMAP
        ],
        height_ratios=[0.15] + [1] * number_of_genes,  # Small title row + gene rows
        wspace=0.3,   # Increased spacing to prevent axis label overlap
        hspace=0.25,  # Increased vertical spacing between gene rows
    )

    axes_dict = {}

    # Create title row
    for col, title in enumerate(['', r'$(u, s)$ phase space', 'Spliced dynamics', 'Predictive spliced', 'Observed spliced']):
        if col == 0:
            continue  # Skip gene label column for titles
        title_ax = fig.add_subplot(gs[0, col])
        title_ax.text(0.5, 0.5, title, ha='center', va='center',
                     transform=title_ax.transAxes, fontsize=8, weight='bold')
        title_ax.axis('off')

    # Create gene rows
    for n in range(number_of_genes):
        row = n + 1  # Offset by 1 for title row
        axes_dict[f"gene_{n}"] = fig.add_subplot(gs[row, 0])
        axes_dict[f"gene_{n}"].axis("off")
        axes_dict[f"phase_{n}"] = fig.add_subplot(gs[row, 1])
        axes_dict[f"dynamics_{n}"] = fig.add_subplot(gs[row, 2])
        axes_dict[f"predictive_{n}"] = fig.add_subplot(gs[row, 3])
        axes_dict[f"observed_{n}"] = fig.add_subplot(gs[row, 4])

    return fig, axes_dict


def _plot_gene_phase_portrait_rainbow(
    adata: AnnData,
    axes_dict: Dict[str, plt.Axes],
    n: int,
    gene_idx: int,
    gene_name: str,
    check_type: str,
    total_genes: int
) -> None:
    """Plot phase portrait (u,s) for a single gene using rainbow plot style."""
    if 'unspliced' in adata.layers and 'spliced' in adata.layers:
        u_gene = adata.layers['unspliced'][:, gene_idx]
        s_gene = adata.layers['spliced'][:, gene_idx]

        # Color by latent time if available, otherwise use a single color
        time_col = None
        for col in ['latent_time', 'cell_time', 't_star']:
            if col in adata.obs:
                time_col = col
                break

        if time_col is not None:
            c = adata.obs[time_col]
            axes_dict[f"phase_{n}"].scatter(s_gene, u_gene, c=c, cmap='viridis', alpha=0.6, s=3, edgecolors='none')
        else:
            axes_dict[f"phase_{n}"].scatter(s_gene, u_gene, alpha=0.6, s=3, color='steelblue', edgecolors='none')
    else:
        axes_dict[f"phase_{n}"].text(0.5, 0.5, 'Expression data\nnot available',
               ha='center', va='center', transform=axes_dict[f"phase_{n}"].transAxes)

    axes_dict[f"phase_{n}"].grid(True, alpha=0.3)


def _plot_gene_spliced_dynamics_rainbow(
    adata: AnnData,
    axes_dict: Dict[str, plt.Axes],
    n: int,
    gene_idx: int,
    gene_name: str,
    check_type: str,
    total_genes: int
) -> None:
    """Plot spliced expression dynamics over time using rainbow plot style."""
    # Find available time column
    time_col = None
    for col in ['latent_time', 'cell_time', 't_star']:
        if col in adata.obs:
            time_col = col
            break

    if 'spliced' in adata.layers and time_col is not None:
        s_gene = adata.layers['spliced'][:, gene_idx]
        time = adata.obs[time_col]

        # Sort by time for better visualization
        sort_idx = np.argsort(time)
        time_sorted = time.iloc[sort_idx] if hasattr(time, 'iloc') else time[sort_idx]
        s_sorted = s_gene[sort_idx]

        # Color by clusters if available
        cluster_col = None
        for col in ['leiden', 'clusters', 'louvain']:
            if col in adata.obs:
                cluster_col = col
                break

        if cluster_col is not None:
            clusters = adata.obs[cluster_col].iloc[sort_idx] if hasattr(adata.obs[cluster_col], 'iloc') else adata.obs[cluster_col][sort_idx]
            unique_clusters = np.unique(clusters)
            colors = plt.cm.Set1(np.linspace(0, 1, len(unique_clusters)))

            for i, cluster in enumerate(unique_clusters):
                mask = clusters == cluster
                axes_dict[f"dynamics_{n}"].scatter(time_sorted[mask], s_sorted[mask],
                          alpha=0.6, s=3, color=colors[i], edgecolors='none')
        else:
            axes_dict[f"dynamics_{n}"].scatter(time_sorted, s_sorted, alpha=0.6, s=3, color='steelblue', edgecolors='none')
    else:
        axes_dict[f"dynamics_{n}"].text(0.5, 0.5, 'Time data\nnot available',
               ha='center', va='center', transform=axes_dict[f"dynamics_{n}"].transAxes)

    axes_dict[f"dynamics_{n}"].grid(True, alpha=0.3)


def _plot_gene_predictive_umap_rainbow(
    adata: AnnData,
    axes_dict: Dict[str, plt.Axes],
    n: int,
    gene_idx: int,
    gene_name: str,
    check_type: str,
    basis: str = "umap"
) -> None:
    """Plot predictive spliced expression in UMAP space using rainbow plot style."""
    from pyrovelocity.plots._common import set_colorbar

    if f'X_{basis}' in adata.obsm and 'spliced' in adata.layers:
        coords = adata.obsm[f'X_{basis}']
        s_gene = adata.layers['spliced'][:, gene_idx]

        # Use log-transformed expression for predictive data (same scale as observed)
        s_gene_log = np.log1p(s_gene)  # log(1 + x) to handle zeros

        # Create scatter plot with log gene expression as color
        im = axes_dict[f"predictive_{n}"].scatter(
            coords[:, 0], coords[:, 1],
            c=s_gene_log, cmap='cividis',
            alpha=0.8, s=3, edgecolors='none'
        )

        # Add colorbar using rainbow plot style
        set_colorbar(
            im,
            axes_dict[f"predictive_{n}"],
            labelsize=5,
            fig=axes_dict[f"predictive_{n}"].figure,
            rainbow=True,
        )

        axes_dict[f"predictive_{n}"].axis('off')
    else:
        axes_dict[f"predictive_{n}"].text(0.5, 0.5, f'{basis.upper()} or\nexpression data\nnot available',
               ha='center', va='center', transform=axes_dict[f"predictive_{n}"].transAxes)
        axes_dict[f"predictive_{n}"].axis('off')


def _plot_gene_observed_umap_rainbow(
    adata: AnnData,
    axes_dict: Dict[str, plt.Axes],
    n: int,
    gene_idx: int,
    gene_name: str,
    check_type: str,
    basis: str = "umap"
) -> None:
    """Plot observed spliced expression in UMAP space using rainbow plot style."""
    from pyrovelocity.plots._common import set_colorbar

    if f'X_{basis}' in adata.obsm and 'spliced' in adata.layers:
        coords = adata.obsm[f'X_{basis}']
        s_gene = adata.layers['spliced'][:, gene_idx]

        # Use log-transformed expression for observed data (same scale as predictive)
        s_gene_log = np.log1p(s_gene)  # log(1 + x) to handle zeros

        # Create scatter plot with log gene expression as color
        im = axes_dict[f"observed_{n}"].scatter(
            coords[:, 0], coords[:, 1],
            c=s_gene_log, cmap='cividis',
            alpha=0.8, s=3, edgecolors='none'
        )

        # Add colorbar using rainbow plot style
        set_colorbar(
            im,
            axes_dict[f"observed_{n}"],
            labelsize=5,
            fig=axes_dict[f"observed_{n}"].figure,
            rainbow=True,
        )

        axes_dict[f"observed_{n}"].axis('off')
    else:
        axes_dict[f"observed_{n}"].text(0.5, 0.5, f'{basis.upper()} or\nexpression data\nnot available',
               ha='center', va='center', transform=axes_dict[f"observed_{n}"].transAxes)
        axes_dict[f"observed_{n}"].axis('off')


def _set_temporal_dynamics_labels(
    axes_dict: Dict[str, plt.Axes],
    n: int,
    gene_name: str,
    total_genes: int,
    default_fontsize: int
) -> None:
    """Set labels for temporal dynamics plot using rainbow plot style."""
    # Set gene name in the gene label column
    axes_dict[f"gene_{n}"].text(
        0.0, 0.5, gene_name[:7],
        transform=axes_dict[f"gene_{n}"].transAxes,
        rotation=0, va='center', ha='center',
        fontsize=default_fontsize, weight='normal'
    )

    # Set axis labels only for bottom row (like rainbow plot)
    if n == total_genes - 1:
        # Set x-axis labels
        axes_dict[f"phase_{n}"].set_xlabel(
            r'spliced, $\hat{\mu}(s)$',
            loc="left",
            labelpad=0.7,
            fontsize=default_fontsize
        )
        axes_dict[f"dynamics_{n}"].set_xlabel(
            r'shared time, $\hat{\mu}(t)$',
            loc="left",
            labelpad=0.7,
            fontsize=default_fontsize
        )

        # Set y-axis labels on the RIGHT side (like rainbow plot)
        axes_dict[f"phase_{n}"].set_ylabel(
            r'unspliced, $\hat{\mu}(u)$',
            loc="bottom",
            labelpad=0.7,
            fontsize=default_fontsize
        )
        axes_dict[f"phase_{n}"].yaxis.set_label_position("right")

        axes_dict[f"dynamics_{n}"].set_ylabel(
            r'spliced, $\hat{\mu}(s)$',
            loc="bottom",
            labelpad=0.7,
            fontsize=default_fontsize
        )
        axes_dict[f"dynamics_{n}"].yaxis.set_label_position("right")
    else:
        # Remove axis labels for non-bottom rows
        axes_dict[f"phase_{n}"].set_xlabel('')
        axes_dict[f"phase_{n}"].set_ylabel('')
        axes_dict[f"dynamics_{n}"].set_xlabel('')
        axes_dict[f"dynamics_{n}"].set_ylabel('')

    # Set tick parameters
    axes_dict[f"phase_{n}"].tick_params(labelsize=default_fontsize * 0.6)
    axes_dict[f"dynamics_{n}"].tick_params(labelsize=default_fontsize * 0.6)


def _set_temporal_dynamics_aspect(axes_dict: Dict[str, plt.Axes]) -> None:
    """Set aspect ratios for temporal dynamics plot using rainbow plot style."""
    # Set equal aspect for UMAP plots (like rainbow plot)
    for key in axes_dict.keys():
        if 'predictive_' in key or 'observed_' in key:
            axes_dict[key].set_aspect('equal')
        elif 'phase_' in key:
            # Phase plots can have auto aspect
            axes_dict[key].set_aspect('auto')
        elif 'dynamics_' in key:
            # Dynamics plots can have auto aspect
            axes_dict[key].set_aspect('auto')


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

        # Format pattern names for display
        formatted_patterns = [_format_pattern_name(pattern) for pattern in patterns]

        ax.pie(counts, labels=formatted_patterns, colors=colors,
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
    """
    Classify expression patterns from parameter samples using soft scoring approach.

    This function uses the same pattern classification logic as the prior hyperparameter
    calibration script to ensure consistency between validation scripts.

    Updated to use the simplified 3-pattern classification system:
    - pre_activation: All patterns where activation occurred before observation window
    - transient: Complete activation-decay cycle within observation window
    - sustained: Net increase over observation window (includes late activation)
    """
    # Check for required parameters - now using relative temporal parameters
    required_params = ['R_on', 'tilde_t_on_star', 'tilde_delta_star']
    if not all(key in parameters for key in required_params):
        # Fallback to absolute parameters if relative ones not available
        required_params = ['R_on', 't_on_star', 'delta_star']
        if not all(key in parameters for key in required_params):
            return {}
        use_relative_params = False
    else:
        use_relative_params = True

    # Extract parameter arrays
    R_on = parameters['R_on'].flatten()

    if use_relative_params:
        # Use relative temporal parameters (preferred)
        tilde_t_on_star = parameters['tilde_t_on_star'].flatten()
        tilde_delta_star = parameters['tilde_delta_star'].flatten()
    else:
        # Fallback to absolute parameters
        t_on_star = parameters['t_on_star'].flatten()
        delta_star = parameters['delta_star'].flatten()

    n_samples = len(R_on)

    # Updated pattern constraints matching prior-hyperparameter-calibration.py
    # These work with relative temporal parameters (tilde_t_on_star, tilde_delta_star)
    pattern_constraints = {
        'pre_activation': {
            # All patterns where activation occurred before observation window
            # Results in observable decay-only dynamics from activated steady-state
            'tilde_t_on_star': ('<', 0.0),      # Relative activation before observation starts
            'R_on': ('>', 2.0),                  # Moderate to strong fold change
        },
        'transient': {
            # Complete activation-decay cycle within observation window
            # Activation early enough and pulse short enough to see full cycle
            'tilde_t_on_star': ('>', 0.0),      # Relative activation within observation window
            'tilde_t_on_star_upper': ('<', 0.5), # Early enough to complete cycle (50% of timeline)
            'tilde_delta_star': ('<', 0.4),     # Short enough pulse to see decay (40% of timeline)
            'R_on': ('>', 2.0),                  # Sufficient fold change to observe
        },
        'sustained': {
            # Net increase over observation window (includes late activation)
            # Either long pulse or late activation that doesn't complete decay
            'tilde_t_on_star': ('>', 0.0),      # Relative activation within observation window
            'tilde_t_on_star_upper': ('<', 0.3), # Early activation onset (30% of timeline)
            'tilde_delta_star': ('>', 0.5),     # Long activation duration (50% of timeline)
            'R_on': ('>', 2.0),                  # Strong fold change
        }
    }

    # Soft scoring function (same as in calibration script)
    def sigmoid_score(value: float, threshold: float, direction: str, steepness: float = 5.0) -> float:
        """Compute soft score using sigmoid function."""
        if direction == '>':
            return torch.sigmoid(torch.tensor(steepness * (value - threshold))).item()
        else:  # direction == '<'
            return torch.sigmoid(torch.tensor(steepness * (threshold - value))).item()

    # Compute pattern scores for each sample
    pattern_scores = {}
    for pattern in pattern_constraints.keys():
        pattern_scores[pattern] = torch.zeros(n_samples)

    for i in range(n_samples):
        for pattern, constraints in pattern_constraints.items():
            scores = []

            if use_relative_params:
                # Use relative temporal parameters (preferred approach)
                if pattern == 'pre_activation':
                    scores = [
                        sigmoid_score(R_on[i], 2.0, '>'),
                        sigmoid_score(tilde_t_on_star[i], 0.0, '<')
                    ]
                elif pattern == 'transient':
                    scores = [
                        sigmoid_score(R_on[i], 2.0, '>'),
                        sigmoid_score(tilde_t_on_star[i], 0.0, '>'),
                        sigmoid_score(tilde_t_on_star[i], 0.5, '<'),
                        sigmoid_score(tilde_delta_star[i], 0.4, '<')
                    ]
                elif pattern == 'sustained':
                    scores = [
                        sigmoid_score(R_on[i], 2.0, '>'),
                        sigmoid_score(tilde_t_on_star[i], 0.0, '>'),
                        sigmoid_score(tilde_t_on_star[i], 0.3, '<'),
                        sigmoid_score(tilde_delta_star[i], 0.5, '>')
                    ]
            else:
                # Fallback to absolute parameters with adjusted thresholds
                # Note: These thresholds assume T_M_star ~ 50-60 for scaling
                if pattern == 'pre_activation':
                    scores = [
                        sigmoid_score(R_on[i], 2.0, '>'),
                        sigmoid_score(t_on_star[i], 0.0, '<')
                    ]
                elif pattern == 'transient':
                    scores = [
                        sigmoid_score(R_on[i], 2.0, '>'),
                        sigmoid_score(t_on_star[i], 0.0, '>'),
                        sigmoid_score(t_on_star[i], 25.0, '<'),  # 0.5 * 50 (typical T_M_star)
                        sigmoid_score(delta_star[i], 20.0, '<')  # 0.4 * 50 (typical T_M_star)
                    ]
                elif pattern == 'sustained':
                    scores = [
                        sigmoid_score(R_on[i], 2.0, '>'),
                        sigmoid_score(t_on_star[i], 0.0, '>'),
                        sigmoid_score(t_on_star[i], 15.0, '<'),  # 0.3 * 50 (typical T_M_star)
                        sigmoid_score(delta_star[i], 25.0, '>')  # 0.5 * 50 (typical T_M_star)
                    ]

            # Compute geometric mean of scores
            if scores:
                pattern_scores[pattern][i] = torch.prod(torch.tensor(scores)) ** (1.0 / len(scores))

    # Assign each sample to the pattern with highest score
    assignments = torch.zeros(n_samples, dtype=torch.long)
    pattern_names = list(pattern_constraints.keys())

    for i in range(n_samples):
        scores = [pattern_scores[pattern][i] for pattern in pattern_names]
        assignments[i] = torch.argmax(torch.tensor(scores))

    # Count pattern assignments
    pattern_counts = {}
    for i, pattern in enumerate(pattern_names):
        count = (assignments == i).sum().item()
        if count > 0:
            pattern_counts[pattern] = count

    # Add unknown category for any unassigned samples (shouldn't happen with soft scoring)
    total_assigned = sum(pattern_counts.values())
    if total_assigned < n_samples:
        pattern_counts['unknown'] = n_samples - total_assigned

    return pattern_counts


# Alias for posterior predictive checks (same function, different check_type)
@beartype
def plot_posterior_predictive_checks(
    model: Any,
    posterior_adata: AnnData,
    posterior_parameters: Dict[str, torch.Tensor],
    figsize: Tuple[int, int] = (15, 10),
    save_path: Optional[str] = None,
    figure_name: Optional[str] = None,
    create_individual_plots: bool = True,
    combine_individual_pdfs: bool = False,
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
        figure_name=figure_name,
        create_individual_plots=create_individual_plots,
        combine_individual_pdfs=combine_individual_pdfs,
    )
