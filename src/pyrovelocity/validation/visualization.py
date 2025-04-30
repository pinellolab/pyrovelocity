"""
PyroVelocity visualization tools.

This module provides tools for visualizing the comparison between different
implementations of PyroVelocity (legacy, modular, and JAX).

The tools include:
- plot_parameter_comparison: Plot parameter comparison results
- plot_velocity_comparison: Plot velocity comparison results
- plot_uncertainty_comparison: Plot uncertainty comparison results
- plot_performance_comparison: Plot performance comparison results
- plot_parameter_distributions: Plot parameter distributions
- plot_velocity_vector_field: Plot velocity vector field
- plot_uncertainty_heatmap: Plot uncertainty heatmap
- plot_performance_radar: Plot performance radar chart
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from beartype import beartype


@beartype
def plot_parameter_comparison(
    parameter_comparison: Dict[str, Dict[str, Dict[str, float]]],
    figsize: Tuple[int, int] = (12, 8),
    metrics: Optional[List[str]] = None,
) -> plt.Figure:
    """
    Plot parameter comparison results.

    Args:
        parameter_comparison: Dictionary of parameter comparison results
        figsize: Figure size
        metrics: List of metrics to plot (default: all)

    Returns:
        Matplotlib figure
    """
    # Get parameters and metrics
    parameters = list(parameter_comparison.keys())
    if metrics is None:
        # Get all metrics from the first parameter and comparison
        first_param = parameters[0]
        first_comp = list(parameter_comparison[first_param].keys())[0]
        metrics = list(parameter_comparison[first_param][first_comp].keys())

    # Create figure
    fig, axes = plt.subplots(len(metrics), len(parameters), figsize=figsize)

    # Adjust axes for single parameter or metric
    if len(parameters) == 1:
        axes = axes.reshape(-1, 1)
    if len(metrics) == 1:
        axes = axes.reshape(1, -1)

    # Plot comparison results
    for i, param in enumerate(parameters):
        for j, metric in enumerate(metrics):
            # Get axis
            ax = axes[j, i]

            # Get comparison results for this parameter and metric
            comparisons = []
            values = []
            for comp, results in parameter_comparison[param].items():
                if metric in results:
                    comparisons.append(comp)
                    values.append(results[metric])

            # Plot bar chart
            ax.bar(comparisons, values)

            # Set title and labels
            if j == 0:
                ax.set_title(param)
            if i == 0:
                ax.set_ylabel(metric)

            # Rotate x-tick labels
            ax.set_xticklabels(comparisons, rotation=45, ha="right")

    # Adjust layout
    plt.tight_layout()

    return fig


@beartype
def plot_velocity_comparison(
    velocity_comparison: Dict[str, Dict[str, Any]],
    figsize: Tuple[int, int] = (10, 6),
    metrics: Optional[List[str]] = None,
) -> plt.Figure:
    """
    Plot velocity comparison results.

    Args:
        velocity_comparison: Dictionary of velocity comparison results
        figsize: Figure size
        metrics: List of metrics to plot (default: all)

    Returns:
        Matplotlib figure
    """
    # Get comparisons and metrics
    comparisons = list(velocity_comparison.keys())

    # Check if there are any error messages in the comparison results
    has_errors = False
    for comp in comparisons:
        if "error" in velocity_comparison[comp]:
            has_errors = True
            break

    if has_errors:
        # Create a figure to display error messages
        fig, ax = plt.subplots(figsize=figsize)

        # Display error messages
        ax.text(0.5, 0.5, "Errors in comparison results:", ha="center", va="center", fontsize=12, fontweight="bold")
        y_pos = 0.4
        for comp in comparisons:
            if "error" in velocity_comparison[comp]:
                error_msg = f"{comp}: {velocity_comparison[comp]['error']}"
                ax.text(0.5, y_pos, error_msg, ha="center", va="center", fontsize=10)
                y_pos -= 0.1

                # Also print shape information if available
                if "velocity1_shape" in velocity_comparison[comp] and "velocity2_shape" in velocity_comparison[comp]:
                    shape_msg = f"Shapes: {velocity_comparison[comp]['velocity1_shape']} vs {velocity_comparison[comp]['velocity2_shape']}"
                    ax.text(0.5, y_pos, shape_msg, ha="center", va="center", fontsize=10)
                    y_pos -= 0.1

        # Remove axis ticks
        ax.set_xticks([])
        ax.set_yticks([])

        return fig

    # If no errors, proceed with normal plotting
    if metrics is None:
        # Get all metrics from the first comparison
        first_comp = comparisons[0]
        # Filter out non-numeric metrics
        metrics = [k for k in velocity_comparison[first_comp].keys()
                  if isinstance(velocity_comparison[first_comp][k], (int, float))]

    # Create figure
    fig, axes = plt.subplots(1, len(metrics), figsize=figsize)

    # Adjust axes for single metric
    if len(metrics) == 1:
        axes = [axes]

    # Plot comparison results
    for i, metric in enumerate(metrics):
        # Get axis
        ax = axes[i]

        # Get values for this metric
        values = []
        for comp in comparisons:
            if metric in velocity_comparison[comp] and isinstance(velocity_comparison[comp][metric], (int, float)):
                values.append(velocity_comparison[comp][metric])
            else:
                values.append(0)  # Use 0 as a placeholder for missing or non-numeric values

        # Plot bar chart
        ax.bar(comparisons, values)

        # Set title and labels
        ax.set_title(metric)

        # Rotate x-tick labels
        ax.set_xticklabels(comparisons, rotation=45, ha="right")

    # Adjust layout
    plt.tight_layout()

    return fig


@beartype
def plot_uncertainty_comparison(
    uncertainty_comparison: Dict[str, Dict[str, Any]],
    figsize: Tuple[int, int] = (10, 6),
    metrics: Optional[List[str]] = None,
) -> plt.Figure:
    """
    Plot uncertainty comparison results.

    Args:
        uncertainty_comparison: Dictionary of uncertainty comparison results
        figsize: Figure size
        metrics: List of metrics to plot (default: all)

    Returns:
        Matplotlib figure
    """
    # Get comparisons and metrics
    comparisons = list(uncertainty_comparison.keys())

    # Check if there are any error messages in the comparison results
    has_errors = False
    for comp in comparisons:
        if "error" in uncertainty_comparison[comp]:
            has_errors = True
            break

    if has_errors:
        # Create a figure to display error messages
        fig, ax = plt.subplots(figsize=figsize)

        # Display error messages
        ax.text(0.5, 0.5, "Errors in comparison results:", ha="center", va="center", fontsize=12, fontweight="bold")
        y_pos = 0.4
        for comp in comparisons:
            if "error" in uncertainty_comparison[comp]:
                error_msg = f"{comp}: {uncertainty_comparison[comp]['error']}"
                ax.text(0.5, y_pos, error_msg, ha="center", va="center", fontsize=10)
                y_pos -= 0.1

                # Also print shape information if available
                if "shape1" in uncertainty_comparison[comp] and "shape2" in uncertainty_comparison[comp]:
                    shape_msg = f"Shapes: {uncertainty_comparison[comp]['shape1']} vs {uncertainty_comparison[comp]['shape2']}"
                    ax.text(0.5, y_pos, shape_msg, ha="center", va="center", fontsize=10)
                    y_pos -= 0.1

        # Remove axis ticks
        ax.set_xticks([])
        ax.set_yticks([])

        return fig

    # If no errors, proceed with normal plotting
    if metrics is None:
        # Get all metrics from the first comparison
        first_comp = comparisons[0]
        # Filter out non-numeric metrics
        metrics = [k for k in uncertainty_comparison[first_comp].keys()
                  if isinstance(uncertainty_comparison[first_comp][k], (int, float))]

    # Create figure
    fig, axes = plt.subplots(1, len(metrics), figsize=figsize)

    # Adjust axes for single metric
    if len(metrics) == 1:
        axes = [axes]

    # Plot comparison results
    for i, metric in enumerate(metrics):
        # Get axis
        ax = axes[i]

        # Get values for this metric
        values = []
        for comp in comparisons:
            if metric in uncertainty_comparison[comp] and isinstance(uncertainty_comparison[comp][metric], (int, float)):
                values.append(uncertainty_comparison[comp][metric])
            else:
                values.append(0)  # Use 0 as a placeholder for missing or non-numeric values

        # Plot bar chart
        ax.bar(comparisons, values)

        # Set title and labels
        ax.set_title(metric)

        # Rotate x-tick labels
        ax.set_xticklabels(comparisons, rotation=45, ha="right")

    # Adjust layout
    plt.tight_layout()

    return fig


@beartype
def plot_performance_comparison(
    performance_comparison: Dict[str, Dict[str, Any]],
    figsize: Tuple[int, int] = (10, 6),
    metrics: Optional[List[str]] = None,
) -> plt.Figure:
    """
    Plot performance comparison results.

    Args:
        performance_comparison: Dictionary of performance comparison results
        figsize: Figure size
        metrics: List of metrics to plot (default: all)

    Returns:
        Matplotlib figure
    """
    # Get comparisons and metrics
    comparisons = list(performance_comparison.keys())

    # Check if there are any error messages in the comparison results
    has_errors = False
    for comp in comparisons:
        if "error" in performance_comparison[comp]:
            has_errors = True
            break

    if has_errors:
        # Create a figure to display error messages
        fig, ax = plt.subplots(figsize=figsize)

        # Display error messages
        ax.text(0.5, 0.5, "Errors in comparison results:", ha="center", va="center", fontsize=12, fontweight="bold")
        y_pos = 0.4
        for comp in comparisons:
            if "error" in performance_comparison[comp]:
                error_msg = f"{comp}: {performance_comparison[comp]['error']}"
                ax.text(0.5, y_pos, error_msg, ha="center", va="center", fontsize=10)
                y_pos -= 0.1

        # Remove axis ticks
        ax.set_xticks([])
        ax.set_yticks([])

        return fig

    # If no errors, proceed with normal plotting
    if metrics is None:
        # Get all metrics from the first comparison
        first_comp = comparisons[0]
        # Filter out non-numeric metrics
        metrics = [k for k in performance_comparison[first_comp].keys()
                  if isinstance(performance_comparison[first_comp][k], (int, float))]

    # Create figure
    fig, axes = plt.subplots(1, len(metrics), figsize=figsize)

    # Adjust axes for single metric
    if len(metrics) == 1:
        axes = [axes]

    # Plot comparison results
    for i, metric in enumerate(metrics):
        # Get axis
        ax = axes[i]

        # Get values for this metric
        values = []
        for comp in comparisons:
            if metric in performance_comparison[comp] and isinstance(performance_comparison[comp][metric], (int, float)):
                values.append(performance_comparison[comp][metric])
            else:
                values.append(0)  # Use 0 as a placeholder for missing or non-numeric values

        # Plot bar chart
        ax.bar(comparisons, values)

        # Set title and labels
        ax.set_title(metric)

        # Add horizontal line at y=1.0
        ax.axhline(y=1.0, color="r", linestyle="--")

        # Rotate x-tick labels
        ax.set_xticklabels(comparisons, rotation=45, ha="right")

    # Adjust layout
    plt.tight_layout()

    return fig


@beartype
def plot_parameter_distributions(
    results: Dict[str, Dict[str, Any]],
    parameter: str,
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """
    Plot parameter distributions for different implementations.

    Args:
        results: Dictionary of validation results
        parameter: Parameter to plot
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot parameter distributions
    for impl, impl_results in results.items():
        # Check that posterior samples are available
        if "posterior_samples" not in impl_results:
            continue

        # Check that parameter is available
        if parameter not in impl_results["posterior_samples"]:
            continue

        # Get parameter values
        param_values = impl_results["posterior_samples"][parameter]

        # Convert to numpy array
        if isinstance(param_values, np.ndarray):
            pass
        elif isinstance(param_values, list):
            param_values = np.array(param_values)
        elif hasattr(param_values, "detach") and hasattr(param_values, "cpu") and hasattr(param_values, "numpy"):
            # PyTorch tensor
            param_values = param_values.detach().cpu().numpy()
        else:
            # Try to convert to numpy array
            param_values = np.array(param_values)

        # Flatten array
        param_values = param_values.flatten()

        # Plot distribution
        sns.kdeplot(param_values, label=impl, ax=ax)

    # Set title and labels
    ax.set_title(f"{parameter} Distribution")
    ax.set_xlabel(parameter)
    ax.set_ylabel("Density")

    # Add legend
    ax.legend()

    return fig


@beartype
def plot_velocity_vector_field(
    results: Dict[str, Dict[str, Any]],
    coordinates: np.ndarray,
    figsize: Tuple[int, int] = (15, 5),
    resample_method: str = "nearest",
) -> plt.Figure:
    """
    Plot velocity vector field for different implementations.

    This function plots velocity vector fields for different implementations.
    It handles arrays with different shapes by resampling them to match the
    coordinates.

    Args:
        results: Dictionary of validation results
        coordinates: Cell coordinates (n_cells, 2)
        figsize: Figure size
        resample_method: Method for resampling velocity arrays ('nearest', 'linear', 'cubic')

    Returns:
        Matplotlib figure
    """
    # Import resampling function
    from pyrovelocity.validation.comparison import resample_array

    # Get implementations
    implementations = list(results.keys())

    # Create figure
    fig, axes = plt.subplots(1, len(implementations), figsize=figsize)

    # Adjust axes for single implementation
    if len(implementations) == 1:
        axes = [axes]

    # Plot velocity vector field
    for i, impl in enumerate(implementations):
        # Get axis
        ax = axes[i]

        # Check that velocity is available
        if "velocity" not in results[impl]:
            ax.set_title(f"{impl} (No velocity)")
            continue

        # Get velocity
        velocity = results[impl]["velocity"]

        # Convert to numpy array
        if isinstance(velocity, np.ndarray):
            pass
        elif isinstance(velocity, list):
            velocity = np.array(velocity)
        elif hasattr(velocity, "detach") and hasattr(velocity, "cpu") and hasattr(velocity, "numpy"):
            # PyTorch tensor
            velocity = velocity.detach().cpu().numpy()
        else:
            # Try to convert to numpy array
            velocity = np.array(velocity)

        # Check that velocity and coordinates have the same number of cells
        if velocity.shape[0] != coordinates.shape[0]:
            # Try to resample velocity to match coordinates
            try:
                print(f"Resampling velocity for {impl} from shape {velocity.shape} to match coordinates shape {coordinates.shape[0]}")

                # Check if velocity has at least 2 dimensions
                if velocity.ndim < 2:
                    # Reshape to 2D if needed
                    velocity = velocity.reshape(-1, 1)

                # Resample velocity to match coordinates
                target_shape = (coordinates.shape[0], velocity.shape[1])
                velocity = resample_array(velocity, target_shape, method=resample_method)

                print(f"Resampled velocity shape: {velocity.shape}")
            except Exception as e:
                # If resampling fails, show error message
                ax.set_title(f"{impl} (Shape mismatch: {e})")
                continue

        # Check if velocity has at least 2 columns for vector field
        if velocity.shape[1] < 2:
            ax.set_title(f"{impl} (Not enough dimensions for vector field)")
            continue

        # Plot vector field
        ax.quiver(
            coordinates[:, 0],
            coordinates[:, 1],
            velocity[:, 0],
            velocity[:, 1],
            scale=20,
            width=0.002,
            color="black",
            alpha=0.5,
        )

        # Plot cell coordinates
        ax.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            s=5,
            c="gray",
            alpha=0.5,
        )

        # Set title
        ax.set_title(impl)

        # Remove axis ticks
        ax.set_xticks([])
        ax.set_yticks([])

    # Adjust layout
    plt.tight_layout()

    return fig


@beartype
def plot_uncertainty_heatmap(
    results: Dict[str, Dict[str, Any]],
    figsize: Tuple[int, int] = (15, 5),
    resample_method: str = "nearest",
) -> plt.Figure:
    """
    Plot uncertainty heatmap for different implementations.

    This function plots uncertainty heatmaps for different implementations.
    It handles arrays with different shapes by reshaping them for visualization.

    Args:
        results: Dictionary of validation results
        figsize: Figure size
        resample_method: Method for resampling uncertainty arrays ('nearest', 'linear', 'cubic')

    Returns:
        Matplotlib figure
    """
    # Import resampling function
    from pyrovelocity.validation.comparison import resample_array

    # Get implementations
    implementations = list(results.keys())

    # Create figure
    fig, axes = plt.subplots(1, len(implementations), figsize=figsize)

    # Adjust axes for single implementation
    if len(implementations) == 1:
        axes = [axes]

    # Plot uncertainty heatmap
    for i, impl in enumerate(implementations):
        # Get axis
        ax = axes[i]

        # Check that uncertainty is available
        if "uncertainty" not in results[impl]:
            ax.set_title(f"{impl} (No uncertainty)")
            continue

        # Get uncertainty
        uncertainty = results[impl]["uncertainty"]

        # Convert to numpy array
        if isinstance(uncertainty, np.ndarray):
            pass
        elif isinstance(uncertainty, list):
            uncertainty = np.array(uncertainty)
        elif hasattr(uncertainty, "detach") and hasattr(uncertainty, "cpu") and hasattr(uncertainty, "numpy"):
            # PyTorch tensor
            uncertainty = uncertainty.detach().cpu().numpy()
        else:
            # Try to convert to numpy array
            uncertainty = np.array(uncertainty)

        # Handle different shapes for visualization
        try:
            # If uncertainty is 1D, reshape it to 2D for visualization
            if uncertainty.ndim == 1:
                # Reshape to a square-ish 2D array
                size = int(np.sqrt(uncertainty.size))
                uncertainty_2d = uncertainty[:size*size].reshape(size, size)
                print(f"Reshaped 1D uncertainty of length {uncertainty.size} to 2D array of shape {uncertainty_2d.shape}")
                uncertainty = uncertainty_2d

            # If uncertainty has more than 2 dimensions, flatten to 2D
            elif uncertainty.ndim > 2:
                # Flatten all but the first dimension
                new_shape = (uncertainty.shape[0], -1)
                uncertainty = uncertainty.reshape(new_shape)
                print(f"Reshaped uncertainty from {uncertainty.shape} to 2D array")

            # Plot heatmap
            im = ax.imshow(uncertainty, cmap="viridis", aspect="auto")

            # Add colorbar
            plt.colorbar(im, ax=ax)

            # Set title
            ax.set_title(f"{impl} ({uncertainty.shape})")

            # Set labels
            if uncertainty.ndim == 2:
                ax.set_xlabel(f"Dimension 2 (size {uncertainty.shape[1]})")
                ax.set_ylabel(f"Dimension 1 (size {uncertainty.shape[0]})")

        except Exception as e:
            # If reshaping fails, show error message
            ax.set_title(f"{impl} (Error: {e})")
            ax.text(0.5, 0.5, f"Shape: {uncertainty.shape}\nError: {e}",
                   ha="center", va="center", transform=ax.transAxes)

    # Adjust layout
    plt.tight_layout()

    return fig


@beartype
def plot_performance_radar(
    results: Dict[str, Dict[str, Any]],
    figsize: Tuple[int, int] = (8, 8),
) -> plt.Figure:
    """
    Plot performance radar chart for different implementations.

    Args:
        results: Dictionary of validation results
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    # Get implementations
    implementations = list(results.keys())

    # Get performance metrics
    metrics = set()
    for impl in implementations:
        if "performance" in results[impl]:
            metrics.update(results[impl]["performance"].keys())
    metrics = list(metrics)

    # Create figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, polar=True)

    # Set number of angles
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()

    # Close the polygon
    angles += angles[:1]

    # Set labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)

    # Plot performance radar
    for impl in implementations:
        # Check that performance is available
        if "performance" not in results[impl]:
            continue

        # Get performance
        performance = results[impl]["performance"]

        # Get values for each metric
        values = []
        for metric in metrics:
            if metric in performance:
                values.append(performance[metric])
            else:
                values.append(0.0)

        # Close the polygon
        values += values[:1]

        # Plot radar
        ax.plot(angles, values, label=impl)
        ax.fill(angles, values, alpha=0.1)

    # Add legend
    ax.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))

    return fig
