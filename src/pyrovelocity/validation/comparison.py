"""
PyroVelocity comparison utilities.

This module provides utilities for comparing different implementations of
PyroVelocity (legacy, modular, and JAX).

The utilities include:
- compare_parameters: Compare model parameters between implementations
- compare_velocities: Compare velocity estimates between implementations
- compare_uncertainties: Compare uncertainty estimates between implementations
- compare_performance: Compare performance metrics between implementations
- statistical_comparison: Perform statistical comparison between arrays
- detect_outliers: Detect outliers in comparison results
- detect_systematic_bias: Detect systematic bias in comparison results
- identify_edge_cases: Identify edge cases in comparison results
- normalize_shapes: Normalize shapes of two arrays for comparison
- resample_array: Resample an array to match a target shape
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
import torch
from beartype import beartype
from scipy import stats

from pyrovelocity.validation.metrics import (
    compute_parameter_metrics,
    compute_performance_metrics,
    compute_uncertainty_metrics,
    compute_velocity_metrics,
)


@beartype
def compare_parameters(
    results: Dict[str, Dict[str, Any]]
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Compare model parameters between implementations.

    Args:
        results: Dictionary of validation results

    Returns:
        Dictionary of parameter comparison results
    """
    # Initialize comparison dictionary
    comparison = {}

    # Get implementations
    implementations = list(results.keys())

    # Check that posterior samples are available
    for impl in implementations:
        if "posterior_samples" not in results[impl]:
            raise ValueError(f"Posterior samples not available for {impl} implementation")

    # Get parameters
    parameters = set()
    for impl in implementations:
        parameters.update(results[impl]["posterior_samples"].keys())

    # Compare parameters
    for param in parameters:
        # Initialize parameter comparison dictionary
        comparison[param] = {}

        # Compare each pair of implementations
        for i, impl1 in enumerate(implementations):
            for impl2 in implementations[i+1:]:
                # Check that parameter is available for both implementations
                if param in results[impl1]["posterior_samples"] and param in results[impl2]["posterior_samples"]:
                    # Get parameters
                    params1 = results[impl1]["posterior_samples"][param]
                    params2 = results[impl2]["posterior_samples"][param]

                    # Convert to numpy arrays
                    if isinstance(params1, torch.Tensor):
                        params1 = params1.detach().cpu().numpy()
                    elif isinstance(params1, jnp.ndarray):
                        params1 = np.array(params1)

                    if isinstance(params2, torch.Tensor):
                        params2 = params2.detach().cpu().numpy()
                    elif isinstance(params2, jnp.ndarray):
                        params2 = np.array(params2)

                    # Compute metrics
                    metrics = compute_parameter_metrics(
                        {param: params1},
                        {param: params2}
                    )

                    # Store metrics
                    comparison[param][f"{impl1}_vs_{impl2}"] = metrics[param]

    return comparison


@beartype
def compare_velocities(
    results: Dict[str, Dict[str, Any]],
    normalize_method: str = "nearest",
    target_strategy: str = "max"
) -> Dict[str, Dict[str, Any]]:
    """
    Compare velocity estimates between implementations.

    This function compares velocity estimates between different implementations.
    It handles arrays with different shapes by normalizing them before comparison.

    Args:
        results: Dictionary of validation results
        normalize_method: Method for normalizing shapes ('nearest', 'linear', 'cubic')
        target_strategy: Strategy for determining target shape ('max', 'min', 'first', 'second')

    Returns:
        Dictionary of velocity comparison results, which may include error messages
    """
    # Initialize comparison dictionary
    comparison = {}

    # Get implementations
    implementations = list(results.keys())

    # Check that velocity estimates are available
    for impl in implementations:
        if "velocity" not in results[impl]:
            raise ValueError(f"Velocity estimates not available for {impl} implementation")

    # Compare velocities
    for i, impl1 in enumerate(implementations):
        for impl2 in implementations[i+1:]:
            # Get velocity estimates
            velocity1 = results[impl1]["velocity"]
            velocity2 = results[impl2]["velocity"]

            # Convert to numpy arrays
            if isinstance(velocity1, torch.Tensor):
                velocity1 = velocity1.detach().cpu().numpy()
            elif isinstance(velocity1, jnp.ndarray):
                velocity1 = np.array(velocity1)
            else:
                velocity1 = np.array(velocity1)

            if isinstance(velocity2, torch.Tensor):
                velocity2 = velocity2.detach().cpu().numpy()
            elif isinstance(velocity2, jnp.ndarray):
                velocity2 = np.array(velocity2)
            else:
                velocity2 = np.array(velocity2)

            # Check if shapes match
            if velocity1.shape != velocity2.shape:
                # Log shape mismatch
                print(f"Shape mismatch between {impl1} ({velocity1.shape}) and {impl2} ({velocity2.shape})")
                print(f"Normalizing shapes using method='{normalize_method}', target='{target_strategy}'")

                try:
                    # Normalize shapes
                    velocity1, velocity2 = normalize_shapes(
                        velocity1,
                        velocity2,
                        method=normalize_method,
                        target=target_strategy
                    )
                    print(f"Normalized shapes: {velocity1.shape}")
                except Exception as e:
                    # Handle normalization errors
                    print(f"Error normalizing shapes: {e}")
                    print("Using flattened arrays for comparison")
                    velocity1 = velocity1.flatten()
                    velocity2 = velocity2.flatten()

                    # If arrays have different lengths after flattening, truncate to the shorter length
                    min_length = min(len(velocity1), len(velocity2))
                    velocity1 = velocity1[:min_length]
                    velocity2 = velocity2[:min_length]

            # Compute metrics
            try:
                metrics = compute_velocity_metrics(velocity1, velocity2)

                # Store metrics
                comparison[f"{impl1}_vs_{impl2}"] = metrics
            except Exception as e:
                # Handle metric computation errors
                print(f"Error computing metrics: {e}")
                comparison[f"{impl1}_vs_{impl2}"] = {
                    "error": str(e),
                    "velocity1_shape": velocity1.shape,
                    "velocity2_shape": velocity2.shape
                }

    return comparison


@beartype
def compare_uncertainties(
    results: Dict[str, Dict[str, Any]],
    normalize_method: str = "nearest",
    target_strategy: str = "max"
) -> Dict[str, Dict[str, Any]]:
    """
    Compare uncertainty estimates between implementations.

    This function compares uncertainty estimates between different implementations.
    It handles arrays with different shapes by normalizing them before comparison.

    Args:
        results: Dictionary of validation results
        normalize_method: Method for normalizing shapes ('nearest', 'linear', 'cubic')
        target_strategy: Strategy for determining target shape ('max', 'min', 'first', 'second')

    Returns:
        Dictionary of uncertainty comparison results, which may include error messages
    """
    # Initialize comparison dictionary
    comparison = {}

    # Get implementations
    implementations = list(results.keys())

    # Check that uncertainty estimates are available
    for impl in implementations:
        if "uncertainty" not in results[impl]:
            raise ValueError(f"Uncertainty estimates not available for {impl} implementation")

    # Compare uncertainties
    for i, impl1 in enumerate(implementations):
        for impl2 in implementations[i+1:]:
            # Get uncertainty estimates
            uncertainty1 = results[impl1]["uncertainty"]
            uncertainty2 = results[impl2]["uncertainty"]

            # Convert to numpy arrays
            if isinstance(uncertainty1, torch.Tensor):
                uncertainty1 = uncertainty1.detach().cpu().numpy()
            elif isinstance(uncertainty1, jnp.ndarray):
                uncertainty1 = np.array(uncertainty1)
            else:
                uncertainty1 = np.array(uncertainty1)

            if isinstance(uncertainty2, torch.Tensor):
                uncertainty2 = uncertainty2.detach().cpu().numpy()
            elif isinstance(uncertainty2, jnp.ndarray):
                uncertainty2 = np.array(uncertainty2)
            else:
                uncertainty2 = np.array(uncertainty2)

            # Check if shapes match
            if uncertainty1.shape != uncertainty2.shape:
                # Log shape mismatch
                print(f"Shape mismatch between {impl1} ({uncertainty1.shape}) and {impl2} ({uncertainty2.shape})")
                print(f"Normalizing shapes using method='{normalize_method}', target='{target_strategy}'")

                try:
                    # Normalize shapes
                    uncertainty1, uncertainty2 = normalize_shapes(
                        uncertainty1,
                        uncertainty2,
                        method=normalize_method,
                        target=target_strategy
                    )
                    print(f"Normalized shapes: {uncertainty1.shape}")
                except Exception as e:
                    # Handle normalization errors
                    print(f"Error normalizing shapes: {e}")
                    print("Using flattened arrays for comparison")
                    uncertainty1 = uncertainty1.flatten()
                    uncertainty2 = uncertainty2.flatten()

                    # If arrays have different lengths after flattening, truncate to the shorter length
                    min_length = min(len(uncertainty1), len(uncertainty2))
                    uncertainty1 = uncertainty1[:min_length]
                    uncertainty2 = uncertainty2[:min_length]

            # Compute metrics
            try:
                metrics = compute_uncertainty_metrics(uncertainty1, uncertainty2)

                # Store metrics
                comparison[f"{impl1}_vs_{impl2}"] = metrics
            except Exception as e:
                # Handle metric computation errors
                print(f"Error computing metrics: {e}")
                comparison[f"{impl1}_vs_{impl2}"] = {
                    "error": str(e),
                    "uncertainty1_shape": uncertainty1.shape,
                    "uncertainty2_shape": uncertainty2.shape
                }

    return comparison


@beartype
def compare_performance(
    results: Dict[str, Dict[str, Any]]
) -> Dict[str, Dict[str, float]]:
    """
    Compare performance metrics between implementations.

    Args:
        results: Dictionary of validation results

    Returns:
        Dictionary of performance comparison results
    """
    # Initialize comparison dictionary
    comparison = {}

    # Get implementations
    implementations = list(results.keys())

    # Check that performance metrics are available
    for impl in implementations:
        if "performance" not in results[impl]:
            raise ValueError(f"Performance metrics not available for {impl} implementation")

    # Compare performance
    for i, impl1 in enumerate(implementations):
        for impl2 in implementations[i+1:]:
            # Get performance metrics
            performance1 = results[impl1]["performance"]
            performance2 = results[impl2]["performance"]

            # Compute metrics
            metrics = compute_performance_metrics(performance1, performance2)

            # Store metrics
            comparison[f"{impl1}_vs_{impl2}"] = metrics

    return comparison


@beartype
def statistical_comparison(
    x: np.ndarray,
    y: np.ndarray
) -> Dict[str, Dict[str, float]]:
    """
    Perform statistical comparison between two arrays.

    Args:
        x: First array
        y: Second array

    Returns:
        Dictionary of statistical comparison results
    """
    # Flatten arrays
    x_flat = x.flatten()
    y_flat = y.flatten()

    # Perform t-test
    t_stat, t_pval = stats.ttest_ind(x_flat, y_flat)

    # Perform Wilcoxon test
    try:
        w_stat, w_pval = stats.wilcoxon(x_flat, y_flat)
    except ValueError:
        # Wilcoxon test requires at least one non-zero difference
        w_stat, w_pval = np.nan, np.nan

    # Perform Kolmogorov-Smirnov test
    ks_stat, ks_pval = stats.ks_2samp(x_flat, y_flat)

    # Return results
    return {
        "t_test": {
            "statistic": float(t_stat),
            "p_value": float(t_pval),
        },
        "wilcoxon_test": {
            "statistic": float(w_stat) if not np.isnan(w_stat) else None,
            "p_value": float(w_pval) if not np.isnan(w_pval) else None,
        },
        "ks_test": {
            "statistic": float(ks_stat),
            "p_value": float(ks_pval),
        },
    }


@beartype
def detect_outliers(
    data: np.ndarray,
    threshold: float = 3.0
) -> List[int]:
    """
    Detect outliers in an array using Z-score.

    Args:
        data: Input array
        threshold: Z-score threshold for outlier detection

    Returns:
        List of outlier indices
    """
    # Flatten array
    data_flat = data.flatten()

    # Compute Z-scores
    z_scores = np.abs((data_flat - np.mean(data_flat)) / np.std(data_flat))

    # Detect outliers
    outliers = np.where(z_scores > threshold)[0]

    return outliers.tolist()


@beartype
def detect_systematic_bias(
    x: np.ndarray,
    y: np.ndarray,
    threshold: float = 0.05
) -> Dict[str, Any]:
    """
    Detect systematic bias between two arrays.

    Args:
        x: First array
        y: Second array
        threshold: P-value threshold for bias detection

    Returns:
        Dictionary of bias detection results
    """
    # Flatten arrays
    x_flat = x.flatten()
    y_flat = y.flatten()

    # Compute difference
    diff = y_flat - x_flat

    # Compute mean difference
    mean_diff = np.mean(diff)

    # Perform one-sample t-test
    t_stat, p_value = stats.ttest_1samp(diff, 0.0)

    # Detect bias
    bias_detected = p_value < threshold

    # Return results
    return {
        "detected": bool(bias_detected),
        "mean_difference": float(mean_diff),
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
    }


@beartype
def identify_edge_cases(
    data: np.ndarray,
    percentile: float = 1.0
) -> Dict[str, Any]:
    """
    Identify edge cases in an array.

    Args:
        data: Input array
        percentile: Percentile threshold for edge case detection

    Returns:
        Dictionary of edge case detection results
    """
    # Flatten array
    data_flat = data.flatten()

    # Compute min and max values
    min_val = np.min(data_flat)
    max_val = np.max(data_flat)

    # Compute percentiles
    lower_percentile = np.percentile(data_flat, percentile)
    upper_percentile = np.percentile(data_flat, 100 - percentile)

    # Identify extreme values
    extreme_values = np.where((data_flat <= lower_percentile) | (data_flat >= upper_percentile))[0]

    # Return results
    return {
        "min_values": float(min_val),
        "max_values": float(max_val),
        "lower_percentile": float(lower_percentile),
        "upper_percentile": float(upper_percentile),
        "extreme_values": extreme_values.tolist(),
    }


@beartype
def resample_array(
    array: np.ndarray,
    target_shape: Tuple[int, ...],
    method: str = "nearest"
) -> np.ndarray:
    """
    Resample an array to match a target shape.

    This function resamples an array to match a target shape, which is useful
    for comparing arrays with different shapes. It uses scipy's zoom function
    to perform the resampling and handles arrays with different dimensions.

    Args:
        array: Input array to resample
        target_shape: Target shape for the resampled array
        method: Resampling method ('nearest', 'linear', 'cubic')

    Returns:
        Resampled array with the target shape
    """
    from scipy.ndimage import zoom

    # Ensure array is a numpy array
    array = np.array(array)

    # Handle dimension mismatch
    if array.ndim != len(target_shape):
        # If array has fewer dimensions than target, add dimensions
        if array.ndim < len(target_shape):
            # Reshape array to have the same number of dimensions as target
            new_shape = list(array.shape)
            while len(new_shape) < len(target_shape):
                new_shape.append(1)
            array = array.reshape(new_shape)
        # If array has more dimensions than target, flatten extra dimensions
        else:
            # Keep the first len(target_shape)-1 dimensions as is
            # and flatten the rest into the last dimension
            if len(target_shape) > 1:
                new_shape = list(array.shape[:len(target_shape)-1])
                new_shape.append(-1)  # Flatten remaining dimensions
                array = array.reshape(new_shape)
            else:
                # If target is 1D, flatten the entire array
                array = array.flatten()

    # Calculate zoom factors
    zoom_factors = []
    for i in range(min(len(target_shape), array.ndim)):
        # Handle zero dimensions
        if array.shape[i] == 0 or target_shape[i] == 0:
            zoom_factors.append(1.0)
        else:
            zoom_factors.append(target_shape[i] / array.shape[i])

    # Add zoom factors for additional dimensions if needed
    if array.ndim < len(target_shape):
        zoom_factors.extend([1.0] * (len(target_shape) - array.ndim))

    # Handle special case: if any zoom factor is 0, use resize instead of zoom
    if any(factor == 0 for factor in zoom_factors):
        return np.resize(array, target_shape)

    try:
        # Resample array
        resampled = zoom(array, zoom_factors, order=0 if method == "nearest" else 1, mode="nearest")

        # Reshape if needed
        if resampled.shape != target_shape:
            # This can happen if the dimensions don't match exactly
            # We'll reshape to match the target shape
            resampled = np.resize(resampled, target_shape)

        return resampled
    except Exception as e:
        # If zoom fails, fall back to resize
        print(f"Warning: zoom failed with error: {e}. Falling back to resize.")
        return np.resize(array, target_shape)


@beartype
def normalize_shapes(
    array1: np.ndarray,
    array2: np.ndarray,
    method: str = "nearest",
    target: str = "max"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalize shapes of two arrays for comparison.

    This function normalizes the shapes of two arrays to make them comparable.
    It can either resample both arrays to the larger shape, the smaller shape,
    or a specified target shape. It handles arrays with different dimensions
    by padding or truncating dimensions as needed.

    Args:
        array1: First array
        array2: Second array
        method: Resampling method ('nearest', 'linear', 'cubic')
        target: Target shape strategy ('max', 'min', 'first', or 'second')

    Returns:
        Tuple of normalized arrays with the same shape
    """
    # Ensure arrays are numpy arrays
    array1 = np.array(array1)
    array2 = np.array(array2)

    # Check if shapes already match
    if array1.shape == array2.shape:
        return array1, array2

    # Handle different dimensions
    if array1.ndim != array2.ndim:
        # Reshape arrays to have the same number of dimensions
        if array1.ndim < array2.ndim:
            # Reshape array1 to match array2's dimensions
            new_shape = list(array1.shape)
            while len(new_shape) < array2.ndim:
                new_shape.append(1)
            array1 = array1.reshape(new_shape)
        else:
            # Reshape array2 to match array1's dimensions
            new_shape = list(array2.shape)
            while len(new_shape) < array1.ndim:
                new_shape.append(1)
            array2 = array2.reshape(new_shape)

    # Determine target shape
    if target == "max":
        # Use the larger shape for each dimension
        target_shape = tuple(max(s1, s2) for s1, s2 in zip(array1.shape, array2.shape))
    elif target == "min":
        # Use the smaller shape for each dimension
        target_shape = tuple(min(s1, s2) for s1, s2 in zip(array1.shape, array2.shape))
    elif target == "first":
        # Use the shape of the first array
        target_shape = array1.shape
    elif target == "second":
        # Use the shape of the second array
        target_shape = array2.shape
    else:
        raise ValueError(f"Invalid target strategy: {target}")

    # Resample arrays to target shape
    normalized1 = resample_array(array1, target_shape, method=method)
    normalized2 = resample_array(array2, target_shape, method=method)

    return normalized1, normalized2
