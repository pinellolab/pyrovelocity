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
"""

from typing import Dict, Any, Optional, Tuple, List, Union, Callable

import numpy as np
import torch
import jax
import jax.numpy as jnp
from scipy import stats
from beartype import beartype

from pyrovelocity.validation.metrics import (
    compute_parameter_metrics,
    compute_velocity_metrics,
    compute_uncertainty_metrics,
    compute_performance_metrics,
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
    results: Dict[str, Dict[str, Any]]
) -> Dict[str, Dict[str, float]]:
    """
    Compare velocity estimates between implementations.
    
    Args:
        results: Dictionary of validation results
    
    Returns:
        Dictionary of velocity comparison results
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
            
            if isinstance(velocity2, torch.Tensor):
                velocity2 = velocity2.detach().cpu().numpy()
            elif isinstance(velocity2, jnp.ndarray):
                velocity2 = np.array(velocity2)
            
            # Compute metrics
            metrics = compute_velocity_metrics(velocity1, velocity2)
            
            # Store metrics
            comparison[f"{impl1}_vs_{impl2}"] = metrics
    
    return comparison


@beartype
def compare_uncertainties(
    results: Dict[str, Dict[str, Any]]
) -> Dict[str, Dict[str, float]]:
    """
    Compare uncertainty estimates between implementations.
    
    Args:
        results: Dictionary of validation results
    
    Returns:
        Dictionary of uncertainty comparison results
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
            
            if isinstance(uncertainty2, torch.Tensor):
                uncertainty2 = uncertainty2.detach().cpu().numpy()
            elif isinstance(uncertainty2, jnp.ndarray):
                uncertainty2 = np.array(uncertainty2)
            
            # Compute metrics
            metrics = compute_uncertainty_metrics(uncertainty1, uncertainty2)
            
            # Store metrics
            comparison[f"{impl1}_vs_{impl2}"] = metrics
    
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
