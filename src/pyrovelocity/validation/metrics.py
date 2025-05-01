"""
PyroVelocity validation metrics.

This module provides metrics for measuring the similarity between different
implementations of PyroVelocity (legacy, modular, and JAX).

The metrics include:
- Parameter comparison metrics (MSE, correlation, KL divergence, Wasserstein distance)
- Velocity comparison metrics (MSE, correlation, cosine similarity, magnitude similarity)
- Uncertainty comparison metrics (MSE, correlation, distribution similarity)
- Performance comparison metrics (training time ratio, inference time ratio, memory usage ratio)
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
import torch
from beartype import beartype
from scipy import stats

# Basic metrics

@beartype
def mean_squared_error(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate the mean squared error between two arrays.

    Args:
        x: First array
        y: Second array

    Returns:
        Mean squared error or NaN if calculation fails
    """
    try:
        # Check for NaN or Inf values
        if np.isnan(x).any() or np.isnan(y).any() or np.isinf(x).any() or np.isinf(y).any():
            # Replace NaN and Inf with zeros
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
            print("Warning: Arrays contain NaN or Inf values. Replacing with zeros.")

        # Calculate MSE
        return float(np.mean((x - y) ** 2))
    except Exception as e:
        print(f"Error calculating mean squared error: {e}")
        return float('nan')


@beartype
def correlation(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate the correlation between two arrays.

    Args:
        x: First array
        y: Second array

    Returns:
        Correlation coefficient or NaN if calculation fails
    """
    try:
        # Check for NaN or Inf values
        if np.isnan(x).any() or np.isnan(y).any() or np.isinf(x).any() or np.isinf(y).any():
            # Replace NaN and Inf with zeros
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
            print("Warning: Arrays contain NaN or Inf values. Replacing with zeros.")

        # Flatten arrays
        x_flat = x.flatten()
        y_flat = y.flatten()

        # Check if arrays have constant values
        if np.all(x_flat == x_flat[0]) or np.all(y_flat == y_flat[0]):
            print("Warning: One or both arrays have constant values. Correlation is undefined.")
            return 0.0

        # Calculate correlation
        corr_matrix = np.corrcoef(x_flat, y_flat)
        if corr_matrix.shape == (2, 2):
            return float(corr_matrix[0, 1])
        else:
            print(f"Warning: Correlation matrix has unexpected shape: {corr_matrix.shape}")
            return 0.0
    except Exception as e:
        print(f"Error calculating correlation: {e}")
        return float('nan')


@beartype
def cosine_similarity(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate the cosine similarity between two arrays.

    Args:
        x: First array
        y: Second array

    Returns:
        Cosine similarity or NaN if calculation fails
    """
    try:
        # Check for NaN or Inf values
        if np.isnan(x).any() or np.isnan(y).any() or np.isinf(x).any() or np.isinf(y).any():
            # Replace NaN and Inf with zeros
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
            print("Warning: Arrays contain NaN or Inf values. Replacing with zeros.")

        # Flatten arrays
        x_flat = x.flatten()
        y_flat = y.flatten()

        # Check for zero norm
        x_norm = np.linalg.norm(x_flat)
        y_norm = np.linalg.norm(y_flat)

        if x_norm == 0 or y_norm == 0:
            print("Warning: One or both arrays have zero norm. Cosine similarity is undefined.")
            return 0.0

        # Calculate cosine similarity
        return float(np.dot(x_flat, y_flat) / (x_norm * y_norm))
    except Exception as e:
        print(f"Error calculating cosine similarity: {e}")
        return float('nan')


@beartype
def kl_divergence(x: np.ndarray, y: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    Calculate the KL divergence between two arrays.

    Args:
        x: First array (probability distribution)
        y: Second array (probability distribution)
        epsilon: Small constant to avoid division by zero

    Returns:
        KL divergence or NaN if calculation fails
    """
    try:
        # Check for NaN or Inf values
        if np.isnan(x).any() or np.isnan(y).any() or np.isinf(x).any() or np.isinf(y).any():
            # Replace NaN and Inf with zeros
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
            print("Warning: Arrays contain NaN or Inf values. Replacing with zeros.")

        # Check for negative values
        if np.any(x < 0) or np.any(y < 0):
            print("Warning: Arrays contain negative values. Taking absolute values.")
            x = np.abs(x)
            y = np.abs(y)

        # Check for zero sums
        x_sum = np.sum(x)
        y_sum = np.sum(y)

        if x_sum == 0 or y_sum == 0:
            print("Warning: One or both arrays sum to zero. KL divergence is undefined.")
            return 0.0

        # Ensure arrays are probability distributions
        x = x / x_sum
        y = y / y_sum

        # Add small constant to avoid division by zero
        x = x + epsilon
        y = y + epsilon

        # Renormalize
        x = x / np.sum(x)
        y = y / np.sum(y)

        # Calculate KL divergence
        kl_div = float(np.sum(x * np.log(x / y)))

        # Check for NaN or Inf in result
        if np.isnan(kl_div) or np.isinf(kl_div):
            print("Warning: KL divergence calculation resulted in NaN or Inf. Returning 0.")
            return 0.0

        return kl_div
    except Exception as e:
        print(f"Error calculating KL divergence: {e}")
        return float('nan')


@beartype
def wasserstein_distance(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate the Wasserstein distance between two arrays.

    Args:
        x: First array
        y: Second array

    Returns:
        Wasserstein distance or NaN if calculation fails
    """
    try:
        # Check for NaN or Inf values
        if np.isnan(x).any() or np.isnan(y).any() or np.isinf(x).any() or np.isinf(y).any():
            # Replace NaN and Inf with zeros
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
            print("Warning: Arrays contain NaN or Inf values. Replacing with zeros.")

        # Flatten arrays
        x_flat = x.flatten()
        y_flat = y.flatten()

        # Check if arrays are empty
        if x_flat.size == 0 or y_flat.size == 0:
            print("Warning: One or both arrays are empty. Wasserstein distance is undefined.")
            return 0.0

        # Calculate Wasserstein distance
        distance = float(stats.wasserstein_distance(x_flat, y_flat))

        # Check for NaN or Inf in result
        if np.isnan(distance) or np.isinf(distance):
            print("Warning: Wasserstein distance calculation resulted in NaN or Inf. Returning 0.")
            return 0.0

        return distance
    except Exception as e:
        print(f"Error calculating Wasserstein distance: {e}")
        return float('nan')


# Parameter comparison metrics

@beartype
def compute_parameter_metrics(
    params1: Dict[str, np.ndarray],
    params2: Dict[str, np.ndarray]
) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics for comparing model parameters.

    Args:
        params1: First set of parameters
        params2: Second set of parameters

    Returns:
        Dictionary of parameter metrics
    """
    # Initialize metrics dictionary
    metrics = {}

    # Compute metrics for each parameter
    for param in params1.keys():
        if param in params2:
            # Convert parameters to numpy arrays
            p1 = np.array(params1[param])
            p2 = np.array(params2[param])

            # Compute metrics
            metrics[param] = {
                "mse": mean_squared_error(p1, p2),
                "correlation": correlation(p1, p2),
                "kl_divergence": kl_divergence(np.abs(p1), np.abs(p2)),
                "wasserstein_distance": wasserstein_distance(p1, p2),
            }

    return metrics


# Velocity comparison metrics

@beartype
def compute_velocity_metrics(
    velocity1: Union[np.ndarray, Dict[str, Union[np.ndarray, torch.Tensor, jnp.ndarray]]],
    velocity2: Union[np.ndarray, Dict[str, Union[np.ndarray, torch.Tensor, jnp.ndarray]]]
) -> Dict[str, float]:
    """
    Compute metrics for comparing velocity estimates.

    This function computes metrics for comparing velocity estimates between
    different implementations. It handles arrays with different shapes by
    ensuring they are compatible before computing metrics.

    Args:
        velocity1: First velocity estimate, either a numpy array or a dictionary with a 'velocity' key
        velocity2: Second velocity estimate, either a numpy array or a dictionary with a 'velocity' key

    Returns:
        Dictionary of velocity metrics
    """
    # Handle dictionary input
    if isinstance(velocity1, dict) and 'velocity' in velocity1:
        velocity1 = velocity1['velocity']
    if isinstance(velocity2, dict) and 'velocity' in velocity2:
        velocity2 = velocity2['velocity']

    # Convert velocities to numpy arrays
    if isinstance(velocity1, torch.Tensor):
        v1 = velocity1.detach().cpu().numpy()
    elif isinstance(velocity1, jnp.ndarray):
        v1 = np.array(velocity1)
    else:
        v1 = np.array(velocity1)

    if isinstance(velocity2, torch.Tensor):
        v2 = velocity2.detach().cpu().numpy()
    elif isinstance(velocity2, jnp.ndarray):
        v2 = np.array(velocity2)
    else:
        v2 = np.array(velocity2)

    # Check if shapes match
    if v1.shape != v2.shape:
        # Shapes don't match, but we can still compute some metrics
        # For metrics that require the same shape, we'll flatten the arrays
        v1_flat = v1.flatten()
        v2_flat = v2.flatten()

        # If arrays have different lengths after flattening, truncate to the shorter length
        min_length = min(len(v1_flat), len(v2_flat))
        v1_flat = v1_flat[:min_length]
        v2_flat = v2_flat[:min_length]

        # Compute metrics that can handle different shapes
        metrics = {
            "mse": mean_squared_error(v1_flat, v2_flat),
            "correlation": correlation(v1_flat, v2_flat),
            "cosine_similarity": cosine_similarity(v1_flat, v2_flat),
            "magnitude_similarity": 1.0 - np.abs(np.linalg.norm(v1_flat) - np.linalg.norm(v2_flat)) / (np.linalg.norm(v1_flat) + np.linalg.norm(v2_flat)),
            "shape_mismatch": True,
            "shape1": str(v1.shape),
            "shape2": str(v2.shape)
        }
    else:
        # Shapes match, compute metrics normally
        metrics = {
            "mse": mean_squared_error(v1, v2),
            "correlation": correlation(v1, v2),
            "cosine_similarity": cosine_similarity(v1, v2),
            "magnitude_similarity": 1.0 - np.abs(np.linalg.norm(v1) - np.linalg.norm(v2)) / (np.linalg.norm(v1) + np.linalg.norm(v2)),
            "shape_mismatch": False
        }

    return metrics


# Uncertainty comparison metrics

@beartype
def compute_uncertainty_metrics(
    uncertainty1: np.ndarray,
    uncertainty2: np.ndarray
) -> Dict[str, float]:
    """
    Compute metrics for comparing uncertainty estimates.

    This function computes metrics for comparing uncertainty estimates between
    different implementations. It handles arrays with different shapes by
    ensuring they are compatible before computing metrics.

    Args:
        uncertainty1: First uncertainty estimate
        uncertainty2: Second uncertainty estimate

    Returns:
        Dictionary of uncertainty metrics
    """
    # Convert uncertainties to numpy arrays
    u1 = np.array(uncertainty1)
    u2 = np.array(uncertainty2)

    # Check if shapes match
    if u1.shape != u2.shape:
        # Shapes don't match, but we can still compute some metrics
        # For metrics that require the same shape, we'll flatten the arrays
        u1_flat = u1.flatten()
        u2_flat = u2.flatten()

        # If arrays have different lengths after flattening, truncate to the shorter length
        min_length = min(len(u1_flat), len(u2_flat))
        u1_flat = u1_flat[:min_length]
        u2_flat = u2_flat[:min_length]

        # Compute metrics that can handle different shapes
        metrics = {
            "mse": mean_squared_error(u1_flat, u2_flat),
            "correlation": correlation(u1_flat, u2_flat),
            "distribution_similarity": 1.0 - wasserstein_distance(u1_flat, u2_flat) / (np.mean(u1_flat) + np.mean(u2_flat)),
            "shape_mismatch": True,
            "shape1": str(u1.shape),
            "shape2": str(u2.shape)
        }
    else:
        # Shapes match, compute metrics normally
        metrics = {
            "mse": mean_squared_error(u1, u2),
            "correlation": correlation(u1, u2),
            "distribution_similarity": 1.0 - wasserstein_distance(u1, u2) / (np.mean(u1) + np.mean(u2)),
            "shape_mismatch": False
        }

    return metrics


# Performance comparison metrics

@beartype
def compute_performance_metrics(
    performance1: Dict[str, float],
    performance2: Dict[str, float]
) -> Dict[str, float]:
    """
    Compute metrics for comparing performance.

    Args:
        performance1: First performance metrics
        performance2: Second performance metrics

    Returns:
        Dictionary of performance comparison metrics
    """
    # Compute ratios
    metrics = {}

    # Training time ratio
    if "training_time" in performance1 and "training_time" in performance2:
        metrics["training_time_ratio"] = performance2["training_time"] / performance1["training_time"]

    # Inference time ratio
    if "inference_time" in performance1 and "inference_time" in performance2:
        metrics["inference_time_ratio"] = performance2["inference_time"] / performance1["inference_time"]

    # Memory usage ratio
    if "memory_usage" in performance1 and "memory_usage" in performance2:
        metrics["memory_usage_ratio"] = performance2["memory_usage"] / performance1["memory_usage"]

    return metrics
