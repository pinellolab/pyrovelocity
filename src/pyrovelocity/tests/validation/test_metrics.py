"""
Tests for the PyroVelocity validation metrics.

This module contains tests for the validation metrics, which provide tools
for measuring the similarity between different implementations of PyroVelocity.
"""

import os
from typing import Dict, Any, Optional, Tuple, List, Union

import numpy as np
import pytest
import torch
import jax.numpy as jnp
from scipy import stats

# Import the validation metrics
from pyrovelocity.validation.metrics import (
    compute_parameter_metrics,
    compute_velocity_metrics,
    compute_uncertainty_metrics,
    compute_performance_metrics,
    mean_squared_error,
    correlation,
    cosine_similarity,
    kl_divergence,
    wasserstein_distance,
)


class TestBasicMetrics:
    """Tests for basic metrics used in validation."""
    
    def test_mean_squared_error(self):
        """Test mean squared error calculation."""
        # Create test data
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
        
        # Calculate MSE
        mse = mean_squared_error(x, y)
        
        # Check that MSE is calculated correctly
        expected_mse = np.mean((x - y) ** 2)
        assert np.isclose(mse, expected_mse)
    
    def test_correlation(self):
        """Test correlation calculation."""
        # Create test data
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
        
        # Calculate correlation
        corr = correlation(x, y)
        
        # Check that correlation is calculated correctly
        expected_corr = np.corrcoef(x, y)[0, 1]
        assert np.isclose(corr, expected_corr)
    
    def test_cosine_similarity(self):
        """Test cosine similarity calculation."""
        # Create test data
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
        
        # Calculate cosine similarity
        cos_sim = cosine_similarity(x, y)
        
        # Check that cosine similarity is calculated correctly
        norm_x = np.linalg.norm(x)
        norm_y = np.linalg.norm(y)
        expected_cos_sim = np.dot(x, y) / (norm_x * norm_y)
        assert np.isclose(cos_sim, expected_cos_sim)
    
    def test_kl_divergence(self):
        """Test KL divergence calculation."""
        # Create test data
        x = np.array([0.1, 0.2, 0.3, 0.4])
        y = np.array([0.15, 0.25, 0.35, 0.25])
        
        # Calculate KL divergence
        kl = kl_divergence(x, y)
        
        # Check that KL divergence is calculated correctly
        expected_kl = np.sum(x * np.log(x / y))
        assert np.isclose(kl, expected_kl)
    
    def test_wasserstein_distance(self):
        """Test Wasserstein distance calculation."""
        # Create test data
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
        
        # Calculate Wasserstein distance
        wd = wasserstein_distance(x, y)
        
        # Check that Wasserstein distance is calculated correctly
        expected_wd = stats.wasserstein_distance(x, y)
        assert np.isclose(wd, expected_wd)


class TestParameterMetrics:
    """Tests for parameter comparison metrics."""
    
    def test_compute_parameter_metrics(self):
        """Test computing parameter comparison metrics."""
        # Create test data
        params1 = {
            "alpha": np.array([1.0, 2.0, 3.0]),
            "beta": np.array([0.5, 1.0, 1.5]),
            "gamma": np.array([0.1, 0.2, 0.3]),
        }
        params2 = {
            "alpha": np.array([1.1, 2.1, 3.1]),
            "beta": np.array([0.6, 1.1, 1.6]),
            "gamma": np.array([0.15, 0.25, 0.35]),
        }
        
        # Compute parameter metrics
        metrics = compute_parameter_metrics(params1, params2)
        
        # Check that metrics were computed for all parameters
        assert "alpha" in metrics
        assert "beta" in metrics
        assert "gamma" in metrics
        
        # Check that all metrics were computed for each parameter
        for param in ["alpha", "beta", "gamma"]:
            assert "mse" in metrics[param]
            assert "correlation" in metrics[param]
            assert "kl_divergence" in metrics[param]
            assert "wasserstein_distance" in metrics[param]
        
        # Check that metrics are reasonable
        assert metrics["alpha"]["mse"] > 0
        assert 0 <= metrics["alpha"]["correlation"] <= 1
        assert metrics["alpha"]["kl_divergence"] >= 0
        assert metrics["alpha"]["wasserstein_distance"] >= 0


class TestVelocityMetrics:
    """Tests for velocity comparison metrics."""
    
    def test_compute_velocity_metrics(self):
        """Test computing velocity comparison metrics."""
        # Create test data
        velocity1 = np.random.rand(10, 5)  # 10 cells, 5 genes
        velocity2 = np.random.rand(10, 5)
        
        # Compute velocity metrics
        metrics = compute_velocity_metrics(velocity1, velocity2)
        
        # Check that all metrics were computed
        assert "mse" in metrics
        assert "correlation" in metrics
        assert "cosine_similarity" in metrics
        assert "magnitude_similarity" in metrics
        
        # Check that metrics are reasonable
        assert metrics["mse"] >= 0
        assert -1 <= metrics["correlation"] <= 1
        assert -1 <= metrics["cosine_similarity"] <= 1
        assert metrics["magnitude_similarity"] >= 0


class TestUncertaintyMetrics:
    """Tests for uncertainty comparison metrics."""
    
    def test_compute_uncertainty_metrics(self):
        """Test computing uncertainty comparison metrics."""
        # Create test data
        uncertainty1 = np.random.rand(10, 5)  # 10 cells, 5 genes
        uncertainty2 = np.random.rand(10, 5)
        
        # Compute uncertainty metrics
        metrics = compute_uncertainty_metrics(uncertainty1, uncertainty2)
        
        # Check that all metrics were computed
        assert "mse" in metrics
        assert "correlation" in metrics
        assert "distribution_similarity" in metrics
        
        # Check that metrics are reasonable
        assert metrics["mse"] >= 0
        assert -1 <= metrics["correlation"] <= 1
        assert 0 <= metrics["distribution_similarity"] <= 1


class TestPerformanceMetrics:
    """Tests for performance comparison metrics."""
    
    def test_compute_performance_metrics(self):
        """Test computing performance comparison metrics."""
        # Create test data
        performance1 = {
            "training_time": 1.0,
            "inference_time": 0.5,
            "memory_usage": 100.0,
        }
        performance2 = {
            "training_time": 1.2,
            "inference_time": 0.6,
            "memory_usage": 120.0,
        }
        
        # Compute performance metrics
        metrics = compute_performance_metrics(performance1, performance2)
        
        # Check that all metrics were computed
        assert "training_time_ratio" in metrics
        assert "inference_time_ratio" in metrics
        assert "memory_usage_ratio" in metrics
        
        # Check that metrics are reasonable
        assert metrics["training_time_ratio"] > 0
        assert metrics["inference_time_ratio"] > 0
        assert metrics["memory_usage_ratio"] > 0
        
        # Check that ratios are calculated correctly
        assert np.isclose(metrics["training_time_ratio"], performance2["training_time"] / performance1["training_time"])
        assert np.isclose(metrics["inference_time_ratio"], performance2["inference_time"] / performance1["inference_time"])
        assert np.isclose(metrics["memory_usage_ratio"], performance2["memory_usage"] / performance1["memory_usage"])
