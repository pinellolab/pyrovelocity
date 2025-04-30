"""
Tests for the PyroVelocity comparison utilities.

This module contains tests for the comparison utilities, which provide tools
for comparing different implementations of PyroVelocity.
"""

import os
from typing import Dict, Any, Optional, Tuple, List, Union

import numpy as np
import pytest
import torch
import jax.numpy as jnp
import pandas as pd
from scipy import stats

# Import the comparison utilities
from pyrovelocity.validation.comparison import (
    compare_parameters,
    compare_velocities,
    compare_uncertainties,
    compare_performance,
    statistical_comparison,
    detect_outliers,
    detect_systematic_bias,
    identify_edge_cases,
)


class TestParameterComparison:
    """Tests for parameter comparison utilities."""
    
    def test_compare_parameters(self):
        """Test comparing parameters between implementations."""
        # Create test data
        results = {
            "legacy": {
                "posterior_samples": {
                    "alpha": np.random.rand(10, 5),
                    "beta": np.random.rand(10, 5),
                    "gamma": np.random.rand(10, 5),
                }
            },
            "modular": {
                "posterior_samples": {
                    "alpha": np.random.rand(10, 5),
                    "beta": np.random.rand(10, 5),
                    "gamma": np.random.rand(10, 5),
                }
            },
            "jax": {
                "posterior_samples": {
                    "alpha": np.random.rand(10, 5),
                    "beta": np.random.rand(10, 5),
                    "gamma": np.random.rand(10, 5),
                }
            }
        }
        
        # Compare parameters
        comparison = compare_parameters(results)
        
        # Check that comparison results were generated for all parameters
        assert "alpha" in comparison
        assert "beta" in comparison
        assert "gamma" in comparison
        
        # Check that comparison results include all implementation pairs
        for param in ["alpha", "beta", "gamma"]:
            assert "legacy_vs_modular" in comparison[param]
            assert "legacy_vs_jax" in comparison[param]
            assert "modular_vs_jax" in comparison[param]
        
        # Check that comparison results include all metrics
        for param in ["alpha", "beta", "gamma"]:
            for pair in ["legacy_vs_modular", "legacy_vs_jax", "modular_vs_jax"]:
                assert "mse" in comparison[param][pair]
                assert "correlation" in comparison[param][pair]
                assert "kl_divergence" in comparison[param][pair]
                assert "wasserstein_distance" in comparison[param][pair]


class TestVelocityComparison:
    """Tests for velocity comparison utilities."""
    
    def test_compare_velocities(self):
        """Test comparing velocities between implementations."""
        # Create test data
        results = {
            "legacy": {
                "velocity": np.random.rand(10, 5),
            },
            "modular": {
                "velocity": np.random.rand(10, 5),
            },
            "jax": {
                "velocity": np.random.rand(10, 5),
            }
        }
        
        # Compare velocities
        comparison = compare_velocities(results)
        
        # Check that comparison results include all implementation pairs
        assert "legacy_vs_modular" in comparison
        assert "legacy_vs_jax" in comparison
        assert "modular_vs_jax" in comparison
        
        # Check that comparison results include all metrics
        for pair in ["legacy_vs_modular", "legacy_vs_jax", "modular_vs_jax"]:
            assert "mse" in comparison[pair]
            assert "correlation" in comparison[pair]
            assert "cosine_similarity" in comparison[pair]
            assert "magnitude_similarity" in comparison[pair]


class TestUncertaintyComparison:
    """Tests for uncertainty comparison utilities."""
    
    def test_compare_uncertainties(self):
        """Test comparing uncertainties between implementations."""
        # Create test data
        results = {
            "legacy": {
                "uncertainty": np.random.rand(10, 5),
            },
            "modular": {
                "uncertainty": np.random.rand(10, 5),
            },
            "jax": {
                "uncertainty": np.random.rand(10, 5),
            }
        }
        
        # Compare uncertainties
        comparison = compare_uncertainties(results)
        
        # Check that comparison results include all implementation pairs
        assert "legacy_vs_modular" in comparison
        assert "legacy_vs_jax" in comparison
        assert "modular_vs_jax" in comparison
        
        # Check that comparison results include all metrics
        for pair in ["legacy_vs_modular", "legacy_vs_jax", "modular_vs_jax"]:
            assert "mse" in comparison[pair]
            assert "correlation" in comparison[pair]
            assert "distribution_similarity" in comparison[pair]


class TestPerformanceComparison:
    """Tests for performance comparison utilities."""
    
    def test_compare_performance(self):
        """Test comparing performance between implementations."""
        # Create test data
        results = {
            "legacy": {
                "performance": {
                    "training_time": 1.0,
                    "inference_time": 0.5,
                    "memory_usage": 100.0,
                }
            },
            "modular": {
                "performance": {
                    "training_time": 1.2,
                    "inference_time": 0.6,
                    "memory_usage": 120.0,
                }
            },
            "jax": {
                "performance": {
                    "training_time": 0.8,
                    "inference_time": 0.4,
                    "memory_usage": 80.0,
                }
            }
        }
        
        # Compare performance
        comparison = compare_performance(results)
        
        # Check that comparison results include all implementation pairs
        assert "legacy_vs_modular" in comparison
        assert "legacy_vs_jax" in comparison
        assert "modular_vs_jax" in comparison
        
        # Check that comparison results include all metrics
        for pair in ["legacy_vs_modular", "legacy_vs_jax", "modular_vs_jax"]:
            assert "training_time_ratio" in comparison[pair]
            assert "inference_time_ratio" in comparison[pair]
            assert "memory_usage_ratio" in comparison[pair]


class TestStatisticalComparison:
    """Tests for statistical comparison utilities."""
    
    def test_statistical_comparison(self):
        """Test statistical comparison between implementations."""
        # Create test data
        x = np.random.rand(100)
        y = np.random.rand(100)
        
        # Perform statistical comparison
        comparison = statistical_comparison(x, y)
        
        # Check that comparison results include all statistical tests
        assert "t_test" in comparison
        assert "wilcoxon_test" in comparison
        assert "ks_test" in comparison
        
        # Check that t-test results include p-value and statistic
        assert "statistic" in comparison["t_test"]
        assert "p_value" in comparison["t_test"]
        
        # Check that Wilcoxon test results include p-value and statistic
        assert "statistic" in comparison["wilcoxon_test"]
        assert "p_value" in comparison["wilcoxon_test"]
        
        # Check that KS test results include p-value and statistic
        assert "statistic" in comparison["ks_test"]
        assert "p_value" in comparison["ks_test"]


class TestDiscrepancyIdentification:
    """Tests for discrepancy identification utilities."""
    
    def test_detect_outliers(self):
        """Test detecting outliers in comparison results."""
        # Create test data
        data = np.random.rand(100)
        data[0] = 10.0  # Add an outlier
        
        # Detect outliers
        outliers = detect_outliers(data)
        
        # Check that outliers were detected
        assert len(outliers) > 0
        assert 0 in outliers  # The first element should be detected as an outlier
    
    def test_detect_systematic_bias(self):
        """Test detecting systematic bias in comparison results."""
        # Create test data
        x = np.random.rand(100)
        y = x + 0.1  # Add a systematic bias
        
        # Detect systematic bias
        bias = detect_systematic_bias(x, y)
        
        # Check that bias was detected
        assert bias["detected"]
        assert np.isclose(bias["mean_difference"], 0.1, atol=0.05)
    
    def test_identify_edge_cases(self):
        """Test identifying edge cases in comparison results."""
        # Create test data
        data = np.random.rand(100, 5)  # 100 cells, 5 genes
        
        # Identify edge cases
        edge_cases = identify_edge_cases(data)
        
        # Check that edge cases were identified
        assert isinstance(edge_cases, dict)
        assert "min_values" in edge_cases
        assert "max_values" in edge_cases
        assert "extreme_values" in edge_cases
