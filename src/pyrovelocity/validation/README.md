# PyroVelocity Validation Framework

This module provides a framework for validating and comparing different implementations of PyroVelocity (legacy, modular, and JAX).

## Overview

The validation framework consists of the following components:

- **Framework**: Core validation framework for running and comparing different implementations
- **Metrics**: Validation metrics for measuring the similarity between implementations
- **Comparison**: Utilities for comparing results between implementations
- **Visualization**: Tools for visualizing comparison results

## Usage

### Basic Usage

```python
import anndata as ad
from pyrovelocity.io.datasets import pancreas
from pyrovelocity.validation.framework import run_validation

# Load data
adata = pancreas()

# Run validation
results = run_validation(
    adata=adata,
    max_epochs=10,
    num_samples=30,
    use_legacy=True,
    use_modular=True,
    use_jax=True,
)

# Extract results and comparison
model_results = results["results"]
comparison = results["comparison"]
```

### Advanced Usage

```python
import anndata as ad
from pyrovelocity.io.datasets import pancreas
from pyrovelocity.validation.framework import ValidationRunner
from pyrovelocity.validation.visualization import plot_parameter_comparison

# Load data
adata = pancreas()

# Initialize ValidationRunner
runner = ValidationRunner(adata)

# Set up models
runner.setup_legacy_model(model_type="deterministic")
runner.setup_modular_model(model_type="standard")
runner.setup_jax_model(model_type="standard")

# Run validation
results = runner.run_validation(
    max_epochs=10,
    num_samples=30,
    use_scalene=True,
)

# Compare implementations
comparison = runner.compare_implementations()

# Visualize comparison
fig = plot_parameter_comparison(comparison["parameter_comparison"])
```

## Components

### Framework

The framework module provides the following components:

- `ValidationRunner`: A class for running validation on different implementations
- `run_validation`: A function for running validation with default settings
- `compare_implementations`: A function for comparing implementation results

### Metrics

The metrics module provides the following metrics:

- **Parameter Comparison Metrics**:
  - Mean Squared Error (MSE)
  - Correlation
  - KL Divergence
  - Wasserstein Distance

- **Velocity Comparison Metrics**:
  - Mean Squared Error
  - Correlation
  - Cosine Similarity
  - Magnitude Similarity

- **Uncertainty Comparison Metrics**:
  - Mean Squared Error
  - Correlation
  - Distribution Similarity

- **Performance Comparison Metrics**:
  - Training Time Ratio
  - Inference Time Ratio
  - Memory Usage Ratio

### Comparison

The comparison module provides the following utilities:

- **Parameter Comparison**:
  - `compare_parameters`: Compare model parameters between implementations

- **Velocity Comparison**:
  - `compare_velocities`: Compare velocity estimates between implementations

- **Uncertainty Comparison**:
  - `compare_uncertainties`: Compare uncertainty estimates between implementations

- **Performance Comparison**:
  - `compare_performance`: Compare performance metrics between implementations

- **Statistical Comparison**:
  - `statistical_comparison`: Perform statistical comparison between arrays
  - `detect_outliers`: Detect outliers in comparison results
  - `detect_systematic_bias`: Detect systematic bias in comparison results
  - `identify_edge_cases`: Identify edge cases in comparison results

### Visualization

The visualization module provides the following tools:

- **Parameter Visualization**:
  - `plot_parameter_comparison`: Plot parameter comparison results
  - `plot_parameter_distributions`: Plot parameter distributions

- **Velocity Visualization**:
  - `plot_velocity_comparison`: Plot velocity comparison results
  - `plot_velocity_vector_field`: Plot velocity vector field

- **Uncertainty Visualization**:
  - `plot_uncertainty_comparison`: Plot uncertainty comparison results
  - `plot_uncertainty_heatmap`: Plot uncertainty heatmap

- **Performance Visualization**:
  - `plot_performance_comparison`: Plot performance comparison results
  - `plot_performance_radar`: Plot performance radar chart
