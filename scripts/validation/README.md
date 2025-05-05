# PyroVelocity Validation Framework

This directory contains scripts for validating that the PyTorch/Pyro modular implementation (specifically the standard model) produces results equivalent to the legacy implementation.

## Overview

The validation framework enables:
1. Running multiple implementations on the same data
2. Comparing parameter estimates across implementations
3. Comparing velocity estimates across implementations
4. Comparing uncertainty estimates across implementations
5. Comparing performance metrics across implementations

The framework focuses on validating the standard model from the modular implementation against the legacy implementation, using only SVI (not MCMC) as the inference method.

## Architectural Differences

### 1. Workflow vs. Method-Based Approach

- **Legacy Implementation**: Uses a workflow-based approach where:
  - `generate_posterior_samples()` creates posterior samples
  - `compute_statistics_from_posterior_samples()` processes these samples and stores results in the AnnData object
  - Results are stored in the AnnData object at `adata.layers["velocity_pyro"]` and other locations

- **Modular Implementation**: Provides direct method access through:
  - `get_velocity()` that computes velocity directly from posterior samples
  - `get_velocity_uncertainty()` that computes uncertainty directly from posterior samples
  - Returns results as numpy arrays or tensors without modifying the AnnData object

### 2. Data Storage Pattern

- **Legacy Implementation**:
  - Stores velocity in `adata.layers["velocity_pyro"]`
  - Stores embedded velocity in `adata.obsm[f"velocity_pyro_{basis}"]`
  - Stores uncertainty in posterior samples as 'fdri'

- **Modular Implementation**:
  - Returns velocity as a numpy array or tensor
  - Returns uncertainty as a numpy array or tensor
  - Does not modify the AnnData object

## Validation Scripts

### 1. Legacy Model Validation

The `validate_legacy_model.py` script validates the legacy model's workflow by:
- Training the model
- Generating posterior samples
- Computing statistics from posterior samples
- Extracting velocity from the AnnData object

```bash
# Run legacy model validation
./run_legacy_validation.sh
```

### 2. Legacy Model Visualization

The `visualize_legacy_velocity.py` script visualizes the velocity field from the legacy model by:
- Training the model
- Generating posterior samples
- Computing statistics from posterior samples
- Plotting the velocity field using scvelo

```bash
# Run legacy model visualization
./run_legacy_visualization.sh
```

### 3. Direct Comparison

The `direct_comparison.py` script attempts to directly compare the legacy and modular implementations by:
- Training both models on the same data
- Using adapter functions to extract velocity and uncertainty from the legacy model
- Comparing the results

```bash
# Run direct comparison
./run_direct_comparison.sh
```

### 4. Improved Validation Framework

The `improved_framework.py` script provides an improved validation framework that:
- Uses adapter functions to extract velocity and uncertainty from the legacy model
- Handles the differences in APIs between the legacy and modular implementations
- Provides a consistent interface for comparing results

```bash
# Run improved validation framework
./run_improved_validation.sh
```

### 5. Basic Validation

The `basic_validation.py` script provides a simple validation of the legacy and modular implementations using a small synthetic dataset or a downsampled pancreas dataset.

```bash
# Run validation with default settings
python basic_validation.py --use-legacy --use-modular

# Run validation with specific parameters
python basic_validation.py --use-legacy --use-modular --max-epochs 500 --num-samples 50
```

### 6. Legacy vs Modular Validation

The `legacy_vs_modular.py` script provides a more detailed validation of the legacy and modular implementations using a downsampled pancreas dataset.

```bash
# Run validation with default settings
python legacy_vs_modular.py

# Run validation with specific parameters
python legacy_vs_modular.py --max-epochs 500 --num-samples 50 --n-cells 200 --n-genes 200
```

### 7. Fixture Validation

The `fixture_validation.py` script validates the legacy and modular implementations using test fixtures from the PyroVelocity test suite.

```bash
# Run validation with default settings
python fixture_validation.py

# Run validation with specific parameters
python fixture_validation.py --max-epochs 500 --num-samples 50
```

### 8. Pancreas Validation

The `pancreas_validation.py` script validates the legacy and modular implementations using the pancreas dataset.

```bash
# Run validation with default settings
python pancreas_validation.py

# Run validation with specific parameters
python pancreas_validation.py --max-epochs 500 --num-samples 50 --n-cells 200 --n-genes 200
```

## Command Line Arguments

All validation scripts support the following command line arguments:

- `--max-epochs`: Maximum number of epochs for training (default: 100)
- `--num-samples`: Number of posterior samples to generate (default: 30)
- `--use-scalene`: Whether to use Scalene for performance profiling (default: False)
- `--output-dir`: Directory to save validation results (default: varies by script)
- `--normalize-method`: Method for normalizing shapes when comparing implementations (default: "nearest")
- `--target-strategy`: Strategy for determining target shape when normalizing (default: "max")
- `--learning-rate`: Learning rate for training (default: 0.01)
- `--batch-size`: Batch size for training (default: 128)
- `--use-gpu`: Whether to use GPU for training (default: False)
- `--seed`: Random seed for reproducibility (default: 42)

Some scripts also support additional arguments:

- `--n-cells`: Number of cells to use from the dataset (default: 100)
- `--n-genes`: Number of genes to use from the dataset (default: 100)

## Legacy Adapter

The `legacy_adapter.py` script provides adapter functions to bridge the architectural differences between the legacy and modular implementations:

- `get_velocity_from_legacy_model()`: Extracts velocity from a legacy model by:
  - Calling `compute_statistics_from_posterior_samples()` to compute velocity
  - Extracting it from `adata.layers["velocity_pyro"]`
  - Returning it as a numpy array

- `get_velocity_uncertainty_from_legacy_model()`: Extracts uncertainty from a legacy model by:
  - Using the FDR values computed during `vector_field_uncertainty` as a proxy for uncertainty
  - Or computing standard deviation across posterior samples if FDR values are not available
  - Returning it as a numpy array

## Validation Results

The validation scripts generate the following outputs:

1. **Text Reports**:
   - Parameter comparison report
   - Velocity comparison report
   - Uncertainty comparison report
   - Performance comparison report
   - Summary report

2. **Visualizations**:
   - Parameter comparison plots
   - Velocity comparison plots
   - Uncertainty comparison plots
   - Performance comparison plots
   - Parameter distribution plots

## Validation Metrics

The validation framework uses the following metrics to compare implementations:

1. **Parameter Metrics**:
   - Mean Squared Error (MSE)
   - Correlation
   - KL Divergence
   - Wasserstein Distance

2. **Velocity Metrics**:
   - Mean Squared Error (MSE)
   - Correlation
   - Cosine Similarity
   - Magnitude Similarity

3. **Uncertainty Metrics**:
   - Mean Squared Error (MSE)
   - Correlation
   - Distribution Similarity

4. **Performance Metrics**:
   - Training Time Ratio
   - Inference Time Ratio

## Validation Conclusion

The validation framework provides a conclusion based on the comparison results:

- If parameter estimates and velocity estimates show high correlation (>= 0.9), the modular implementation is considered a valid replacement for the legacy implementation.
- If parameter estimates or velocity estimates show low correlation (< 0.9), further investigation is needed to understand the differences.

## Example Usage

```bash
# Run basic validation with default settings
python basic_validation.py --use-legacy --use-modular

# Run legacy vs modular validation with specific parameters
python legacy_vs_modular.py --max-epochs 500 --num-samples 50 --n-cells 200 --n-genes 200

# Run fixture validation with default settings
python fixture_validation.py

# Run pancreas validation with specific parameters
python pancreas_validation.py --max-epochs 500 --num-samples 50 --n-cells 200 --n-genes 200
```

## Extending the Framework

The validation framework can be extended to validate other implementations (e.g., JAX) by:

1. Adding a new setup method to the `ValidationRunner` class
2. Updating the validation scripts to include the new implementation
3. Adding new comparison metrics if needed

For example, to validate the JAX implementation, you would:

1. Ensure the JAX implementation is available
2. Call `runner.setup_jax_model()` in the validation script
3. Run the validation with `--use-jax`
