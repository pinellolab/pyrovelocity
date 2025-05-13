# PyroVelocity Validation Bridge

This document explains the architectural differences between the legacy and modular implementations of PyroVelocity and how the validation framework bridges these differences.

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

### 3. Velocity Computation

- **Legacy Implementation**:
  - Computes velocity as `beta * ut / scale - gamma * st`
  - Stores it in the AnnData object
  - Uses `compute_mean_vector_field()` to compute embedded velocity

- **Modular Implementation**:
  - Computes velocity using a similar formula
  - Returns it directly
  - Provides separate methods for computing embedded velocity

## Bridging the Gap

To bridge these architectural differences, we've created:

### 1. Legacy Adapter Functions

- `get_velocity_from_legacy_model()`: Extracts velocity from a legacy model by:
  - Calling `compute_statistics_from_posterior_samples()` to compute velocity
  - Extracting it from `adata.layers["velocity_pyro"]`
  - Returning it as a numpy array

- `get_velocity_uncertainty_from_legacy_model()`: Extracts uncertainty from a legacy model by:
  - Using the FDR values computed during `vector_field_uncertainty` as a proxy for uncertainty
  - Or computing standard deviation across posterior samples if FDR values are not available
  - Returning it as a numpy array

### 2. Updated Validation Framework

- `ImprovedValidationRunner`: An improved version of the validation framework that:
  - Uses the legacy adapter functions to extract velocity and uncertainty from the legacy model
  - Handles the differences in APIs between the legacy and modular implementations
  - Provides a consistent interface for comparing results

### 3. Direct Comparison Script

- `direct_comparison.py`: A script that directly compares the legacy and modular implementations by:
  - Training both models on the same data
  - Using the legacy adapter functions to extract velocity and uncertainty from the legacy model
  - Comparing the results

## Usage

### Running Direct Comparison

```bash
# Run direct comparison with default settings
./run_direct_comparison.sh

# Or run with specific parameters
python direct_comparison.py --max-epochs 10 --num-samples 5
```

### Running Improved Validation

```bash
# Run improved validation with default settings
./run_improved_validation.sh

# Or run with specific parameters
python run_improved_validation.py --max-epochs 10 --num-samples 5
```

## Results

The validation results are saved to:

- `direct_comparison_results/`: Results from the direct comparison
- `improved_validation_results/`: Results from the improved validation

Each directory contains:

- `comparison_summary_report.txt`: A summary of the comparison results
- Other files with detailed results

## Conclusion

By bridging the architectural differences between the legacy and modular implementations, we can properly validate that the modular implementation produces results equivalent to the legacy implementation. This validation is necessary for safely deprecating the legacy code and proceeding with the JAX implementation refinement.
