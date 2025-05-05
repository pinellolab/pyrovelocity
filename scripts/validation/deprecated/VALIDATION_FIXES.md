# PyroVelocity Validation Framework Fixes

This document describes the issues identified in the PyroVelocity validation framework and the fixes implemented to address them.

## Identified Issues

1. **Legacy Model Validation Issues**:
   - The `setup_legacy_model` method in `ValidationRunner` doesn't explicitly set `validation_fraction=0`
   - The legacy model has issues with the validation dataloader

2. **Modular Model Initialization Issues**:
   - The `setup_modular_model` method is very basic and doesn't pass any configuration parameters
   - It doesn't handle errors properly when required keys are missing

3. **Comparison with Failed Models**:
   - The `compare_implementations` method doesn't properly handle cases where models fail to train
   - Error handling in the validation scripts needs improvement

## Implemented Fixes

### 1. Fixed Scripts

We've created the following scripts to address these issues:

1. **pancreas_fixture_validation.py**:
   - Uses the preprocessed pancreas fixture data
   - Implements fixes for the legacy and modular model setup
   - Improves error handling during validation
   - Handles failed models in comparison

2. **improved_framework.py**:
   - Contains an improved version of the `ValidationRunner` class
   - Fixes all the identified issues
   - Can be used as a drop-in replacement for the core validation framework

3. **run_improved_validation.py**:
   - Uses the improved validation framework
   - Provides a simple interface for running validation

4. **fix_validation_framework.py**:
   - Creates a patch file with the fixes
   - Provides instructions for applying the fixes to the core validation framework

### 2. Key Fixes

#### Legacy Model Setup

```python
def setup_legacy_model(self, **kwargs) -> None:
    try:
        # Set up AnnData for legacy model
        PyroVelocity.setup_anndata(self.adata)

        # Create legacy model with validation_fraction=0 to avoid validation dataloader issues
        model_kwargs = kwargs.copy()
        # Explicitly set validation_fraction to 0
        model_kwargs["validation_fraction"] = 0

        # Create legacy model
        model = PyroVelocity(self.adata, **model_kwargs)

        # Add model to ValidationRunner
        self.add_model("legacy", model)
        print("Legacy model setup successful")
    except Exception as e:
        print(f"Error setting up legacy model: {e}")
        import traceback
        traceback.print_exc()
        # Return None to indicate failure
        return None
```

#### Modular Model Setup

```python
def setup_modular_model(self, **kwargs) -> None:
    try:
        # Set up AnnData for modular model
        PyroVelocityModel.setup_anndata(self.adata.copy())

        # Create standard model with explicit configuration
        model_type = kwargs.get("model_type", "standard")
        
        if model_type == "standard":
            model = create_standard_model()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Add model to ValidationRunner
        self.add_model("modular", model)
        print("Modular model setup successful")
    except Exception as e:
        print(f"Error setting up modular model: {e}")
        import traceback
        traceback.print_exc()
        # Return None to indicate failure
        return None
```

#### Run Validation Error Handling

```python
# Train model
try:
    if name == "legacy":
        # For legacy model, we need to remove validation_fraction
        # to avoid issues with the validation dataloader
        legacy_kwargs = kwargs.copy()
        if "validation_fraction" in legacy_kwargs:
            del legacy_kwargs["validation_fraction"]
        model.train(max_epochs=max_epochs, **legacy_kwargs)
    elif name == "modular":
        model.train(adata=self.adata, max_epochs=max_epochs, **kwargs)
    # ...
except Exception as e:
    print(f"Error training {name} model: {e}")
    import traceback
    traceback.print_exc()
    # Store error in results
    self.results[name]["error"] = str(e)
    self.results[name]["traceback"] = traceback.format_exc()
    # Skip the rest of the processing for this model
    continue
```

#### Compare Implementations with Failed Models

```python
# Check if any models failed
failed_models = []
for model_name, model_result in self.results.items():
    if "error" in model_result:
        failed_models.append(model_name)

# If any models failed, return a minimal comparison result
if failed_models:
    print(f"Skipping detailed comparison due to failed models: {failed_models}")
    return {"failed_models": failed_models}
```

## Usage

### Using the Improved Validation Framework

```bash
# Run validation with default settings
python run_improved_validation.py

# Run validation with specific parameters
python run_improved_validation.py --max-epochs 20 --num-samples 10
```

### Using the Pancreas Fixture Validation Script

```bash
# Run validation with default settings
python pancreas_fixture_validation.py

# Run validation with specific parameters
python pancreas_fixture_validation.py --max-epochs 20 --num-samples 10
```

### Applying Fixes to Core Validation Framework

1. Create the patch file:
   ```bash
   python fix_validation_framework.py
   ```

2. Manually apply the changes from the patch file to `src/pyrovelocity/validation/framework.py`

## Recommendations

1. Use the `pancreas_fixture_validation.py` script for immediate validation
2. Use the `improved_framework.py` for more complex validation scenarios
3. Apply the fixes to the core validation framework for long-term use
