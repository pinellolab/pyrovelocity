#!/usr/bin/env python
"""
Fix Validation Framework Script for PyroVelocity.

This script applies the fixes from the improved validation framework to the
core validation framework in pyrovelocity/validation/framework.py.

The fixes include:
1. Better error handling in setup_legacy_model and setup_modular_model
2. Explicit handling of validation_fraction in legacy model
3. Better error handling in run_validation
4. Handling of failed models in compare_implementations

Example usage:
    python fix_validation_framework.py
"""

import os
import sys
import argparse

# Add the src directory to the path to import test fixtures
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Fix Validation Framework Script for PyroVelocity"
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create a backup of the original framework.py file",
    )
    return parser.parse_args()


def fix_setup_legacy_model():
    """Fix the setup_legacy_model method in ValidationRunner."""
    return """
    @beartype
    def setup_legacy_model(self, **kwargs) -> None:
        \"\"\"
        Set up the legacy PyroVelocity model.

        Args:
            **kwargs: Keyword arguments for PyroVelocity constructor
        \"\"\"
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
    """


def fix_setup_modular_model():
    """Fix the setup_modular_model method in ValidationRunner."""
    return """
    @beartype
    def setup_modular_model(self, **kwargs) -> None:
        \"\"\"
        Set up the modular PyroVelocity model.

        Args:
            **kwargs: Additional keyword arguments for model creation
        \"\"\"
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
    """


def fix_run_validation():
    """Fix the run_validation method in ValidationRunner."""
    return """
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
                elif name == "jax":
                    # For JAX model, we need to prepare data from AnnData
                    from pyrovelocity.models.jax.data.anndata import (
                        prepare_anndata,
                    )

                    # Prepare data from AnnData
                    data_dict = prepare_anndata(
                        self.adata,
                        spliced_layer="spliced",
                        unspliced_layer="unspliced",
                    )

                    # Create inference config
                    inference_config = {
                        "num_warmup": max_epochs // 2,
                        "num_samples": max_epochs // 2,
                        "num_chains": 1,
                    }

                    # Run inference
                    import jax

                    from pyrovelocity.models.jax.inference.unified import (
                        run_inference,
                    )

                    # Create a JAX random key from the seed
                    key = jax.random.PRNGKey(kwargs.get("seed", 0))

                    _, inference_state = run_inference(
                        model=model,
                        args=(),
                        kwargs=data_dict,
                        config=inference_config,
                        key=key,
                    )

                    # Store inference state
                    self.results[name]["inference_state"] = inference_state
            except Exception as e:
                print(f"Error training {name} model: {e}")
                import traceback
                traceback.print_exc()
                # Store error in results
                self.results[name]["error"] = str(e)
                self.results[name]["traceback"] = traceback.format_exc()
                # Skip the rest of the processing for this model
                continue
    """


def fix_compare_implementations():
    """Fix the compare_implementations method in ValidationRunner."""
    return """
    @beartype
    def compare_implementations(self) -> Dict[str, Any]:
        \"\"\"
        Compare the results of different implementations after running validation.

        This method compares the results of different PyroVelocity implementations
        that have been validated using the run_validation method. It compares:
        1. Parameters: alpha, beta, gamma estimates
        2. Velocities: Velocity vectors
        3. Uncertainties: Uncertainty estimates
        4. Performance: Training and inference times

        The method delegates to the comparison utilities in the validation.comparison
        module to perform the actual comparisons.

        Returns:
            Dictionary of comparison results
        \"\"\"
        # Check that results are available
        if not self.results:
            raise ValueError("No results available. Run validation first.")

        # Check if any models failed
        failed_models = []
        for model_name, model_result in self.results.items():
            if "error" in model_result:
                failed_models.append(model_name)

        # If any models failed, return a minimal comparison result
        if failed_models:
            print(f"Skipping detailed comparison due to failed models: {failed_models}")
            return {"failed_models": failed_models}

        # Get model names
        model_names = list(self.results.keys())

        # Check that we have at least two models to compare
        if len(model_names) < 2:
            raise ValueError("Need at least two models to compare.")

        # Initialize comparison results
        comparison_results = {}

        # Compare parameters
        parameter_comparison = {}
        for param in ["alpha", "beta", "gamma"]:
            param_comparison = {}
            for i in range(len(model_names)):
                for j in range(i + 1, len(model_names)):
                    model1 = model_names[i]
                    model2 = model_names[j]
                    comp_name = f"{model1}_vs_{model2}"

                    # Get parameter samples
                    param_samples1 = self.results[model1]["posterior_samples"][param]
                    param_samples2 = self.results[model2]["posterior_samples"][param]

                    # Compare parameters
                    param_metrics = compare_parameters(
                        {param: param_samples1}, {param: param_samples2}
                    )[param]

                    # Store comparison results
                    param_comparison[comp_name] = param_metrics

            # Store parameter comparison results
            parameter_comparison[param] = param_comparison

        # Store parameter comparison results
        comparison_results["parameter_comparison"] = parameter_comparison

        # Compare velocities
        velocity_comparison = {}
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                model1 = model_names[i]
                model2 = model_names[j]
                comp_name = f"{model1}_vs_{model2}"

                # Get velocities
                velocity1 = self.results[model1]["velocity"]
                velocity2 = self.results[model2]["velocity"]

                # Compare velocities
                velocity_metrics = compare_velocities(velocity1, velocity2)

                # Store comparison results
                velocity_comparison[comp_name] = velocity_metrics

        # Store velocity comparison results
        comparison_results["velocity_comparison"] = velocity_comparison

        # Compare uncertainties
        uncertainty_comparison = {}
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                model1 = model_names[i]
                model2 = model_names[j]
                comp_name = f"{model1}_vs_{model2}"

                # Get uncertainties
                uncertainty1 = self.results[model1]["uncertainty"]
                uncertainty2 = self.results[model2]["uncertainty"]

                # Compare uncertainties
                uncertainty_metrics = compare_uncertainties(uncertainty1, uncertainty2)

                # Store comparison results
                uncertainty_comparison[comp_name] = uncertainty_metrics

        # Store uncertainty comparison results
        comparison_results["uncertainty_comparison"] = uncertainty_comparison

        # Compare performance
        performance_comparison = {}
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                model1 = model_names[i]
                model2 = model_names[j]
                comp_name = f"{model1}_vs_{model2}"

                # Get performance metrics
                performance1 = self.results[model1]["performance"]
                performance2 = self.results[model2]["performance"]

                # Compare performance
                performance_metrics = compare_performance(performance1, performance2)

                # Store comparison results
                performance_comparison[comp_name] = performance_metrics

        # Store performance comparison results
        comparison_results["performance_comparison"] = performance_comparison

        # Return comparison results
        return comparison_results
    """


def create_patch_file():
    """Create a patch file with the fixes."""
    patch_content = """# Patch file for fixing the validation framework
# Apply this patch to pyrovelocity/validation/framework.py

# Fix for setup_legacy_model
def setup_legacy_model(self, **kwargs) -> None:
    \"\"\"
    Set up the legacy PyroVelocity model.

    Args:
        **kwargs: Keyword arguments for PyroVelocity constructor
    \"\"\"
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

# Fix for setup_modular_model
def setup_modular_model(self, **kwargs) -> None:
    \"\"\"
    Set up the modular PyroVelocity model.

    Args:
        **kwargs: Additional keyword arguments for model creation
    \"\"\"
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

# Fix for run_validation
# Replace the training section with:
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
            elif name == "jax":
                # For JAX model, we need to prepare data from AnnData
                from pyrovelocity.models.jax.data.anndata import (
                    prepare_anndata,
                )

                # Prepare data from AnnData
                data_dict = prepare_anndata(
                    self.adata,
                    spliced_layer="spliced",
                    unspliced_layer="unspliced",
                )

                # Create inference config
                inference_config = {
                    "num_warmup": max_epochs // 2,
                    "num_samples": max_epochs // 2,
                    "num_chains": 1,
                }

                # Run inference
                import jax

                from pyrovelocity.models.jax.inference.unified import (
                    run_inference,
                )

                # Create a JAX random key from the seed
                key = jax.random.PRNGKey(kwargs.get("seed", 0))

                _, inference_state = run_inference(
                    model=model,
                    args=(),
                    kwargs=data_dict,
                    config=inference_config,
                    key=key,
                )

                # Store inference state
                self.results[name]["inference_state"] = inference_state
        except Exception as e:
            print(f"Error training {name} model: {e}")
            import traceback
            traceback.print_exc()
            # Store error in results
            self.results[name]["error"] = str(e)
            self.results[name]["traceback"] = traceback.format_exc()
            # Skip the rest of the processing for this model
            continue

# Fix for compare_implementations
# Replace the beginning of the method with:
    # Check that results are available
    if not self.results:
        raise ValueError("No results available. Run validation first.")

    # Check if any models failed
    failed_models = []
    for model_name, model_result in self.results.items():
        if "error" in model_result:
            failed_models.append(model_name)

    # If any models failed, return a minimal comparison result
    if failed_models:
        print(f"Skipping detailed comparison due to failed models: {failed_models}")
        return {"failed_models": failed_models}
"""
    
    with open("validation_framework_fixes.patch", "w") as f:
        f.write(patch_content)
    
    print("Patch file created: validation_framework_fixes.patch")


def main():
    """Create the patch file with the fixes."""
    args = parse_args()
    
    # Create the patch file
    create_patch_file()
    
    print("\nTo apply these fixes to the core validation framework, you can:")
    print("1. Manually apply the changes from the patch file")
    print("2. Use the improved validation framework directly")
    print("\nThe improved validation framework is available in:")
    print("- improved_framework.py: Contains the ImprovedValidationRunner class")
    print("- run_improved_validation.py: Script to run validation with the improved framework")
    print("- pancreas_fixture_validation.py: Script to validate using the preprocessed pancreas fixture")


if __name__ == "__main__":
    main()
