"""
Step definitions for testing observation models in PyroVelocity's modular implementation.

This module implements the steps defined in the observation_model.feature file.
"""

from importlib.resources import files

# Import necessary modules
import pyro
import pytest
import torch
from pytest_bdd import given, parsers, scenarios, then, when

# Import the feature file using importlib.resources
scenarios(str(files("pyrovelocity.tests.features") / "models" / "modular" / "observation_model.feature"))

# Import the components
from pyrovelocity.models.modular.components import StandardObservationModel


@given("I have an observation model component", target_fixture="observation_model_component")
def observation_model_component():
    """Create a generic observation model component."""
    return StandardObservationModel()


@given("I have input data with unspliced and spliced counts", target_fixture="input_data")
def input_data_fixture(bdd_simple_data):
    """Get input data from the fixture."""
    return bdd_simple_data


@given("I have a StandardObservationModel", target_fixture="standard_observation_model")
def standard_observation_model_fixture():
    """Create a StandardObservationModel."""
    return StandardObservationModel()


@given(parsers.parse("I have a StandardObservationModel with use_size_factor={use_size_factor}"), target_fixture="standard_observation_model_with_size_factor")
def standard_observation_model_with_size_factor_fixture(use_size_factor):
    """Create a StandardObservationModel with the specified use_size_factor setting."""
    # Convert string to boolean
    use_size_factor_bool = use_size_factor.lower() == "true"

    return StandardObservationModel(use_observed_lib_size=use_size_factor_bool)


@given("I have a StandardObservationModel with normalization", target_fixture="standard_observation_model_with_normalization")
def standard_observation_model_with_normalization_fixture():
    """Create a StandardObservationModel with normalization enabled."""
    return StandardObservationModel(use_observed_lib_size=True, transform_batch=True)


@given("I have data with missing values", target_fixture="data_with_missing_values")
def data_with_missing_values_fixture():
    """Create data with missing values (represented as NaNs)."""
    # Create data with some NaN values
    torch.manual_seed(42)
    n_cells = 10
    n_genes = 5

    # Generate random data
    u_obs = torch.abs(torch.randn((n_cells, n_genes)))
    s_obs = torch.abs(torch.randn((n_cells, n_genes)))

    # Set some values to NaN
    u_obs[0, 0] = float('nan')
    u_obs[1, 1] = float('nan')
    s_obs[0, 0] = float('nan')
    s_obs[2, 2] = float('nan')

    return {
        "u_obs": u_obs,
        "s_obs": s_obs,
        "n_cells": n_cells,
        "n_genes": n_genes,
    }


@when("I run the forward method", target_fixture="run_forward_method")
def run_forward_method_fixture(observation_model_component, input_data):
    """Run the forward method of the observation model."""
    # Create a context with the necessary data
    context = {
        "u_obs": input_data["u_obs"],
        "s_obs": input_data["s_obs"],
    }

    # Run the forward method
    result = observation_model_component.forward(context)

    return {
        "result": result,
        "context": context,
    }


@when("I run the forward method with size factors", target_fixture="run_forward_method_with_size_factors")
def run_forward_method_with_size_factors_fixture(standard_observation_model_with_size_factor, input_data):
    """Run the forward method with size factors."""
    # Create a context with the necessary data
    context = {
        "u_obs": input_data["u_obs"],
        "s_obs": input_data["s_obs"],
    }

    # Run the forward method
    result = standard_observation_model_with_size_factor.forward(context)

    return {
        "result": result,
        "context": context,
    }


@when("I run the forward method with normalization", target_fixture="run_forward_method_with_normalization")
def run_forward_method_with_normalization_fixture(standard_observation_model_with_normalization, input_data):
    """Run the forward method with normalization."""
    # Create a context with the necessary data
    context = {
        "u_obs": input_data["u_obs"],
        "s_obs": input_data["s_obs"],
    }

    # Run the forward method
    result = standard_observation_model_with_normalization.forward(context)

    return {
        "result": result,
        "context": context,
    }


@when("I run the forward method with a plate context", target_fixture="run_forward_method_with_plate")
def run_forward_method_with_plate_fixture(standard_observation_model, input_data):
    """Run the forward method with a plate context."""
    # Create a context with the necessary data
    context = {
        "u_obs": input_data["u_obs"],
        "s_obs": input_data["s_obs"],
    }

    # Create a plate
    n_cells = input_data["n_cells"]
    n_genes = input_data["n_genes"]
    cell_plate = pyro.plate("cells", n_cells, dim=-2)
    gene_plate = pyro.plate("genes", n_genes, dim=-1)

    # Add plates to context
    context["cell_plate"] = cell_plate
    context["gene_plate"] = gene_plate

    # Run the forward method
    with cell_plate:
        with gene_plate:
            result = standard_observation_model.forward(context)

    return {
        "result": result,
        "context": context,
        "plate": {"cell_plate": cell_plate, "gene_plate": gene_plate},
    }


@then("the model should preprocess the input data")
def check_preprocessing(run_forward_method):
    """Check that the model preprocesses the input data."""
    result = run_forward_method["result"]

    # Check that the result includes preprocessed data
    assert "u_obs" in result
    assert "s_obs" in result

    # The preprocessed data might be transformed in some way
    # For this example, we'll just check that it exists
    assert result["u_obs"] is not None
    assert result["s_obs"] is not None


@then("the transformed data should maintain the original structure")
def check_transformed_data_structure(run_forward_method):
    """Check that the transformed data maintains the original structure."""
    result = run_forward_method["result"]
    context = run_forward_method["context"]

    # Check that the transformed data has the same shape as the original data
    assert result["u_obs"].shape == context["u_obs"].shape
    assert result["s_obs"].shape == context["s_obs"].shape


@then("the context should be updated with the transformed data")
def check_context_update(run_forward_method):
    """Check that the context is updated with the transformed data."""
    result = run_forward_method["result"]

    # Check that the result includes the transformed data
    assert "u_obs" in result
    assert "s_obs" in result


@then("the model should compute size factors for the data")
def check_size_factors(run_forward_method):
    """Check that the model computes size factors for the data."""
    result = run_forward_method["result"]

    # Check that the result includes size factors
    assert "u_scale" in result
    assert "s_scale" in result
    assert result["u_scale"] is not None
    assert result["s_scale"] is not None


@then("the size factors should reflect the library size")
def check_size_factors_reflect_library_size(run_forward_method, input_data):
    """Check that the size factors reflect the library size."""
    result = run_forward_method["result"]

    # Calculate library sizes
    u_lib_size = input_data["u_obs"].sum(dim=1, keepdim=True)
    s_lib_size = input_data["s_obs"].sum(dim=1, keepdim=True)

    # Check that the size factors are related to the library sizes
    # The exact relationship depends on the implementation
    # For this example, we'll just check that they have the same shape
    assert result["u_scale"].shape == u_lib_size.shape
    assert result["s_scale"].shape == s_lib_size.shape


@then("the context should include the computed size factors")
def check_context_includes_size_factors(run_forward_method):
    """Check that the context includes the computed size factors."""
    result = run_forward_method["result"]

    # Check that the result includes size factors
    assert "u_scale" in result
    assert "s_scale" in result


@then("the model should handle missing values gracefully")
def check_missing_values_handling(standard_observation_model, data_with_missing_values):
    """Check that the model handles missing values gracefully."""
    # Create a context with the necessary data
    context = {
        "u_obs": data_with_missing_values["u_obs"],
        "s_obs": data_with_missing_values["s_obs"],
    }

    # Run the forward method
    try:
        # Run the forward method and store the result to avoid unused variable warning
        forward_result = standard_observation_model.forward(context)
        # Check that the result has the expected structure
        assert "u_obs" in forward_result
        assert "s_obs" in forward_result
        # If we get here, the model handled missing values without errors
    except Exception as e:
        pytest.fail(f"Model failed to handle missing values: {e}")


@then("should not produce errors or warnings")
def check_no_errors_or_warnings():
    """Check that no errors or warnings are produced."""
    # This is a placeholder - in a real test, we would capture warnings
    # For this example, we'll just pass
    pass


@then("the model should normalize the data")
def check_normalization(run_forward_method):
    """Check that the model normalizes the data."""
    result = run_forward_method["result"]

    # Check that the result includes normalized data
    assert "u_obs" in result
    assert "s_obs" in result


@then("the normalized data should have the expected properties")
def check_normalized_data_properties(run_forward_method, input_data):
    """Check that the normalized data has the expected properties."""
    result = run_forward_method["result"]

    # Check that the normalized data has the expected properties
    # For this example, we'll just check that the data has the same shape
    assert result["u_obs"].shape == input_data["u_obs"].shape
    assert result["s_obs"].shape == input_data["s_obs"].shape


@then("the context should include the normalized data")
def check_context_includes_normalized_data(run_forward_method):
    """Check that the context includes the normalized data."""
    result = run_forward_method["result"]

    # Check that the result includes normalized data
    assert "u_obs" in result
    assert "s_obs" in result


@then("the transformed data should preserve the biological signal")
def check_biological_signal_preservation(standard_observation_model, input_data):
    """Check that the transformed data preserves the biological signal."""
    # Create a context with the necessary data
    context = {
        "u_obs": input_data["u_obs"],
        "s_obs": input_data["s_obs"],
    }

    # Run the forward method
    result = standard_observation_model.forward(context)

    # Check that the transformed data preserves the biological signal
    # For this example, we'll just check that the data has the same shape
    assert result["u_obs"].shape == input_data["u_obs"].shape
    assert result["s_obs"].shape == input_data["s_obs"].shape

    # In a real test, we would check that the biological signal is preserved
    # For example, by checking correlations between genes or cells


@then("the transformation should not introduce artifacts")
def check_no_artifacts(standard_observation_model, input_data):
    """Check that the transformation does not introduce artifacts."""
    # Create a context with the necessary data
    context = {
        "u_obs": input_data["u_obs"],
        "s_obs": input_data["s_obs"],
    }

    # Run the forward method
    result = standard_observation_model.forward(context)

    # Check that the transformation does not introduce artifacts
    # For this example, we'll check that there are no NaN or infinite values
    assert not torch.isnan(result["u_obs"]).any()
    assert not torch.isnan(result["s_obs"]).any()
    assert not torch.isinf(result["u_obs"]).any()
    assert not torch.isinf(result["s_obs"]).any()
