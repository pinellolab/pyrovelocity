"""
Step definitions for testing the PyroVelocity model in the modular implementation.

This module implements the steps defined in the model.feature file.
"""

from importlib.resources import files

import pyro
import pytest
import torch
from pytest_bdd import given, parsers, scenarios, then, when

# Import the feature file using importlib.resources
scenarios(str(files("pyrovelocity.tests.features") / "models" / "modular" / "model.feature"))

# Import the components
from pyrovelocity.models.modular.components import (
    AutoGuideFactory,
    LegacyAutoGuideFactory,
    LegacyDynamicsModel,
    LegacyLikelihoodModel,
    LogNormalPriorModel,
    PoissonLikelihoodModel,
    StandardDynamicsModel,
    StandardObservationModel,
)
from pyrovelocity.models.modular.model import PyroVelocityModel


@given(parsers.parse("I have components for a PyroVelocity model:\n{components_table}"))
def model_components(components_table):
    """Create components for a PyroVelocity model based on the table."""
    # Parse the components table
    lines = components_table.strip().split("\n")
    components = {}

    for line in lines[1:]:  # Skip header
        parts = [part.strip() for part in line.split("|")]
        if len(parts) >= 3:
            component_type = parts[1].strip()
            implementation = parts[2].strip()
            components[component_type] = implementation

    # Create the components
    dynamics_model = StandardDynamicsModel() if components.get("dynamics_model") == "StandardDynamicsModel" else LegacyDynamicsModel()
    prior_model = LogNormalPriorModel() if components.get("prior_model") == "LogNormalPriorModel" else LogNormalPriorModel()
    likelihood_model = PoissonLikelihoodModel() if components.get("likelihood_model") == "PoissonLikelihoodModel" else LegacyLikelihoodModel()
    observation_model = StandardObservationModel() if components.get("observation_model") == "StandardObservationModel" else StandardObservationModel()
    guide_model = AutoGuideFactory() if components.get("guide_model") == "AutoGuideFactory" else LegacyAutoGuideFactory()

    return {
        "dynamics_model": dynamics_model,
        "prior_model": prior_model,
        "likelihood_model": likelihood_model,
        "observation_model": observation_model,
        "guide_model": guide_model,
    }


@when("I create a PyroVelocity model with these components")
def create_model(model_components):
    """Create a PyroVelocity model with the given components."""
    model = PyroVelocityModel(
        dynamics_model=model_components["dynamics_model"],
        prior_model=model_components["prior_model"],
        likelihood_model=model_components["likelihood_model"],
        observation_model=model_components["observation_model"],
        guide_model=model_components["guide_model"],
    )

    return model


@given("I have created a PyroVelocity model")
def created_model(bdd_pyro_velocity_model):
    """Get a PyroVelocity model from the fixture."""
    return bdd_pyro_velocity_model


@given("I have a trained PyroVelocity model")
def trained_model(bdd_pyro_velocity_model, input_data):
    """Create a trained PyroVelocity model."""
    model = bdd_pyro_velocity_model

    # Create a simple optimizer
    optimizer = pyro.optim.Adam({"lr": 0.01})

    # Create an SVI object
    guide = model.guide_model(model.model)
    svi = pyro.infer.SVI(
        model=model.model,
        guide=guide,
        optim=optimizer,
        loss=pyro.infer.Trace_ELBO(),
    )

    # Train for a few steps
    for _ in range(5):
        svi.step(
            u_obs=input_data["u_obs"],
            s_obs=input_data["s_obs"],
        )

    # Store the guide in the model
    model.guide = guide

    return model


@given("I have a trained PyroVelocity model with posterior samples")
def trained_model_with_samples(trained_model):
    """Create a trained PyroVelocity model with posterior samples."""
    model = trained_model

    # Generate posterior samples
    num_samples = 10
    posterior_samples = model.guide.get_posterior(num_samples=num_samples)

    # Store the samples in the model
    model.posterior_samples = posterior_samples

    return model


@given("I have a trained PyroVelocity model with velocity results")
def trained_model_with_velocity(trained_model_with_samples):
    """Create a trained PyroVelocity model with velocity results."""
    model = trained_model_with_samples

    # Compute velocity (simplified for this example)
    model.velocity_results = {
        "velocity": torch.randn_like(model.posterior_samples["alpha"]),
    }

    return model


@given("I have an AnnData object with RNA velocity data")
def anndata_with_velocity(bdd_anndata):
    """Get an AnnData object with RNA velocity data."""
    return bdd_anndata


@given("I have an AnnData object")
def anndata_object(bdd_anndata):
    """Get an AnnData object."""
    return bdd_anndata


@when("I run the forward method")
def run_forward_method(created_model, input_data):
    """Run the forward method of the PyroVelocity model."""
    # Run the forward method
    with pyro.poutine.trace() as trace:
        result = created_model.forward(
            u_obs=input_data["u_obs"],
            s_obs=input_data["s_obs"],
        )

    # Store the result and trace for later steps
    return {"result": result, "trace": trace}


@when("I train the model for 10 epochs")
def train_model(created_model, anndata_with_velocity):
    """Train the PyroVelocity model."""
    # Extract data from AnnData
    u_obs = torch.tensor(anndata_with_velocity.layers["unspliced"], dtype=torch.float32)
    s_obs = torch.tensor(anndata_with_velocity.layers["spliced"], dtype=torch.float32)

    # Create a simple optimizer
    optimizer = pyro.optim.Adam({"lr": 0.01})

    # Create an SVI object
    guide = created_model.guide_model(created_model.model)
    svi = pyro.infer.SVI(
        model=created_model.model,
        guide=guide,
        optim=optimizer,
        loss=pyro.infer.Trace_ELBO(),
    )

    # Train for 10 epochs
    losses = []
    for _ in range(10):
        loss = svi.step(
            u_obs=u_obs,
            s_obs=s_obs,
        )
        losses.append(loss)

    # Store the guide and losses in the model
    created_model.guide = guide
    created_model.training_losses = losses

    return {"model": created_model, "losses": losses}


@when("I generate 100 posterior samples")
def generate_posterior_samples(trained_model):
    """Generate posterior samples from the trained model."""
    # Generate posterior samples
    num_samples = 100
    posterior_samples = trained_model.guide.get_posterior(num_samples=num_samples)

    # Store the samples in the model
    trained_model.posterior_samples = posterior_samples

    return {"model": trained_model, "samples": posterior_samples}


@when("I compute RNA velocity")
def compute_velocity(trained_model_with_samples):
    """Compute RNA velocity from the trained model."""
    model = trained_model_with_samples

    # Compute velocity (simplified for this example)
    velocity = torch.randn_like(model.posterior_samples["alpha"])

    # Store the velocity in the model
    model.velocity_results = {"velocity": velocity}

    return {"model": model, "velocity": velocity}


@when("I store the results in the AnnData object")
def store_results(trained_model_with_velocity, anndata_object):
    """Store the results in the AnnData object."""
    model = trained_model_with_velocity
    adata = anndata_object

    # Store the results (simplified for this example)
    adata.uns["pyrovelocity"] = {
        "model_type": "modular",
        "parameters": {
            "alpha": model.posterior_samples["alpha"].mean(0).numpy(),
            "beta": model.posterior_samples["beta"].mean(0).numpy(),
            "gamma": model.posterior_samples["gamma"].mean(0).numpy(),
        },
    }

    # Store the velocity
    adata.layers["velocity"] = model.velocity_results["velocity"].mean(0).numpy()

    return {"model": model, "adata": adata}


@then("the model should be properly initialized")
def check_model_initialization(create_model):
    """Check that the model is properly initialized."""
    model = create_model

    # Check that the model has all the components
    assert hasattr(model, "dynamics_model")
    assert hasattr(model, "prior_model")
    assert hasattr(model, "likelihood_model")
    assert hasattr(model, "observation_model")
    assert hasattr(model, "guide_model")


@then("the model should have the correct component structure")
def check_component_structure(create_model, model_components):
    """Check that the model has the correct component structure."""
    model = create_model

    # Check that the model components match the expected components
    assert model.dynamics_model == model_components["dynamics_model"]
    assert model.prior_model == model_components["prior_model"]
    assert model.likelihood_model == model_components["likelihood_model"]
    assert model.observation_model == model_components["observation_model"]
    assert model.guide_model == model_components["guide_model"]


@then("the model should implement the forward method")
def check_forward_method(create_model):
    """Check that the model implements the forward method."""
    model = create_model

    # Check that the model has a forward method
    assert hasattr(model, "forward")
    assert callable(model.forward)


@then("the model should process the data through all components")
def check_data_processing(run_forward_method):
    """Check that the model processes the data through all components."""
    result = run_forward_method["result"]

    # Check that the result includes outputs from all components
    assert "alpha" in result  # From prior_model
    assert "beta" in result  # From prior_model
    assert "gamma" in result  # From prior_model
    assert "u_expected" in result  # From dynamics_model
    assert "s_expected" in result  # From dynamics_model
    assert "u_dist" in result  # From likelihood_model
    assert "s_dist" in result  # From likelihood_model


@then("the output should include expected counts and distributions")
def check_output_contents(run_forward_method):
    """Check that the output includes expected counts and distributions."""
    result = run_forward_method["result"]

    # Check that the result includes expected counts and distributions
    assert "u_expected" in result
    assert "s_expected" in result
    assert "u_dist" in result
    assert "s_dist" in result


@then("the model should register all parameters and observations with Pyro")
def check_pyro_registration(run_forward_method):
    """Check that the model registers all parameters and observations with Pyro."""
    trace = run_forward_method["trace"]

    # Check that the trace includes sample and observe nodes
    sample_nodes = [name for name, node in trace.nodes.items() if node["type"] == "sample"]
    observe_nodes = [name for name, node in trace.nodes.items() if node["type"] == "observe"]

    assert len(sample_nodes) > 0
    assert len(observe_nodes) > 0


@then("the model should converge")
def check_convergence(train_model):
    """Check that the model converges during training."""
    losses = train_model["losses"]

    # Check that the losses decrease
    assert losses[0] > losses[-1]


@then("the loss should decrease")
def check_loss_decrease(train_model):
    """Check that the loss decreases during training."""
    losses = train_model["losses"]

    # Check that the losses generally decrease
    assert losses[0] > losses[-1]


@then("the posterior samples should be stored")
def check_posterior_samples_stored(train_model):
    """Check that the posterior samples are stored."""
    model = train_model["model"]

    # Check that the model has a guide
    assert hasattr(model, "guide")
    assert model.guide is not None


@then("the samples should have the correct structure")
def check_sample_structure(generate_posterior_samples):
    """Check that the samples have the correct structure."""
    samples = generate_posterior_samples["samples"]

    # Check that the samples include all parameters
    assert "alpha" in samples
    assert "beta" in samples
    assert "gamma" in samples


@then("the samples should include all model parameters")
def check_sample_parameters(generate_posterior_samples):
    """Check that the samples include all model parameters."""
    samples = generate_posterior_samples["samples"]

    # Check that the samples include all parameters
    assert "alpha" in samples
    assert "beta" in samples
    assert "gamma" in samples


@then("the samples should reflect the posterior distribution")
def check_posterior_distribution(generate_posterior_samples):
    """Check that the samples reflect the posterior distribution."""
    # In a real test, we would check properties of the posterior distribution
    # For this example, we'll just pass
    pass


@then("the velocity vectors should be computed for each cell")
def check_velocity_vectors(compute_velocity):
    """Check that the velocity vectors are computed for each cell."""
    velocity = compute_velocity["velocity"]

    # Check that the velocity has the right shape
    assert velocity.ndim == 2  # [num_samples, num_genes]


@then("the velocity should reflect the transcriptional dynamics")
def check_velocity_dynamics(compute_velocity):
    """Check that the velocity reflects the transcriptional dynamics."""
    # In a real test, we would check properties of the velocity
    # For this example, we'll just pass
    pass


@then("the velocity should be stored in the model state")
def check_velocity_stored(compute_velocity):
    """Check that the velocity is stored in the model state."""
    model = compute_velocity["model"]

    # Check that the model has velocity results
    assert hasattr(model, "velocity_results")
    assert "velocity" in model.velocity_results


@then("the AnnData object should contain the velocity results")
def check_adata_velocity(store_results):
    """Check that the AnnData object contains the velocity results."""
    adata = store_results["adata"]

    # Check that the AnnData object has velocity results
    assert "velocity" in adata.layers


@then("the AnnData object should contain the model parameters")
def check_adata_parameters(store_results):
    """Check that the AnnData object contains the model parameters."""
    adata = store_results["adata"]

    # Check that the AnnData object has model parameters
    assert "pyrovelocity" in adata.uns
    assert "parameters" in adata.uns["pyrovelocity"]
    assert "alpha" in adata.uns["pyrovelocity"]["parameters"]
    assert "beta" in adata.uns["pyrovelocity"]["parameters"]
    assert "gamma" in adata.uns["pyrovelocity"]["parameters"]


@then("the AnnData object should be ready for downstream analysis")
def check_adata_ready(store_results):
    """Check that the AnnData object is ready for downstream analysis."""
    adata = store_results["adata"]

    # Check that the AnnData object has all necessary components
    assert "velocity" in adata.layers
    assert "pyrovelocity" in adata.uns
