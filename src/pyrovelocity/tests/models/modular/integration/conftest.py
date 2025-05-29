"""
Fixtures for BDD testing of PyroVelocity's modular implementation.

This module provides fixtures that are specific to BDD testing, complementing
the fixtures defined in the main conftest.py file.
"""

import numpy as np
import pyro
import pytest
import torch
from anndata import AnnData

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


@pytest.fixture
def bdd_simple_data():
    """Create simple data for BDD testing."""
    # Create random data for testing
    torch.manual_seed(42)
    n_cells = 10
    n_genes = 5

    # Generate random data
    u_obs = torch.abs(torch.randn((n_cells, n_genes)))
    s_obs = torch.abs(torch.randn((n_cells, n_genes)))

    return {
        "u_obs": u_obs,
        "s_obs": s_obs,
        "n_cells": n_cells,
        "n_genes": n_genes,
    }


@pytest.fixture
def bdd_model_parameters():
    """Create model parameters for BDD testing."""
    torch.manual_seed(42)
    n_genes = 5

    # Generate random parameters
    alpha = torch.abs(torch.randn(n_genes))
    beta = torch.abs(torch.randn(n_genes))
    gamma = torch.abs(torch.randn(n_genes))

    return {
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
    }


@pytest.fixture
def bdd_standard_dynamics_model():
    """Create a StandardDynamicsModel for BDD testing."""
    return StandardDynamicsModel()


@pytest.fixture
def bdd_legacy_dynamics_model():
    """Create a LegacyDynamicsModel for BDD testing."""
    return LegacyDynamicsModel()


@pytest.fixture
def bdd_lognormal_prior_model():
    """Create a LogNormalPriorModel for BDD testing."""
    return LogNormalPriorModel()


@pytest.fixture
def bdd_poisson_likelihood_model():
    """Create a PoissonLikelihoodModel for BDD testing."""
    return PoissonLikelihoodModel()


@pytest.fixture
def bdd_legacy_likelihood_model():
    """Create a LegacyLikelihoodModel for BDD testing."""
    return LegacyLikelihoodModel()


@pytest.fixture
def bdd_standard_observation_model():
    """Create a StandardObservationModel for BDD testing."""
    return StandardObservationModel()


@pytest.fixture
def bdd_auto_guide_factory():
    """Create an AutoGuideFactory for BDD testing."""
    return AutoGuideFactory(guide_type="AutoNormal")


@pytest.fixture
def bdd_legacy_auto_guide_factory():
    """Create a LegacyAutoGuideFactory for BDD testing."""
    return LegacyAutoGuideFactory()


@pytest.fixture
def bdd_pyro_velocity_model(
    bdd_standard_dynamics_model,
    bdd_lognormal_prior_model,
    bdd_poisson_likelihood_model,
    bdd_auto_guide_factory,
):
    """Create a PyroVelocityModel for BDD testing."""
    return PyroVelocityModel(
        dynamics_model=bdd_standard_dynamics_model,
        prior_model=bdd_lognormal_prior_model,
        likelihood_model=bdd_poisson_likelihood_model,
        guide_model=bdd_auto_guide_factory,
    )


@pytest.fixture
def bdd_anndata():
    """Create a simple AnnData object for BDD testing."""
    # Create random data
    np.random.seed(42)
    n_cells = 10
    n_genes = 5
    
    X = np.random.rand(n_cells, n_genes)
    layers = {
        "spliced": np.random.rand(n_cells, n_genes),
        "unspliced": np.random.rand(n_cells, n_genes),
    }
    
    # Create AnnData object
    adata = AnnData(X=X, layers=layers)
    
    return adata


@pytest.fixture(autouse=True)
def clear_pyro_param_store():
    """Clear Pyro's parameter store before and after each test."""
    pyro.clear_param_store()
    yield
    pyro.clear_param_store()
