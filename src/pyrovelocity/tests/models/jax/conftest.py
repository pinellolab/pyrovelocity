"""Common test fixtures for PyroVelocity JAX/NumPyro implementation."""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
from pyrovelocity.models.jax.core.utils import create_key, split_key


@pytest.fixture
def jax_key():
    """Fixture for JAX random key."""
    return create_key(42)


@pytest.fixture
def jax_array_1d():
    """Fixture for 1D JAX array."""
    return jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])


@pytest.fixture
def jax_array_2d():
    """Fixture for 2D JAX array."""
    return jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])


@pytest.fixture
def numpy_array_1d():
    """Fixture for 1D NumPy array."""
    return np.array([1.0, 2.0, 3.0, 4.0, 5.0])


@pytest.fixture
def numpy_array_2d():
    """Fixture for 2D NumPy array."""
    return np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])


@pytest.fixture
def model_parameters():
    """Fixture for model parameters."""
    return {
        "alpha": jnp.array([1.0, 2.0, 3.0]),
        "beta": jnp.array([0.5, 1.0, 1.5]),
        "gamma": jnp.array([0.3, 0.6, 0.9]),
    }


@pytest.fixture
def cell_gene_data():
    """Fixture for cell-gene data."""
    num_cells = 10
    num_genes = 5

    # Create random data
    key = create_key(42)
    key1, key2 = split_key(key)

    u_data = jnp.abs(jax.random.normal(key1, (num_cells, num_genes)))
    s_data = jnp.abs(jax.random.normal(key2, (num_cells, num_genes)))

    return {
        "u_obs": u_data,
        "s_obs": s_data,
        "num_cells": num_cells,
        "num_genes": num_genes,
    }


@pytest.fixture
def training_state(jax_key, model_parameters):
    """Fixture for training state."""
    from pyrovelocity.models.jax.core.state import TrainingState

    return TrainingState(
        step=0,
        params=model_parameters,
        opt_state={},
        key=jax_key,
    )


@pytest.fixture
def inference_state(model_parameters):
    """Fixture for inference state."""
    from pyrovelocity.models.jax.core.state import InferenceState

    posterior_samples = {
        "alpha": jnp.stack([model_parameters["alpha"] for _ in range(10)]),
        "beta": jnp.stack([model_parameters["beta"] for _ in range(10)]),
        "gamma": jnp.stack([model_parameters["gamma"] for _ in range(10)]),
    }

    return InferenceState(posterior_samples=posterior_samples)


@pytest.fixture
def model_config():
    """Fixture for model configuration."""
    from pyrovelocity.models.jax.core.state import ModelConfig

    return ModelConfig()


@pytest.fixture
def inference_config():
    """Fixture for inference configuration."""
    from pyrovelocity.models.jax.core.state import InferenceConfig

    return InferenceConfig()
