import arviz as az
import jax
import jax.numpy as jnp
import pytest
from numpyro.infer import MCMC
from numpyro.infer import NUTS
from numpyro.infer import Predictive

from pyrovelocity.logging import configure_logging
from pyrovelocity.models import (
    deterministic_transcription_splicing_probabilistic_model,
)
from pyrovelocity.models._deterministic_inference import (
    generate_prior_predictive_samples,
)
from pyrovelocity.models._deterministic_inference import (
    generate_test_data_for_deterministic_model_inference,
)


logger = configure_logging(__name__)


@pytest.fixture
def setup_data():
    return generate_test_data_for_deterministic_model_inference(
        num_genes=1,
        num_cells=3,
        num_timepoints=4,
        num_modalities=2,
    )


def test_priors(setup_data):
    (
        times,
        data,
        num_genes,
        num_cells,
        num_timepoints,
        num_modalities,
    ) = setup_data
    rng_key = jax.random.PRNGKey(0)

    def model():
        deterministic_transcription_splicing_probabilistic_model(times, data)

    predictive = Predictive(model, num_samples=500)
    prior_samples = predictive(rng_key)

    assert "initial_conditions" in prior_samples
    assert "gamma" in prior_samples
    assert "sigma" in prior_samples

    u0_samples, s0_samples = (
        prior_samples["initial_conditions"][:, :, 0],
        prior_samples["initial_conditions"][:, :, 1],
    )
    assert u0_samples.shape == (500, num_cells)
    assert s0_samples.shape == (500, num_cells)
    assert jnp.all(
        u0_samples > 0
    ), "All initial u0 values should be positive (LogNormal)"
    assert jnp.all(
        s0_samples > 0
    ), "All initial s0 values should be positive (LogNormal)"

    gamma_samples = prior_samples["gamma"]
    assert gamma_samples.shape == (500, num_cells)
    assert jnp.all(
        gamma_samples > 0
    ), "All gamma values should be positive (LogNormal)"

    sigma_samples = prior_samples["sigma"]
    assert sigma_samples.shape == (500, num_modalities)
    assert jnp.all(
        sigma_samples > 0
    ), "All sigma values should be positive (HalfNormal)"


def test_ode_solution(setup_data):
    (
        times,
        data,
        num_genes,
        num_cells,
        num_timepoints,
        num_modalities,
    ) = setup_data
    rng_key = jax.random.PRNGKey(0)

    def model():
        deterministic_transcription_splicing_probabilistic_model(times, data)

    predictive = Predictive(model, num_samples=10)
    samples = predictive(rng_key)

    observations = samples["observations"]
    assert observations.shape == (
        10,
        num_genes,
        num_cells,
        num_timepoints,
        num_modalities,
    )
    assert jnp.all(
        jnp.isfinite(observations)
    ), "Observations contain non-finite values"

    u_pred = observations[..., 0]
    s_pred = observations[..., 1]
    assert u_pred.shape == (10, num_genes, num_cells, num_timepoints)
    assert s_pred.shape == (10, num_genes, num_cells, num_timepoints)
    assert jnp.all(
        jnp.isfinite(u_pred)
    ), "u predictions contain non-finite values"
    assert jnp.all(
        jnp.isfinite(s_pred)
    ), "s predictions contain non-finite values"


def test_model_sampling_statements_prior_predictive(setup_data):
    times, data, num_cells, num_genes, _, _ = setup_data
    rng_key = jax.random.PRNGKey(0)
    rng_key, rng_key_ = jax.random.split(rng_key)

    prior_predictive = Predictive(
        deterministic_transcription_splicing_probabilistic_model,
        num_samples=7,
    )
    prior_predictions = prior_predictive(rng_key_, times=times, data=data)
    prior_predictions_summary = [
        f"Key: {k} => Type: {type(v).__name__}, Shape: {getattr(v, 'shape', 'No shape attribute')}"
        for k, v in prior_predictions.items()
    ]

    logger.info(
        f"\nSummary of prior predictive samples:\n\n"
        + "\n".join(prior_predictions_summary)
    )

    assert all(
        f"u_obs_{i}_{j}" in prior_predictions
        for i in range(num_genes)
        for j in range(num_cells)
    )
    assert all(
        f"s_obs_{i}_{j}" in prior_predictions
        for i in range(num_genes)
        for j in range(num_cells)
    )


def test_model_sampling_statements_posterior_predictive(setup_data):
    times, data, num_cells, num_genes, _, _ = setup_data
    rng_key = jax.random.PRNGKey(0)
    rng_key, rng_key_ = jax.random.split(rng_key)

    nuts_kernel = NUTS(deterministic_transcription_splicing_probabilistic_model)
    mcmc = MCMC(nuts_kernel, num_warmup=2, num_samples=6, num_chains=1)
    mcmc.run(rng_key_, times=times, data=data)

    samples = mcmc.get_samples()
    posterior_predictive = Predictive(
        deterministic_transcription_splicing_probabilistic_model,
        samples,
    )
    posterior_predictions = posterior_predictive(
        rng_key_,
        data=data,
        times=times,
    )

    posterior_predictions_summary = [
        f"Key: {k} => Type: {type(v).__name__}, Shape: {getattr(v, 'shape', 'No shape attribute')}"
        for k, v in posterior_predictions.items()
    ]
    logger.info(
        f"\nSummary of posterior predictive samples:\n\n"
        + "\n".join(posterior_predictions_summary)
    )

    assert all(
        f"u_obs_{i}_{j}" in posterior_predictions
        for i in range(num_genes)
        for j in range(num_cells)
    )
    assert all(
        f"s_obs_{i}_{j}" in posterior_predictions
        for i in range(num_genes)
        for j in range(num_cells)
    )

