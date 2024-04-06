import diffrax
from diffrax import Solution
from jax import numpy as jnp
from jax.config import config

from pyrovelocity.models._deterministic_simulation import dstate_dt_dimensioned
from pyrovelocity.models._deterministic_simulation import dstate_dt_dimless
from pyrovelocity.models._deterministic_simulation import (
    solve_transcription_splicing_model,
)
from pyrovelocity.models._deterministic_simulation import (
    solve_transcription_splicing_model_impl,
)


config.update("jax_debug_nans", True)


def test_dstate_dt_dimless_basic():
    """Test the basic functionality of dstate_dt_dimless."""
    t = jnp.array([0.0])
    state = jnp.array([1.0, 0.5])
    params = (0.5,)
    expected_output = jnp.array([0, 0.75])
    output = dstate_dt_dimless(t, state, params)
    assert jnp.allclose(
        output, expected_output
    ), f"Expected {expected_output}, got {output}"


def test_dstate_dt_dimensioned_basic():
    """Test the basic functionality of dstate_dt_dimensioned."""
    t = jnp.array([0.0])
    state = jnp.array([1.0, 0.5])
    params = (1.0, 0.5, 0.2)
    expected_output = jnp.array([0.5, 0.4])
    output = dstate_dt_dimensioned(t, state, params)
    assert jnp.allclose(
        output, expected_output
    ), f"Expected {expected_output}, got {output}"


def test_solve_transcription_splicing_model_basic():
    """Test solving the model with a simple set of inputs."""
    ts = jnp.linspace(0, 10, 100)
    initial_state = jnp.array([0.5, 0.5])
    params = (0.5,)

    solution = solve_transcription_splicing_model_impl(
        ts, initial_state, params
    )

    assert isinstance(
        solution, Solution
    ), "Returned object should be a Solution instance"

    assert solution.ts.shape == (
        100,
    ), f"Expected ts shape to be (100,), got {solution.ts.shape}"
    assert solution.ys.shape == (
        100,
        2,
    ), f"Expected ys shape to be (100, 2), got {solution.ys.shape}"

    assert (
        solution.result == diffrax.RESULTS.successful
    ), f"Solver did not succeed; result was {solution.result}"


def test_solve_transcription_splicing_model_dimensioned():
    """Test solving the dimensioned model with a simple set of inputs."""
    ts = jnp.linspace(0, 10, 100)
    initial_state = jnp.array([1.0, 0.5])
    params = (1.0, 0.5, 0.2)

    solution = solve_transcription_splicing_model_impl(
        ts, initial_state, params, model=dstate_dt_dimensioned
    )

    assert isinstance(
        solution, Solution
    ), "Returned object should be a Solution instance"
    assert solution.ts.shape == (
        100,
    ), "Expected ts shape to match input time shape"
    assert solution.ys.shape == (
        100,
        2,
    ), "Expected ys shape to match input state shape"
    assert (
        solution.result == diffrax.RESULTS.successful
    ), "Solver did not succeed; check the result attribute"


def test_solve_transcription_splicing_model_impl_jit_equivalent():
    ts = jnp.linspace(0, 10, 100)
    initial_state = jnp.array([0.5, 0.5])
    params = (0.5,)

    solution_impl = solve_transcription_splicing_model_impl(
        ts,
        initial_state,
        params,
    )
    solution = solve_transcription_splicing_model(
        ts,
        initial_state,
        params,
    )
    assert jnp.allclose(
        solution_impl.ts, solution.ts
    ), "ts values should be equal"
    assert jnp.allclose(
        solution_impl.ys, solution.ys
    ), "ys values should be equal"


def test_solve_transcription_splicing_model_impl_dimensioned_jit_equivalent():
    ts = jnp.linspace(0, 10, 100)
    initial_state = jnp.array([1.0, 0.5])
    params = (1.0, 0.5, 0.2)

    solution_impl = solve_transcription_splicing_model_impl(
        ts,
        initial_state,
        params,
        model=dstate_dt_dimensioned,
    )
    solution = solve_transcription_splicing_model(
        ts,
        initial_state,
        params,
        model=dstate_dt_dimensioned,
    )
    assert jnp.allclose(
        solution_impl.ts, solution.ts
    ), "ts values should be equal"
    assert jnp.allclose(
        solution_impl.ys, solution.ys
    ), "ys values should be equal"
