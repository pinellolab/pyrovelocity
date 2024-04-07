import logging
from typing import Tuple

import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Callable
from diffrax import ODETerm
from diffrax import SaveAt
from diffrax import Solution
from diffrax import Tsit5
from diffrax import diffeqsolve
from jaxtyping import ArrayLike
from jaxtyping import Float
from jaxtyping import PyTree
from jaxtyping import jaxtyped

from pyrovelocity.logging import configure_logging


__all__ = ["solve_transcription_splicing_model"]

logging.getLogger("jax").setLevel(logging.ERROR)
logger = configure_logging(__name__)


TimeArrayType = Float[ArrayLike, "num_times"]
StateType = Float[ArrayLike, "num_vars"]
ParamsType = Tuple[float, ...]
ModelType = Callable[
    [
        PyTree[ArrayLike, "num_times"],
        PyTree[ArrayLike, "num_vars"],
        Tuple[float, ...],
    ],
    PyTree[ArrayLike, "num_vars"],
]


@jaxtyped(typechecker=beartype)
def dstate_dt_dimless(
    t: PyTree[ArrayLike, "num_times"],
    state: PyTree[ArrayLike, "num_vars"],
    params: PyTree[ArrayLike, "num_params"],
) -> PyTree[ArrayLike, "num_vars"]:
    """
    Simulate the transcription-splicing model.

    Args:
        t (float): time
        state (StateType): state vector
        params (ParamsType): scalar parameter

    Returns:
        PyTree[ArrayLike, "num_vars"]:
            time derivative of the state vector like StateType
    """
    u_star, s_star = state
    (gamma_star,) = params
    du_star_dt = 1 - u_star
    ds_star_dt = u_star - gamma_star * s_star
    return jnp.stack([du_star_dt, ds_star_dt])


@jaxtyped(typechecker=beartype)
def dstate_dt_dimensioned(
    t: PyTree[ArrayLike, "num_times"],
    state: PyTree[ArrayLike, "num_vars"],
    params: PyTree[ArrayLike, "num_params"],
) -> PyTree[ArrayLike, "num_vars"]:
    u, s = state
    alpha, beta, gamma = params
    du_dt = alpha - beta * u
    ds_dt = beta * u - gamma * s
    return jnp.stack([du_dt, ds_dt])


@jaxtyped(typechecker=beartype)
def solve_transcription_splicing_model_impl(
    ts: TimeArrayType,
    initial_state: StateType,
    params: ParamsType,
    time_step: float = 0.01,
    model: ModelType = dstate_dt_dimless,
) -> Solution:
    """
    Solve the transcription-splicing model.

    Args:
        ts (TimeArrayType): array of time points
        initial_state (StateType): initial state vector
        params (ParamsType): Tuple of scalar parameter(s).
        time_step (float): time step for the solver.
        model (ModelType): model function.

    Returns:
        Solution: diffrax Solution object.
    """
    term = ODETerm(model)
    t0 = ts[0]
    t1 = ts[-1]
    saveat = SaveAt(ts=ts)

    solution = diffeqsolve(
        term,
        solver=Tsit5(),
        t0=t0,
        t1=t1,
        dt0=time_step,
        y0=initial_state,
        args=params,
        saveat=saveat,
    )

    return solution


def solve_transcription_splicing_model(
    ts: TimeArrayType,
    initial_state: StateType,
    params: ParamsType,
    time_step: float = 0.01,
    model: ModelType = dstate_dt_dimless,
) -> Solution:
    """
    Solve the transcription-splicing model.

    This is a mirror of the `solve_transcription_splicing_model_impl` function
    with type-checking disabled for JIT compilation. This function should be
    removed when JIT compilation is enabled for the type-checked function.

    Args:
        ts (TimeArrayType): array of time points
        initial_state (StateType): initial state vector
        params (ParamsType): Tuple of scalar parameter(s).
        time_step (float): time step for the solver.
        model (ModelType): model function.

    Returns:
        Solution: diffrax Solution object.
    """
    term = ODETerm(model)
    t0 = ts[0]
    t1 = ts[-1]
    saveat = SaveAt(ts=ts)

    solution = diffeqsolve(
        term,
        solver=Tsit5(),
        t0=t0,
        t1=t1,
        dt0=time_step,
        y0=initial_state,
        args=params,
        saveat=saveat,
    )

    return solution


# TODO: enable JIT compilation of the type-checked solve function.
solve_transcription_splicing_model = jax.jit(
    # solve_transcription_splicing_model_impl,
    solve_transcription_splicing_model,
    static_argnames=["model"],
)
