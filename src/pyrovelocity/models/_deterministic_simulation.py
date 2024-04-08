import logging
from typing import Tuple

import diffrax
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


__all__ = [
    "solve_transcription_splicing_model",
    "solve_transcription_splicing_model_analytical",
]

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

    Examples:
        >>> from pyrovelocity.models import solve_transcription_splicing_model
        >>> from jax import numpy as jnp
        >>> ts0 = jnp.linspace(0, 4, 40)
        >>> ts1 = jnp.linspace(4 + (10 - 4) / 20, 10, 20)
        >>> ts = jnp.concatenate([ts0, ts1])
        >>> initial_state = jnp.array([0.1, 0.1])
        >>> params = (0.99,)
        >>> solution = solve_transcription_splicing_model(
        ...     ts,
        ...     initial_state,
        ...     params,
        >>> )
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


@jaxtyped(typechecker=beartype)
def calculate_xi(
    initial_u_star: Float[ArrayLike, ""],
    gamma_star: Float[ArrayLike, ""],
):
    """
    Calculate the Xi parameter for the analytical solution.

    Args:
        initial_u_star (float): The initial value of u_star.
        gamma_star (float): The gamma_star parameter.

    Returns:
        A float representing the value of Xi.
    """
    return (initial_u_star - 1) / (gamma_star - 1)


@jaxtyped(typechecker=beartype)
def analytical_solution_dstate_dt_dimless(
    ts: TimeArrayType,
    initial_state: StateType,
    params: ParamsType,
) -> PyTree[ArrayLike, "num_vars"]:
    """
    Compute the analytical solution for the dimensionless transcription-splicing
    model.

    Args:
        ts (TimeArrayType): Array of time points. initial_state (StateType):
        Initial state vector [u_star_0, s_star_0]. params (ParamsType): Tuple
        containing a single parameter (gamma_star).

    Returns:
        PyTree[ArrayLike, "num_vars"]:
            A PyTree containing the solution arrays for u_star and s_star.
    """
    (gamma_star,) = params
    u_star_0, s_star_0 = initial_state
    t_star_0 = ts[0]
    elapsed_ts = ts - t_star_0

    u_star = 1 + (u_star_0 - 1) * jnp.exp(-elapsed_ts)

    if jnp.isclose(gamma_star, 1.0):
        s_star = (
            1
            + (s_star_0 - 1) * jnp.exp(-elapsed_ts)
            + (u_star_0 - 1) * elapsed_ts * jnp.exp(-elapsed_ts)
        )
    else:
        xi = calculate_xi(u_star_0, gamma_star)

        s_star = (1 / gamma_star) + (
            (s_star_0 - xi - (1 / gamma_star))
            * jnp.exp(-gamma_star * elapsed_ts)
            + xi * jnp.exp(-elapsed_ts)
        )

    return jnp.stack([u_star, s_star]).T


@jaxtyped(typechecker=beartype)
def solve_transcription_splicing_model_analytical(
    ts: TimeArrayType,
    initial_state: StateType,
    params: ParamsType,
) -> Solution:
    """
    Computes the solutions using the analytical approach and wraps the result in
    a Diffrax Solution object.

    Args:
        ts: The time points at which to evaluate the solution.
        initial_state: The initial conditions for the system.
        params: The parameters of the model.

    Returns:
        Solution: A Diffrax Solution object containing the analytical solutions.

    Examples:
        >>> from pyrovelocity.models import solve_transcription_splicing_model_analytical
        >>> from jax import numpy as jnp
        >>> ts0 = jnp.linspace(0, 4, 40)
        >>> ts1 = jnp.linspace(4 + (10 - 4) / 20, 10, 20)
        >>> ts = jnp.concatenate([ts0, ts1])
        >>> initial_state = jnp.array([0.1, 0.1])
        >>> params = (0.99,)
        >>> analytical_solution = solve_transcription_splicing_model_analytical(
        ...     ts,
        ...     initial_state,
        ...     params,
        >>> )
    """

    ys = analytical_solution_dstate_dt_dimless(
        ts,
        initial_state,
        params,
    )

    solution = Solution(
        t0=ts[0],
        t1=ts[-1],
        ts=ts,
        ys=ys,
        interpolation=None,
        stats={},
        result=diffrax.RESULTS.successful,
        solver_state=None,
        controller_state=None,
        made_jump=None,
    )

    return solution
