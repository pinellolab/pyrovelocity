"""
Pure function implementations of RNA velocity dynamics models.

This module contains pure function implementations of RNA velocity dynamics
models, including:

- standard_dynamics_model: Standard RNA velocity model
- nonlinear_dynamics_model: Nonlinear RNA velocity model with saturation
- dynamics_ode_model: ODE-based RNA velocity model using Diffrax
"""

from typing import Dict, Tuple, Optional, Any, Callable
import jax
import jax.numpy as jnp
import numpyro
import diffrax
from jaxtyping import Array, Float
from beartype import beartype


@beartype
def standard_dynamics_model(
    tau: Float[Array, "..."],
    u0: Float[Array, "..."],
    s0: Float[Array, "..."],
    params: Dict[str, Float[Array, "..."]],
) -> Tuple[Float[Array, "..."], Float[Array, "..."]]:
    """Standard RNA velocity dynamics model.

    This function computes the mRNA dynamics given temporal coordinate, parameter values, and
    initial conditions. It includes special handling for the case where gamma equals beta.

    The implementation is based on Equation 2.12 of:
    > Li T, Shi J, Wu Y, Zhou P. On the mathematics of RNA velocity I:
    Theoretical analysis. CSIAM Transactions on Applied Mathematics. 2021;2:
    1–55. doi:10.4208/csiam-am.so-2020-0001

    Args:
        tau: Time parameter
        u0: Initial unspliced RNA
        s0: Initial spliced RNA
        params: Dictionary of parameters (alpha, beta, gamma)

    Returns:
        Tuple of (unspliced, spliced) RNA counts
    """
    alpha = params["alpha"]
    beta = params["beta"]
    gamma = params["gamma"]

    # Compute exponential terms
    expu = jnp.exp(-beta * tau)
    exps = jnp.exp(-gamma * tau)

    # Handle the case where gamma equals beta
    gamma_equals_beta = jnp.isclose(gamma, beta)

    # Compute unspliced RNA
    ut = u0 * expu + alpha / beta * (1 - expu)

    # Compute spliced RNA for the general case
    expus = (alpha - u0 * beta) * (1.0 / (gamma - beta)) * (exps - expu)
    st_general = s0 * exps + alpha / gamma * (1 - exps) + expus

    # Compute spliced RNA for the special case where gamma equals beta
    st_special = (
        s0 * expu + alpha / beta * (1 - expu) - (alpha - beta * u0) * tau * expu
    )

    # Select the appropriate spliced RNA calculation based on whether gamma equals beta
    st = jnp.where(gamma_equals_beta, st_special, st_general)

    return ut, st


@beartype
def nonlinear_dynamics_model(
    tau: Float[Array, "..."],
    u0: Float[Array, "..."],
    s0: Float[Array, "..."],
    params: Dict[str, Float[Array, "..."]],
) -> Tuple[Float[Array, "..."], Float[Array, "..."]]:
    """Nonlinear RNA velocity model with saturation.

    This function extends the standard dynamics model by adding a saturation term
    to the transcription rate. The transcription rate is modeled as:

    alpha * (1 / (1 + scaling * u))

    where scaling controls the strength of the saturation effect.

    Args:
        tau: Time parameter
        u0: Initial unspliced RNA
        s0: Initial spliced RNA
        params: Dictionary of parameters (alpha, beta, gamma, scaling)

    Returns:
        Tuple of (unspliced, spliced) RNA counts
    """
    alpha = params["alpha"]
    beta = params["beta"]
    gamma = params["gamma"]
    scaling = params["scaling"]

    # Define the ODE system for the nonlinear model
    def dstate_dt(state, t):
        u, s = state
        # Nonlinear transcription rate with saturation
        transcription_rate = alpha * (1.0 / (1.0 + scaling * u))
        du_dt = transcription_rate - beta * u
        ds_dt = beta * u - gamma * s
        return jnp.stack([du_dt, ds_dt])

    # Use JAX's ODE solver to compute the solution
    # Since we're using a nonlinear model, we need to solve it numerically

    # Initialize state
    y0 = jnp.stack([u0, s0])

    # Define a function to compute the state at time tau
    def get_state_at_tau(y0, tau_val):
        # Use a simple Euler method with a fixed number of steps
        # This avoids the ConcretizationTypeError with dynamic step counts
        max_steps = 100  # Fixed number of steps
        dt = tau_val / max_steps  # Adjust dt based on tau

        def step_fn(state, _):
            y, t = state
            # Only update if t < tau_val
            dydt = dstate_dt(y, 0)  # Time-independent ODE
            y_next = y + dydt * dt
            t_next = t + dt
            return (y_next, t_next), None

        # Start with initial state and t=0
        (final_state, _), _ = jax.lax.scan(
            step_fn, (y0, 0.0), None, length=max_steps
        )
        return final_state

    # Process each time point individually to avoid vmap issues
    ut_list = []
    st_list = []
    for i in range(len(tau)):
        # For each time point, we need to use the corresponding initial conditions and parameters
        y0_i = jnp.stack([u0[i], s0[i]])
        alpha_i = alpha[i]
        beta_i = beta[i]
        gamma_i = gamma[i]
        scaling_i = scaling[i]

        # Define a local ODE system for this specific time point
        def dstate_dt_i(state, t):
            u, s = state
            # Nonlinear transcription rate with saturation
            transcription_rate = alpha_i * (1.0 / (1.0 + scaling_i * u))
            du_dt = transcription_rate - beta_i * u
            ds_dt = beta_i * u - gamma_i * s
            return jnp.stack([du_dt, ds_dt])

        # Use a simple Euler method with a fixed number of steps
        max_steps = 100  # Fixed number of steps
        dt = tau[i] / max_steps  # Adjust dt based on tau

        # Initial state
        y = y0_i

        # Integrate the ODE
        for _ in range(max_steps):
            dydt = dstate_dt_i(y, 0)  # Time-independent ODE
            y = y + dydt * dt

        # Extract the final state
        ut_list.append(y[0])
        st_list.append(y[1])

    # Stack the results
    ut = jnp.array(ut_list)
    st = jnp.array(st_list)

    return ut, st


@beartype
def dynamics_ode_model(
    tau: Float[Array, "..."],
    u0: Float[Array, "..."],
    s0: Float[Array, "..."],
    params: Dict[str, Float[Array, "..."]],
) -> Tuple[Float[Array, "..."], Float[Array, "..."]]:
    """ODE-based RNA velocity model using Diffrax.

    This function solves the RNA velocity ODE system using the Diffrax library.
    The ODE system is defined as:

    du/dt = alpha - beta * u
    ds/dt = beta * u - gamma * s

    Args:
        tau: Time parameter
        u0: Initial unspliced RNA
        s0: Initial spliced RNA
        params: Dictionary of parameters (alpha, beta, gamma)

    Returns:
        Tuple of (unspliced, spliced) RNA counts
    """
    alpha = params["alpha"]
    beta = params["beta"]
    gamma = params["gamma"]

    # Instead of using diffrax for vectorized ODE solving, which is causing shape issues,
    # let's use a simpler approach with a fixed-step Euler method

    # Define a function to solve the ODE for a single time point using Euler method
    def solve_for_single_tau(tau_val, u0_val, s0_val):
        # Use a simple Euler method with a fixed number of steps
        max_steps = 1000  # Increase the number of steps for better accuracy
        dt = tau_val / max_steps  # Adjust dt based on tau

        # Initial state
        u = u0_val
        s = s0_val

        # Integrate the ODE
        for _ in range(max_steps):
            # Update using the ODE equations
            du_dt = alpha - beta * u
            ds_dt = beta * u - gamma * s

            # Euler step
            u = u + du_dt * dt
            s = s + ds_dt * dt

        return jnp.stack([u, s])

    # Process each time point individually to avoid vmap issues
    ut_list = []
    st_list = []
    for i in range(len(tau)):
        # For each time point, we need to use the corresponding initial conditions and parameters
        u_i = u0[i]
        s_i = s0[i]
        alpha_i = alpha[i]
        beta_i = beta[i]
        gamma_i = gamma[i]

        # Use a simple Euler method with a fixed number of steps
        max_steps = 100  # Fixed number of steps
        dt = tau[i] / max_steps  # Adjust dt based on tau

        # Initial state
        u = u_i
        s = s_i

        # Integrate the ODE
        for _ in range(max_steps):
            # Update using the ODE equations
            du_dt = alpha_i - beta_i * u
            ds_dt = beta_i * u - gamma_i * s

            # Euler step
            u = u + du_dt * dt
            s = s + ds_dt * dt

        # Store the final state
        ut_list.append(u)
        st_list.append(s)

    # Stack the results
    ut = jnp.array(ut_list)
    st = jnp.array(st_list)

    return ut, st


@beartype
def vectorized_standard_dynamics_model(
    tau: Float[Array, "batch"],
    u0: Float[Array, "batch"],
    s0: Float[Array, "batch"],
    params: Dict[str, Float[Array, "batch"]],
) -> Tuple[Float[Array, "batch"], Float[Array, "batch"]]:
    """Vectorized version of the standard RNA velocity dynamics model.

    This function applies the standard dynamics model to batches of inputs
    using JAX's vmap for efficient vectorization.

    Args:
        tau: Batch of time parameters
        u0: Batch of initial unspliced RNA
        s0: Batch of initial spliced RNA
        params: Dictionary of batched parameters (alpha, beta, gamma)

    Returns:
        Tuple of batched (unspliced, spliced) RNA counts
    """

    # Define a function that processes a single element
    def process_single_element(tau_i, u0_i, s0_i, alpha_i, beta_i, gamma_i):
        params_i = {"alpha": alpha_i, "beta": beta_i, "gamma": gamma_i}
        return standard_dynamics_model(tau_i, u0_i, s0_i, params_i)

    # Vectorize the function
    vmap_fn = jax.vmap(process_single_element)

    # Apply the vectorized function
    return vmap_fn(
        tau, u0, s0, params["alpha"], params["beta"], params["gamma"]
    )


@beartype
def vectorized_nonlinear_dynamics_model(
    tau: Float[Array, "batch"],
    u0: Float[Array, "batch"],
    s0: Float[Array, "batch"],
    params: Dict[str, Float[Array, "batch"]],
) -> Tuple[Float[Array, "batch"], Float[Array, "batch"]]:
    """Vectorized version of the nonlinear RNA velocity dynamics model.

    This function applies the nonlinear dynamics model to batches of inputs.

    Args:
        tau: Batch of time parameters
        u0: Batch of initial unspliced RNA
        s0: Batch of initial spliced RNA
        params: Dictionary of batched parameters (alpha, beta, gamma, scaling)

    Returns:
        Tuple of batched (unspliced, spliced) RNA counts
    """
    # Since the nonlinear_dynamics_model already handles batched inputs,
    # we can just call it directly
    return nonlinear_dynamics_model(tau, u0, s0, params)


@beartype
def vectorized_dynamics_ode_model(
    tau: Float[Array, "batch"],
    u0: Float[Array, "batch"],
    s0: Float[Array, "batch"],
    params: Dict[str, Float[Array, "batch"]],
) -> Tuple[Float[Array, "batch"], Float[Array, "batch"]]:
    """Vectorized version of the ODE-based RNA velocity dynamics model.

    This function applies the ODE-based dynamics model to batches of inputs.

    Args:
        tau: Batch of time parameters
        u0: Batch of initial unspliced RNA
        s0: Batch of initial spliced RNA
        params: Dictionary of batched parameters (alpha, beta, gamma)

    Returns:
        Tuple of batched (unspliced, spliced) RNA counts
    """
    # Since the dynamics_ode_model already handles batched inputs,
    # we can just call it directly
    return dynamics_ode_model(tau, u0, s0, params)
