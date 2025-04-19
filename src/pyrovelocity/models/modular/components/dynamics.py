"""Dynamics models for RNA velocity simulation.

This module implements various dynamics models that simulate RNA velocity
by modeling the time evolution of unspliced and spliced mRNA counts.
"""

from typing import Any, ClassVar, Dict, Optional, Tuple, Type

import jax
import jax.numpy as jnp
from beartype import beartype
from diffrax import ODETerm, SaveAt, Solution, Tsit5, diffeqsolve
from jaxtyping import Array, Float, jaxtyped

from pyrovelocity.models.modular.components.base import BaseDynamicsModel
from pyrovelocity.models.modular.interfaces import BatchTensor, ParamTensor
from pyrovelocity.models.modular.registry import DynamicsModelRegistry


@DynamicsModelRegistry.register("standard")
class StandardDynamicsModel(BaseDynamicsModel):
    """Standard dynamics model for RNA velocity.

    This model implements the standard model of RNA velocity:

    du/dt = alpha - beta * u
    ds/dt = beta * u - gamma * s

    where:
    - u is the unspliced mRNA count
    - s is the spliced mRNA count
    - alpha is the transcription rate
    - beta is the splicing rate
    - gamma is the degradation rate

    Attributes:
        name: The name of the model
        description: A brief description of the model
    """

    name: ClassVar[str] = "standard"
    description: ClassVar[str] = "Standard RNA velocity dynamics model"

    @jaxtyped
    @beartype
    def _forward_impl(
        self,
        u: BatchTensor,
        s: BatchTensor,
        alpha: ParamTensor,
        beta: ParamTensor,
        gamma: ParamTensor,
        scaling: Optional[ParamTensor] = None,
        t: Optional[BatchTensor] = None,
    ) -> Tuple[BatchTensor, BatchTensor]:
        """
        Implementation of the forward method for the standard dynamics model.

        This method computes the expected unspliced and spliced RNA counts
        based on the standard dynamics model.

        Args:
            u: Observed unspliced RNA counts
            s: Observed spliced RNA counts
            alpha: Transcription rate
            beta: Splicing rate
            gamma: Degradation rate
            scaling: Optional scaling factor for the dynamics
            t: Optional time points for the dynamics

        Returns:
            Tuple of (expected unspliced counts, expected spliced counts)
        """
        # Compute steady state values
        u_ss = alpha / beta
        s_ss = alpha / gamma

        # Apply scaling if provided
        if scaling is not None:
            u_ss = u_ss * scaling
            s_ss = s_ss * scaling

        return u_ss, s_ss

    @jaxtyped
    @beartype
    def _predict_future_states_impl(
        self,
        current_state: Tuple[BatchTensor, BatchTensor],
        time_delta: BatchTensor,
        alpha: ParamTensor,
        beta: ParamTensor,
        gamma: ParamTensor,
        scaling: Optional[ParamTensor] = None,
    ) -> Tuple[BatchTensor, BatchTensor]:
        """
        Implementation of the predict_future_states method for the standard dynamics model.

        This method predicts future unspliced and spliced RNA counts based on
        the current state and the standard dynamics model.

        Args:
            current_state: Tuple of (current unspliced counts, current spliced counts)
            time_delta: Time difference for prediction
            alpha: Transcription rate
            beta: Splicing rate
            gamma: Degradation rate
            scaling: Optional scaling factor for the dynamics

        Returns:
            Tuple of (predicted unspliced counts, predicted spliced counts)
        """
        # Extract current state
        u_current, s_current = current_state

        # Compute steady state values
        u_ss = alpha / beta
        s_ss = alpha / gamma

        # Compute future state using analytical solution
        # u(t) = u_ss + (u_0 - u_ss) * exp(-beta * t)
        # s(t) = s_ss + (s_0 - s_ss) * exp(-gamma * t) +
        #        beta * (u_0 - u_ss) / (gamma - beta) * (exp(-beta * t) - exp(-gamma * t))

        # Handle the case where beta == gamma
        beta_eq_gamma = jnp.isclose(beta, gamma)
        beta_safe = jnp.where(beta_eq_gamma, beta + 1e-6, beta)

        # Compute future unspliced counts
        u_future = u_ss + (u_current - u_ss) * jnp.exp(-beta * time_delta)

        # Compute future spliced counts
        exp_neg_beta_t = jnp.exp(-beta * time_delta)
        exp_neg_gamma_t = jnp.exp(-gamma * time_delta)

        # Handle the case where beta == gamma
        s_future_beta_eq_gamma = (
            s_ss
            + (s_current - s_ss) * exp_neg_gamma_t
            + (u_current - u_ss) * beta * time_delta * exp_neg_gamma_t
        )

        s_future_beta_neq_gamma = (
            s_ss
            + (s_current - s_ss) * exp_neg_gamma_t
            + beta
            * (u_current - u_ss)
            / (gamma - beta_safe)
            * (exp_neg_beta_t - exp_neg_gamma_t)
        )

        s_future = jnp.where(
            beta_eq_gamma, s_future_beta_eq_gamma, s_future_beta_neq_gamma
        )

        # Apply scaling if provided
        if scaling is not None:
            u_future = u_future * scaling
            s_future = s_future * scaling

        return u_future, s_future

    @jaxtyped
    @beartype
    def simulate(
        self,
        u0: Array,
        s0: Array,
        alpha: Array,
        beta: Array,
        gamma: Array,
        scaling: Array,
        t_max: float,
        n_steps: int,
        **kwargs,
    ) -> Tuple[Array, Array, Array]:
        """Simulate the dynamics model forward in time.

        Args:
            u0: Initial unspliced mRNA counts [genes]
            s0: Initial spliced mRNA counts [genes]
            alpha: Transcription rates [genes]
            beta: Splicing rates [genes]
            gamma: Degradation rates [genes]
            scaling: Scaling factors [genes]
            t_max: Maximum simulation time
            n_steps: Number of simulation steps
            **kwargs: Additional model-specific parameters

        Returns:
            Tuple of (time_points, unspliced_counts, spliced_counts)
            where each array has shape [n_steps, genes]
        """

        # Define the ODE system
        def rhs(t, state, args):
            u, s = state
            dudt = alpha - beta * u
            dsdt = beta * u - gamma * s
            return jnp.stack([dudt, dsdt])

        # Set up time points
        times = jnp.linspace(0, t_max, n_steps)

        # Initial state
        y0 = jnp.stack([u0, s0])

        # Solve the ODE system
        term = ODETerm(rhs)
        saveat = SaveAt(ts=times)

        solution = diffeqsolve(
            term,
            solver=Tsit5(),
            t0=times[0],
            t1=times[-1],
            dt0=t_max / n_steps,
            y0=y0,
            args=None,
            saveat=saveat,
        )

        # Extract results
        u_t = solution.ys[:, 0, :]
        s_t = solution.ys[:, 1, :]

        # Apply scaling factor to match the observed data scale
        u_t = u_t * scaling
        s_t = s_t * scaling

        return times, u_t, s_t

    @jaxtyped
    @beartype
    def steady_state(
        self, alpha: Array, beta: Array, gamma: Array, **kwargs
    ) -> Tuple[Array, Array]:
        """Calculate the steady state values for unspliced and spliced mRNA.

        Args:
            alpha: Transcription rates [genes]
            beta: Splicing rates [genes]
            gamma: Degradation rates [genes]
            **kwargs: Additional model-specific parameters

        Returns:
            Tuple of (unspliced_steady_state, spliced_steady_state)
            each with shape [genes]
        """
        # At steady state, du/dt = 0 and ds/dt = 0
        # From du/dt = 0: alpha - beta * u = 0 => u = alpha / beta
        u_ss = alpha / beta

        # From ds/dt = 0: beta * u - gamma * s = 0 => s = beta * u / gamma
        # Substituting u = alpha / beta: s = beta * (alpha / beta) / gamma = alpha / gamma
        s_ss = alpha / gamma

        return u_ss, s_ss


@DynamicsModelRegistry.register("nonlinear")
class NonlinearDynamicsModel(BaseDynamicsModel):
    """Nonlinear dynamics model for RNA velocity with saturation effects.

    This model implements a nonlinear model of RNA velocity with saturation:

    du/dt = alpha / (1 + u/k_alpha) - beta * u / (k_beta + u)
    ds/dt = beta * u / (k_beta + u) - gamma * s

    where:
    - u is the unspliced mRNA count
    - s is the spliced mRNA count
    - alpha is the transcription rate
    - beta is the splicing rate
    - gamma is the degradation rate
    - k_alpha is the saturation constant for transcription
    - k_beta is the saturation constant for splicing

    Attributes:
        name: The name of the model
        description: A brief description of the model
    """

    name: ClassVar[str] = "nonlinear"
    description: ClassVar[
        str
    ] = "Nonlinear RNA velocity dynamics model with saturation effects"

    @jaxtyped
    @beartype
    def _forward_impl(
        self,
        u: BatchTensor,
        s: BatchTensor,
        alpha: ParamTensor,
        beta: ParamTensor,
        gamma: ParamTensor,
        scaling: Optional[ParamTensor] = None,
        t: Optional[BatchTensor] = None,
        k_alpha: Optional[ParamTensor] = None,
        k_beta: Optional[ParamTensor] = None,
    ) -> Tuple[BatchTensor, BatchTensor]:
        """
        Implementation of the forward method for the nonlinear dynamics model.

        This method computes the expected unspliced and spliced RNA counts
        based on the nonlinear dynamics model with saturation effects.

        Args:
            u: Observed unspliced RNA counts
            s: Observed spliced RNA counts
            alpha: Transcription rate
            beta: Splicing rate
            gamma: Degradation rate
            scaling: Optional scaling factor for the dynamics
            t: Optional time points for the dynamics
            k_alpha: Saturation constant for transcription, defaults to alpha
            k_beta: Saturation constant for splicing, defaults to beta

        Returns:
            Tuple of (expected unspliced counts, expected spliced counts)
        """
        # Set default values for saturation constants if not provided
        if k_alpha is None:
            k_alpha = alpha
        if k_beta is None:
            k_beta = beta

        # Check if k_alpha and k_beta are very large
        # If they are, use the standard model steady state
        if jnp.all(k_alpha > 1e5) and jnp.all(k_beta > 1e5):
            # Use standard model steady state
            u_ss = alpha / beta
            s_ss = alpha / gamma

            # Apply scaling if provided
            if scaling is not None:
                u_ss = u_ss * scaling
                s_ss = s_ss * scaling

            return u_ss, s_ss

        # Calculate steady state using the nonlinear model
        u_ss, s_ss = self.steady_state(alpha, beta, gamma, k_alpha, k_beta)

        # Apply scaling if provided
        if scaling is not None:
            u_ss = u_ss * scaling
            s_ss = s_ss * scaling

        return u_ss, s_ss

    @jaxtyped
    @beartype
    def _predict_future_states_impl(
        self,
        current_state: Tuple[BatchTensor, BatchTensor],
        time_delta: BatchTensor,
        alpha: ParamTensor,
        beta: ParamTensor,
        gamma: ParamTensor,
        scaling: Optional[ParamTensor] = None,
        k_alpha: Optional[ParamTensor] = None,
        k_beta: Optional[ParamTensor] = None,
    ) -> Tuple[BatchTensor, BatchTensor]:
        """
        Implementation of the predict_future_states method for the nonlinear dynamics model.

        This method predicts future unspliced and spliced RNA counts based on
        the current state and the nonlinear dynamics model with saturation effects.

        Args:
            current_state: Tuple of (current unspliced counts, current spliced counts)
            time_delta: Time difference for prediction
            alpha: Transcription rate
            beta: Splicing rate
            gamma: Degradation rate
            scaling: Optional scaling factor for the dynamics
            k_alpha: Saturation constant for transcription, defaults to alpha
            k_beta: Saturation constant for splicing, defaults to beta

        Returns:
            Tuple of (predicted unspliced counts, predicted spliced counts)
        """
        # Set default values for saturation constants if not provided
        if k_alpha is None:
            k_alpha = alpha
        if k_beta is None:
            k_beta = beta

        # Extract current state
        u_current, s_current = current_state

        # Check if k_alpha and k_beta are very large
        # If they are, use the standard model prediction
        if jnp.all(k_alpha > 1e5) and jnp.all(k_beta > 1e5):
            # Use standard model prediction
            standard_model = StandardDynamicsModel()
            return standard_model._predict_future_states_impl(
                current_state, time_delta, alpha, beta, gamma, scaling
            )

        # For the nonlinear model, we need to use numerical integration
        # since there's no simple analytical solution

        # Define the ODE system with saturation effects
        def rhs(t, state, args):
            u, s = state
            dudt = alpha / (1 + u / k_alpha) - beta * u / (k_beta + u)
            dsdt = beta * u / (k_beta + u) - gamma * s
            return jnp.stack([dudt, dsdt])

        # Set up time points for integration
        times = jnp.array([0.0, float(time_delta.item())])

        # Initial state
        y0 = jnp.stack([u_current, s_current])

        # Solve the ODE system
        term = ODETerm(rhs)
        saveat = SaveAt(ts=times)

        solution = diffeqsolve(
            term,
            solver=Tsit5(),
            t0=times[0],
            t1=times[-1],
            dt0=float(time_delta.item()) / 10,  # Use 10 steps for accuracy
            y0=y0,
            args=None,
            saveat=saveat,
        )

        # Extract results (final state)
        u_future = solution.ys[-1, 0, :]
        s_future = solution.ys[-1, 1, :]

        # Apply scaling if provided
        if scaling is not None:
            u_future = u_future * scaling
            s_future = s_future * scaling

        return u_future, s_future

    @jaxtyped
    @beartype
    def simulate(
        self,
        u0: Array,
        s0: Array,
        alpha: Array,
        beta: Array,
        gamma: Array,
        scaling: Array,
        t_max: float,
        n_steps: int,
        k_alpha: Optional[Array] = None,
        k_beta: Optional[Array] = None,
        **kwargs,
    ) -> Tuple[Array, Array, Array]:
        """Simulate the dynamics model forward in time.

        Args:
            u0: Initial unspliced mRNA counts [genes]
            s0: Initial spliced mRNA counts [genes]
            alpha: Transcription rates [genes]
            beta: Splicing rates [genes]
            gamma: Degradation rates [genes]
            scaling: Scaling factors [genes]
            t_max: Maximum simulation time
            n_steps: Number of simulation steps
            k_alpha: Saturation constant for transcription [genes], defaults to alpha
            k_beta: Saturation constant for splicing [genes], defaults to beta
            **kwargs: Additional model-specific parameters

        Returns:
            Tuple of (time_points, unspliced_counts, spliced_counts)
            where each array has shape [n_steps, genes]
        """
        # Set default values for saturation constants if not provided
        if k_alpha is None:
            k_alpha = alpha
        if k_beta is None:
            k_beta = beta

        # Check if k_alpha and k_beta are very large
        # If they are, use the standard model simulation
        if jnp.all(k_alpha > 1e3) and jnp.all(k_beta > 1e3):
            # Use standard model simulation
            standard_model = StandardDynamicsModel()
            return standard_model.simulate(
                u0, s0, alpha, beta, gamma, scaling, t_max, n_steps, **kwargs
            )

        # Define the ODE system with saturation effects
        def rhs(t, state, args):
            u, s = state
            dudt = alpha / (1 + u / k_alpha) - beta * u / (k_beta + u)
            dsdt = beta * u / (k_beta + u) - gamma * s
            return jnp.stack([dudt, dsdt])

        # Set up time points
        times = jnp.linspace(0, t_max, n_steps)

        # Initial state
        y0 = jnp.stack([u0, s0])

        # Solve the ODE system
        term = ODETerm(rhs)
        saveat = SaveAt(ts=times)

        solution = diffeqsolve(
            term,
            solver=Tsit5(),
            t0=times[0],
            t1=times[-1],
            dt0=t_max / n_steps,
            y0=y0,
            args=None,
            saveat=saveat,
        )

        # Extract results
        u_t = solution.ys[:, 0, :]
        s_t = solution.ys[:, 1, :]

        # Apply scaling factor to match the observed data scale
        u_t = u_t * scaling
        s_t = s_t * scaling

        return times, u_t, s_t

    @jaxtyped
    @beartype
    def steady_state(
        self,
        alpha: Array,
        beta: Array,
        gamma: Array,
        k_alpha: Optional[Array] = None,
        k_beta: Optional[Array] = None,
        **kwargs,
    ) -> Tuple[Array, Array]:
        """Calculate the steady state values for unspliced and spliced mRNA.

        Args:
            alpha: Transcription rates [genes]
            beta: Splicing rates [genes]
            gamma: Degradation rates [genes]
            k_alpha: Saturation constant for transcription [genes], defaults to alpha
            k_beta: Saturation constant for splicing [genes], defaults to beta
            **kwargs: Additional model-specific parameters

        Returns:
            Tuple of (unspliced_steady_state, spliced_steady_state)
            each with shape [genes]
        """
        # Set default values for saturation constants if not provided
        if k_alpha is None:
            k_alpha = alpha
        if k_beta is None:
            k_beta = beta

        # Check if k_alpha and k_beta are very large
        # If they are, use the standard model steady state
        # This is a special case where the nonlinear model approaches the standard model
        if jnp.all(k_alpha > 1e5) and jnp.all(k_beta > 1e5):
            # Use standard model steady state
            u_ss = alpha / beta
            s_ss = alpha / gamma
            return u_ss, s_ss

        # At steady state, du/dt = 0:
        # alpha / (1 + u/k_alpha) - beta * u / (k_beta + u) = 0
        # This is a nonlinear equation that requires numerical solution
        # For simplicity, we'll use a fixed-point iteration approach

        # Initial guess: use the standard model steady state
        u_ss = alpha / beta

        # Fixed-point iteration (simplified approach)
        for _ in range(20):  # More iterations for better convergence
            u_ss = alpha * k_alpha / (k_alpha + u_ss) * (k_beta + u_ss) / beta

        # At steady state, ds/dt = 0:
        # beta * u / (k_beta + u) - gamma * s = 0
        # => s = beta * u / (gamma * (k_beta + u))
        s_ss = beta * u_ss / (gamma * (k_beta + u_ss))

        return u_ss, s_ss
