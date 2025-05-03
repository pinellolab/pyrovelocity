"""
Protocol-First dynamics model implementations for PyroVelocity's modular architecture.

This module contains dynamics model implementations that directly implement the
DynamicsModel Protocol without inheriting from BaseDynamicsModel. These implementations
follow the Protocol-First approach, which embraces composition over inheritance
and allows for more flexible component composition.
"""

from typing import Any, ClassVar, Dict, Optional, Tuple, Union

import torch
from beartype import beartype
from jaxtyping import jaxtyped

from pyrovelocity.models.modular.interfaces import (
    BatchTensor,
    DynamicsModel,
    ParamTensor,
)
from pyrovelocity.models.modular.registry import DynamicsModelRegistry
from pyrovelocity.models.modular.utils.context_utils import validate_context


@DynamicsModelRegistry.register("standard_direct")
class StandardDynamicsModelDirect:
    """Standard dynamics model for RNA velocity using analytical solution.

    This model implements the standard model of RNA velocity:

    du/dt = alpha - beta * u
    ds/dt = beta * u - gamma * s

    where:
    - u is the unspliced mRNA count
    - s is the spliced mRNA count
    - alpha is the transcription rate
    - beta is the splicing rate
    - gamma is the degradation rate

    This implementation uses the analytical solution and directly implements
    the DynamicsModel Protocol without inheriting from BaseDynamicsModel.

    Attributes:
        name: The name of the model
        description: A brief description of the model
        shared_time: Whether to use shared time across cells
        t_scale_on: Whether to use time scaling
        cell_specific_kinetics: Type of cell-specific kinetics
        kinetics_num: Number of kinetics
    """

    name: ClassVar[str] = "standard_direct"
    description: ClassVar[str] = "Standard RNA velocity dynamics model (Protocol-First implementation)"

    def __init__(
        self,
        name: str = "dynamics_model_direct",
        shared_time: bool = True,
        t_scale_on: bool = False,
        cell_specific_kinetics: Optional[str] = None,
        kinetics_num: Optional[int] = None,
        **kwargs,
    ):
        """
        Initialize the dynamics model.

        Args:
            name: A unique name for this component instance.
            shared_time: Whether to use shared time across cells.
            t_scale_on: Whether to use time scaling.
            cell_specific_kinetics: Type of cell-specific kinetics.
            kinetics_num: Number of kinetics.
            **kwargs: Additional keyword arguments.
        """
        self.name = name
        self.shared_time = shared_time
        self.t_scale_on = t_scale_on
        self.cell_specific_kinetics = cell_specific_kinetics
        self.kinetics_num = kinetics_num

    @jaxtyped
    @beartype
    def forward(
        self,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Compute the expected unspliced and spliced RNA counts based on the dynamics model.

        This method takes a context dictionary containing observed data and parameters,
        computes the expected unspliced and spliced counts according to the dynamics model,
        and updates the context with the results.

        Args:
            context: Dictionary containing model context with the following required keys:
                - u_obs: Observed unspliced counts (BatchTensor)
                - s_obs: Observed spliced counts (BatchTensor)
                - alpha: Transcription rate (ParamTensor)
                - beta: Splicing rate (ParamTensor)
                - gamma: Degradation rate (ParamTensor)

                And optional keys:
                - scaling: Scaling factor (ParamTensor)
                - t: Time points (BatchTensor)

        Returns:
            Updated context dictionary with the following additional keys:
                - u_expected: Expected unspliced counts (BatchTensor)
                - s_expected: Expected spliced counts (BatchTensor)
        """
        # Intentional duplication: Context validation
        # This functionality might be extracted to a utility function in the future
        validation_result = validate_context(
            self.__class__.__name__,
            context,
            required_keys=["u_obs", "s_obs", "alpha", "beta", "gamma"],
            tensor_keys=["u_obs", "s_obs", "alpha", "beta", "gamma"],
        )

        if isinstance(validation_result, dict):
            # Extract required values from context
            u_obs = context["u_obs"]
            s_obs = context["s_obs"]
            alpha = context["alpha"]
            beta = context["beta"]
            gamma = context["gamma"]
            scaling = context.get("scaling")
            t = context.get("t")

            # Compute expected counts
            u_expected, s_expected = self._compute_expected_counts(
                u_obs, s_obs, alpha, beta, gamma, scaling, t
            )

            # Update context with expected counts
            context["u_expected"] = u_expected
            context["s_expected"] = s_expected

            return context
        else:
            # If validation failed, raise an error
            raise ValueError(f"Error in dynamics model forward pass: {validation_result.error}")

    @jaxtyped
    @beartype
    def _compute_expected_counts(
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
        Compute the expected unspliced and spliced RNA counts.

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
        # Intentional duplication: Compute steady state
        # This functionality might be extracted to a utility function in the future
        u_ss, s_ss = self.steady_state(alpha, beta, gamma)

        # If time points are not provided, use steady state
        if t is None:
            # Use steady state as expected counts
            u_expected = u_ss
            s_expected = s_ss

            # Expand to match batch size if needed
            if u.dim() > u_expected.dim():
                batch_size = u.shape[0]
                u_expected = u_expected.unsqueeze(0).expand(batch_size, -1)
                s_expected = s_expected.unsqueeze(0).expand(batch_size, -1)
        else:
            # Compute expected counts using analytical solution
            u_expected = u_ss - (u_ss - u) * torch.exp(-beta * t)
            s_expected = s_ss - (s_ss - s) * torch.exp(-gamma * t) - (
                beta * (u_ss - u) / (gamma - beta)
            ) * (torch.exp(-beta * t) - torch.exp(-gamma * t))

        # Apply scaling if provided
        if scaling is not None:
            u_expected = u_expected * scaling
            s_expected = s_expected * scaling

        return u_expected, s_expected

    @jaxtyped
    @beartype
    def steady_state(
        self,
        alpha: Union[ParamTensor, torch.Tensor],
        beta: Union[ParamTensor, torch.Tensor],
        gamma: Union[ParamTensor, torch.Tensor],
        **kwargs: Any,
    ) -> Tuple[
        Union[ParamTensor, torch.Tensor], Union[ParamTensor, torch.Tensor]
    ]:
        """
        Compute the steady-state unspliced and spliced RNA counts.

        Args:
            alpha: Transcription rate
            beta: Splicing rate
            gamma: Degradation rate
            **kwargs: Additional model-specific parameters

        Returns:
            Tuple of (steady-state unspliced counts, steady-state spliced counts)
        """
        # At steady state, du/dt = 0 and ds/dt = 0
        # From du/dt = 0: alpha - beta * u = 0 => u = alpha / beta
        u_ss = alpha / beta

        # From ds/dt = 0: beta * u - gamma * s = 0 => s = beta * u / gamma
        # Substituting u = alpha / beta: s = beta * (alpha / beta) / gamma = alpha / gamma
        s_ss = alpha / gamma

        return u_ss, s_ss

    @jaxtyped
    @beartype
    def predict_future_states(
        self,
        current_state: Tuple[torch.Tensor, torch.Tensor],
        time_delta: Union[float, torch.Tensor],
        alpha: torch.Tensor,
        beta: torch.Tensor,
        gamma: torch.Tensor,
        scaling: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict future states based on current state and parameters.

        Args:
            current_state: Tuple of (current unspliced counts, current spliced counts)
            time_delta: Time difference for prediction
            alpha: Transcription rate
            beta: Splicing rate
            gamma: Degradation rate
            scaling: Optional scaling factor for the dynamics
            **kwargs: Additional model-specific parameters

        Returns:
            Tuple of (predicted unspliced counts, predicted spliced counts)
        """
        # Extract current state
        u_current, s_current = current_state

        # Convert time_delta to tensor if it's a float
        if isinstance(time_delta, float):
            time_delta = torch.tensor(time_delta)

        # Compute steady state
        u_ss, s_ss = self.steady_state(alpha, beta, gamma)

        # Compute future state using analytical solution
        u_future = u_ss - (u_ss - u_current) * torch.exp(-beta * time_delta)
        s_future = s_ss - (s_ss - s_current) * torch.exp(-gamma * time_delta) - (
            beta * (u_ss - u_current) / (gamma - beta)
        ) * (torch.exp(-beta * time_delta) - torch.exp(-gamma * time_delta))

        # Apply scaling if provided
        if scaling is not None:
            u_future = u_future * scaling
            s_future = s_future * scaling

        return u_future, s_future

    @jaxtyped
    @beartype
    def simulate(
        self,
        u0: torch.Tensor,
        s0: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        gamma: torch.Tensor,
        scaling: Optional[torch.Tensor] = None,
        t_max: float = 10.0,
        n_steps: int = 100,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Simulate the dynamics model forward in time using analytical solution.

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
        # Create time points
        t = torch.linspace(0, t_max, n_steps)

        # Compute steady state
        u_ss, s_ss = self.steady_state(alpha, beta, gamma)

        # Initialize arrays for results
        u = torch.zeros((n_steps, u0.shape[0]))
        s = torch.zeros((n_steps, s0.shape[0]))

        # Compute solution at each time point
        for i, ti in enumerate(t):
            # Compute solution using analytical formula
            u[i] = u_ss - (u_ss - u0) * torch.exp(-beta * ti)
            s[i] = s_ss - (s_ss - s0) * torch.exp(-gamma * ti) - (
                beta * (u_ss - u0) / (gamma - beta)
            ) * (torch.exp(-beta * ti) - torch.exp(-gamma * ti))

        # Apply scaling if provided
        if scaling is not None:
            u = u * scaling
            s = s * scaling

        return t, u, s


@DynamicsModelRegistry.register("nonlinear_direct")
class NonlinearDynamicsModelDirect:
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

    This implementation directly implements the DynamicsModel Protocol
    without inheriting from BaseDynamicsModel.

    Attributes:
        name: The name of the model
        description: A brief description of the model
        shared_time: Whether to use shared time across cells
        t_scale_on: Whether to use time scaling
        cell_specific_kinetics: Type of cell-specific kinetics
        kinetics_num: Number of kinetics
    """

    name: ClassVar[str] = "nonlinear_direct"
    description: ClassVar[str] = "Nonlinear RNA velocity dynamics model with saturation effects (Protocol-First implementation)"

    def __init__(
        self,
        name: str = "nonlinear_dynamics_model_direct",
        shared_time: bool = True,
        t_scale_on: bool = False,
        cell_specific_kinetics: Optional[str] = None,
        kinetics_num: Optional[int] = None,
        **kwargs,
    ):
        """
        Initialize the nonlinear dynamics model.

        Args:
            name: A unique name for this component instance.
            shared_time: Whether to use shared time across cells.
            t_scale_on: Whether to use time scaling.
            cell_specific_kinetics: Type of cell-specific kinetics.
            kinetics_num: Number of kinetics.
            **kwargs: Additional keyword arguments.
        """
        self.name = name
        self.shared_time = shared_time
        self.t_scale_on = t_scale_on
        self.cell_specific_kinetics = cell_specific_kinetics
        self.kinetics_num = kinetics_num

    @jaxtyped
    @beartype
    def forward(
        self,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Compute the expected unspliced and spliced RNA counts based on the nonlinear dynamics model.

        This method takes a context dictionary containing observed data and parameters,
        computes the expected unspliced and spliced counts according to the nonlinear dynamics model,
        and updates the context with the results.

        Args:
            context: Dictionary containing model context with the following required keys:
                - u_obs: Observed unspliced counts (BatchTensor)
                - s_obs: Observed spliced counts (BatchTensor)
                - alpha: Transcription rate (ParamTensor)
                - beta: Splicing rate (ParamTensor)
                - gamma: Degradation rate (ParamTensor)

                And optional keys:
                - scaling: Scaling factor (ParamTensor)
                - t: Time points (BatchTensor)
                - k_alpha: Saturation constant for transcription (ParamTensor)
                - k_beta: Saturation constant for splicing (ParamTensor)

        Returns:
            Updated context dictionary with the following additional keys:
                - u_expected: Expected unspliced counts (BatchTensor)
                - s_expected: Expected spliced counts (BatchTensor)
        """
        # Validate context
        validation_result = validate_context(
            component_name=self.__class__.__name__,
            context=context,
            required_keys=["u_obs", "s_obs", "alpha", "beta", "gamma"],
            tensor_keys=["u_obs", "s_obs", "alpha", "beta", "gamma"],
        )

        if isinstance(validation_result, dict):
            # Extract required values from context
            u_obs = context["u_obs"]
            s_obs = context["s_obs"]
            alpha = context["alpha"]
            beta = context["beta"]
            gamma = context["gamma"]
            scaling = context.get("scaling")
            t = context.get("t")
            k_alpha = context.get("k_alpha")
            k_beta = context.get("k_beta")

            # Set default values for saturation constants if not provided
            if k_alpha is None:
                k_alpha = alpha
            if k_beta is None:
                k_beta = beta

            # Compute expected counts
            u_expected, s_expected = self._compute_expected_counts(
                u_obs, s_obs, alpha, beta, gamma, scaling, t, k_alpha, k_beta
            )

            # Update context with expected counts
            context["u_expected"] = u_expected
            context["s_expected"] = s_expected

            return context
        else:
            # If validation failed, raise an error
            raise ValueError(f"Error in nonlinear dynamics model forward pass: {validation_result.error}")

    @jaxtyped
    @beartype
    def _compute_expected_counts(
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
        Compute the expected unspliced and spliced RNA counts based on the nonlinear dynamics model.

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

        # For the nonlinear model, we need to use numerical integration
        # since there's no simple analytical solution
        # For now, we'll use a simple Euler method

        # Compute steady state
        u_ss, s_ss = self.steady_state(alpha, beta, gamma, k_alpha, k_beta)

        # If time is not provided, we assume steady state
        if t is None:
            u_expected = u_ss
            s_expected = s_ss

            # Expand to match batch size if needed
            if u.dim() > u_expected.dim():
                batch_size = u.shape[0]
                u_expected = u_expected.unsqueeze(0).expand(batch_size, -1)
                s_expected = s_expected.unsqueeze(0).expand(batch_size, -1)
        else:
            # For the nonlinear model, we need to use numerical integration
            # But for now, we'll use a simple Euler method

            # Define the ODE system with saturation effects
            def dudt(u, s):
                return alpha / (1 + u / k_alpha) - beta * u / (k_beta + u)

            def dsdt(u, s):
                return beta * u / (k_beta + u) - gamma * s

            # Simple Euler integration
            dt = 0.01  # Small time step for stability
            steps = 100  # Number of steps

            # Initialize with observed values
            u_current = u
            s_current = s

            # Integrate forward in time
            for _ in range(steps):
                u_next = u_current + dt * dudt(u_current, s_current)
                s_next = s_current + dt * dsdt(u_current, s_current)
                u_current = u_next
                s_current = s_next

            u_expected = u_current
            s_expected = s_current

        # Apply scaling if provided
        if scaling is not None:
            u_expected = u_expected * scaling
            s_expected = s_expected * scaling

        return u_expected, s_expected

    @jaxtyped
    @beartype
    def steady_state(
        self,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        gamma: torch.Tensor,
        k_alpha: Optional[torch.Tensor] = None,
        k_beta: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the steady state of the nonlinear dynamics model.

        Args:
            alpha: Transcription rates [genes]
            beta: Splicing rates [genes]
            gamma: Degradation rates [genes]
            k_alpha: Saturation constant for transcription [genes], defaults to alpha
            k_beta: Saturation constant for splicing [genes], defaults to beta
            **kwargs: Additional model-specific parameters

        Returns:
            Tuple of (steady state unspliced counts, steady state spliced counts)
        """
        # Set default values for saturation constants if not provided
        if k_alpha is None:
            k_alpha = alpha
        if k_beta is None:
            k_beta = beta

        # For the nonlinear model, the steady state is more complex
        # We'll use a numerical approach to find it

        # Initialize with a guess
        u_ss = alpha / beta
        s_ss = alpha / gamma

        # Refine the steady state using fixed-point iteration
        for _ in range(100):
            u_ss_new = alpha / (beta * u_ss / (k_beta + u_ss))
            s_ss_new = beta * u_ss / (k_beta + u_ss) / gamma

            # Check for convergence
            if torch.allclose(u_ss, u_ss_new) and torch.allclose(s_ss, s_ss_new):
                break

            u_ss = u_ss_new
            s_ss = s_ss_new

        return u_ss, s_ss

    @jaxtyped
    @beartype
    def predict_future_states(
        self,
        current_state: Tuple[torch.Tensor, torch.Tensor],
        time_delta: Union[float, torch.Tensor],
        alpha: torch.Tensor,
        beta: torch.Tensor,
        gamma: torch.Tensor,
        scaling: Optional[torch.Tensor] = None,
        k_alpha: Optional[torch.Tensor] = None,
        k_beta: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict future states based on current state and parameters.

        Args:
            current_state: Tuple of (current unspliced counts, current spliced counts)
            time_delta: Time difference for prediction
            alpha: Transcription rate
            beta: Splicing rate
            gamma: Degradation rate
            scaling: Optional scaling factor for the dynamics
            k_alpha: Saturation constant for transcription, defaults to alpha
            k_beta: Saturation constant for splicing, defaults to beta
            **kwargs: Additional model-specific parameters

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

        # Convert time_delta to tensor if it's a float
        if isinstance(time_delta, float):
            time_delta = torch.tensor(time_delta)

        # For the nonlinear model, we need to use numerical integration
        # But for now, we'll use a simple Euler method

        # Define the ODE system with saturation effects
        def dudt(u, s):
            return alpha / (1 + u / k_alpha) - beta * u / (k_beta + u)

        def dsdt(u, s):
            return beta * u / (k_beta + u) - gamma * s

        # Simple Euler integration
        dt = 0.01  # Small time step for stability

        # Handle both float and tensor time_delta
        if isinstance(time_delta, float):
            time_delta_value = time_delta
        elif isinstance(time_delta, torch.Tensor):
            time_delta_value = time_delta.item()
        else:
            time_delta_value = float(time_delta)  # Try to convert to float

        # Number of steps
        steps = int(time_delta_value / dt)

        # Initialize with current state
        u_future = u_current
        s_future = s_current

        # Integrate forward in time
        for _ in range(steps):
            u_next = u_future + dt * dudt(u_future, s_future)
            s_next = s_future + dt * dsdt(u_future, s_future)
            u_future = u_next
            s_future = s_next

        # Apply scaling if provided
        if scaling is not None:
            u_future = u_future * scaling
            s_future = s_future * scaling

        return u_future, s_future

    @jaxtyped
    @beartype
    def simulate(
        self,
        u0: torch.Tensor,
        s0: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        gamma: torch.Tensor,
        scaling: Optional[torch.Tensor] = None,
        t_max: float = 10.0,
        n_steps: int = 100,
        k_alpha: Optional[torch.Tensor] = None,
        k_beta: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

        # Set up time points
        times = torch.linspace(0, t_max, n_steps)

        # Define the ODE system with saturation effects
        def dudt(u, s):
            return alpha / (1 + u / k_alpha) - beta * u / (k_beta + u)

        def dsdt(u, s):
            return beta * u / (k_beta + u) - gamma * s

        # Initialize arrays to store results
        u_t = torch.zeros((n_steps, u0.shape[0]))
        s_t = torch.zeros((n_steps, s0.shape[0]))

        # Set initial conditions
        u_t[0] = u0
        s_t[0] = s0

        # Simple Euler integration for each time step
        for i in range(1, n_steps):
            dt = float(times[i] - times[i - 1])  # Time step

            # Euler method for each time step
            u_t[i] = u_t[i - 1] + dt * dudt(u_t[i - 1], s_t[i - 1])
            s_t[i] = s_t[i - 1] + dt * dsdt(u_t[i - 1], s_t[i - 1])

        # Apply scaling if provided
        if scaling is not None:
            u_t = u_t * scaling
            s_t = s_t * scaling

        return times, u_t, s_t
