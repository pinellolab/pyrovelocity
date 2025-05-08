"""
Dynamics model implementations for PyroVelocity's modular architecture.

This module contains dynamics model implementations that directly implement the
DynamicsModel Protocol. These implementations follow the Protocol-First approach,
which embraces composition over inheritance and allows for more flexible component composition.
"""

from typing import Any, ClassVar, Dict, Optional, Tuple, Union

import torch
from beartype import beartype
from jaxtyping import jaxtyped

from pyrovelocity.models.modular.constants import CELLS_DIM, GENES_DIM
from pyrovelocity.models.modular.interfaces import (
    BatchTensor,
    DynamicsModel,
    ParamTensor,
)
from pyrovelocity.models.modular.registry import DynamicsModelRegistry
from pyrovelocity.models.modular.utils.context_utils import validate_context


@DynamicsModelRegistry.register("standard")
class StandardDynamicsModel:
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
    the DynamicsModel Protocol.

    Attributes:
        name: The name of the model
        description: A brief description of the model
        shared_time: Whether to use shared time across cells
        t_scale_on: Whether to use time scaling
        cell_specific_kinetics: Type of cell-specific kinetics
        kinetics_num: Number of kinetics
    """

    name: ClassVar[str] = "standard"
    description: ClassVar[str] = "Standard RNA velocity dynamics model"

    def __init__(
        self,
        name: str = "dynamics_model",
        shared_time: bool = True,
        t_scale_on: bool = False,
        cell_specific_kinetics: Optional[str] = None,
        kinetics_num: Optional[int] = None,
        correct_library_size: Union[bool, str] = True,
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
            correct_library_size: Whether to correct for library size.
            **kwargs: Additional keyword arguments.
        """
        self.name = name
        self.shared_time = shared_time
        self.t_scale_on = t_scale_on
        self.cell_specific_kinetics = cell_specific_kinetics
        self.kinetics_num = kinetics_num
        self.correct_library_size = correct_library_size

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
                - u_read_depth: Unspliced read depth (BatchTensor)
                - s_read_depth: Spliced read depth (BatchTensor)

        Returns:
            Updated context dictionary with the following additional keys:
                - u_expected: Expected unspliced counts (BatchTensor)
                - s_expected: Expected spliced counts (BatchTensor)
                - ut: Latent unspliced counts (BatchTensor)
                - st: Latent spliced counts (BatchTensor)
        """
        # Validate context
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

            # Extract read depths if available
            u_read_depth = context.get("u_read_depth")
            s_read_depth = context.get("s_read_depth")

            # If read depths are not provided, create default ones
            if u_read_depth is None and hasattr(self, 'correct_library_size') and self.correct_library_size:
                # Create default read depths (will be used in _compute_expected_counts)
                if u_obs.dim() == 2:
                    # Shape: [batch_size, 1]
                    u_read_depth = torch.ones(u_obs.shape[0], 1)
                    s_read_depth = torch.ones(s_obs.shape[0], 1)
                else:
                    # Shape: [1]
                    u_read_depth = torch.ones(1)
                    s_read_depth = torch.ones(1)

                # Add to context
                context["u_read_depth"] = u_read_depth
                context["s_read_depth"] = s_read_depth

            # Compute expected counts
            u_expected, s_expected = self._compute_expected_counts(
                u_obs, s_obs, alpha, beta, gamma, scaling, t
            )

            # Update context with expected counts
            context["u_expected"] = u_expected
            context["s_expected"] = s_expected

            # Create latent variables ut and st using pyro.deterministic
            # This matches the legacy implementation's approach
            import pyro

            # Create latent variables with the same shape as in the legacy implementation
            # Reshape parameters to match legacy implementation shape if needed
            if u_expected.dim() == 2 and u_expected.shape[0] > 1:
                # Add an extra dimension to match legacy shape (num_cells, 1, n_genes)
                # This is critical for proper broadcasting during velocity calculation
                u_expected_reshaped = u_expected.unsqueeze(1)
                s_expected_reshaped = s_expected.unsqueeze(1)
            else:
                u_expected_reshaped = u_expected
                s_expected_reshaped = s_expected

            # Create latent variables
            ut = pyro.deterministic("ut", u_expected_reshaped)
            st = pyro.deterministic("st", s_expected_reshaped)

            # In the legacy implementation, these are also normalized and scaled
            # We'll add that logic here to match exactly
            one = torch.ones_like(ut) * 1e-6

            # Apply ReLU and add small constant for numerical stability
            ut = torch.relu(ut) + one
            st = torch.relu(st) + one

            # If we're using library size correction, normalize and scale
            if hasattr(self, 'correct_library_size') and self.correct_library_size:
                # Normalize
                ut_norm = ut / torch.sum(ut, dim=-1, keepdim=True)
                st_norm = st / torch.sum(st, dim=-1, keepdim=True)

                # Register as deterministic variables
                ut_norm = pyro.deterministic("ut_norm", ut_norm)
                st_norm = pyro.deterministic("st_norm", st_norm)

                # Add small constant and scale by read depth
                if u_read_depth is not None and s_read_depth is not None:
                    # Reshape read depths to match ut_norm and st_norm
                    if u_read_depth.dim() == 1:
                        u_read_depth = u_read_depth.unsqueeze(1)
                    if s_read_depth.dim() == 1:
                        s_read_depth = s_read_depth.unsqueeze(1)

                    # Scale by read depth
                    ut = (ut_norm + one) * u_read_depth
                    st = (st_norm + one) * s_read_depth
                else:
                    # Use default scaling
                    ut = (ut_norm + one)
                    st = (st_norm + one)

            # Add latent variables to context
            context["ut"] = ut
            context["st"] = st

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
        # Compute steady state
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
        Union[ParamTensor, torch.Tensor],
        Union[ParamTensor, torch.Tensor],
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

        # Set initial conditions
        u[0] = u0.clone()
        s[0] = s0.clone()

        # Compute solution at each time point
        for i, ti in enumerate(t):
            if i > 0:  # Skip the first point which is already set
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


@DynamicsModelRegistry.register("legacy")
class LegacyDynamicsModel:
    """Legacy dynamics model for RNA velocity that exactly matches the legacy implementation.

    This model implements the standard model of RNA velocity but with the exact same
    parameter shapes and behavior as the legacy implementation:

    du/dt = alpha - beta * u
    ds/dt = beta * u - gamma * s

    where:
    - u is the unspliced mRNA count
    - s is the spliced mRNA count
    - alpha is the transcription rate
    - beta is the splicing rate
    - gamma is the degradation rate

    This implementation is specifically designed to match the legacy implementation
    in terms of parameter shapes and behavior.

    Attributes:
        name: The name of the model
        description: A brief description of the model
        shared_time: Whether to use shared time across cells
        t_scale_on: Whether to use time scaling
        cell_specific_kinetics: Type of cell-specific kinetics
        kinetics_num: Number of kinetics
        correct_library_size: Whether to correct for library size
    """

    name: ClassVar[str] = "legacy"
    description: ClassVar[str] = "Legacy RNA velocity dynamics model"

    def __init__(
        self,
        name: str = "legacy_dynamics_model",
        shared_time: bool = True,
        t_scale_on: bool = False,
        cell_specific_kinetics: Optional[str] = None,
        kinetics_num: Optional[int] = None,
        correct_library_size: Union[bool, str] = True,
        **kwargs,
    ):
        """
        Initialize the legacy dynamics model.

        Args:
            name: A unique name for this component instance.
            shared_time: Whether to use shared time across cells.
            t_scale_on: Whether to use time scaling.
            cell_specific_kinetics: Type of cell-specific kinetics.
            kinetics_num: Number of kinetics.
            correct_library_size: Whether to correct for library size.
            **kwargs: Additional keyword arguments.
        """
        self.name = name
        self.shared_time = shared_time
        self.t_scale_on = t_scale_on
        self.cell_specific_kinetics = cell_specific_kinetics
        self.kinetics_num = kinetics_num
        self.correct_library_size = correct_library_size

    @jaxtyped
    @beartype
    def forward(
        self,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Compute the expected unspliced and spliced RNA counts based on the legacy dynamics model.

        This method takes a context dictionary containing observed data and parameters,
        computes the expected unspliced and spliced counts according to the legacy dynamics model,
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
                - u_read_depth: Unspliced read depth (BatchTensor)
                - s_read_depth: Spliced read depth (BatchTensor)

        Returns:
            Updated context dictionary with the following additional keys:
                - u_expected: Expected unspliced counts (BatchTensor)
                - s_expected: Expected spliced counts (BatchTensor)
                - ut: Latent unspliced counts (BatchTensor)
                - st: Latent spliced counts (BatchTensor)
                - u_inf: Steady-state unspliced counts (ParamTensor)
                - s_inf: Steady-state spliced counts (ParamTensor)
                - switching: Switching time (ParamTensor)
        """
        # Validate context
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

            # Print shapes for debugging
            print(f"LegacyDynamicsModel - u_obs shape: {u_obs.shape}")
            print(f"LegacyDynamicsModel - s_obs shape: {s_obs.shape}")
            print(f"LegacyDynamicsModel - alpha shape: {alpha.shape}")
            print(f"LegacyDynamicsModel - beta shape: {beta.shape}")
            print(f"LegacyDynamicsModel - gamma shape: {gamma.shape}")

            # Extract optional values
            t0 = context.get("t0", torch.zeros_like(alpha))
            dt_switching = context.get("dt_switching", torch.zeros_like(alpha))
            u_scale = context.get("u_scale", torch.ones_like(alpha))
            s_scale = context.get("s_scale", torch.ones_like(alpha))

            # Extract read depths if available
            u_read_depth = context.get("u_read_depth")
            s_read_depth = context.get("s_read_depth")

            # If read depths are not provided, create default ones
            if u_read_depth is None and hasattr(self, 'correct_library_size') and self.correct_library_size:
                # Create default read depths
                if u_obs.dim() == 2:
                    # Shape: [batch_size, 1]
                    u_read_depth = torch.ones(u_obs.shape[0], 1)
                    s_read_depth = torch.ones(s_obs.shape[0], 1)
                else:
                    # Shape: [1]
                    u_read_depth = torch.ones(1)
                    s_read_depth = torch.ones(1)

                # Add to context
                context["u_read_depth"] = u_read_depth
                context["s_read_depth"] = s_read_depth

            # Import pyro for deterministic sites
            import pyro

            # Get dimensions
            num_cells = u_obs.shape[0]
            num_genes = alpha.shape[-1] if alpha.dim() > 0 else 1

            # Create plates with consistent dimensions using constants
            # Use GENES_DIM (-1) for genes and CELLS_DIM (-2) for cells
            gene_plate = pyro.plate("genes", num_genes, dim=GENES_DIM)
            cell_plate = pyro.plate("cells", num_cells, dim=CELLS_DIM)

            # SIMPLIFIED PLATE STRUCTURE MATCHING LEGACY MODEL
            # In the legacy model:
            # - Gene parameters have shape [num_samples, 1, n_genes]
            # - Cell parameters have shape [num_samples, n_cells, 1]
            # - Cell-gene interactions have shape [num_samples, n_cells, n_genes]

            # First, reshape gene parameters to match legacy model shape
            # We want alpha, beta, gamma to have shape [1, 1, n_genes]
            if alpha.dim() == 1:  # [n_genes]
                alpha = alpha.unsqueeze(0).unsqueeze(0)  # [1, 1, n_genes]
                beta = beta.unsqueeze(0).unsqueeze(0)  # [1, 1, n_genes]
                gamma = gamma.unsqueeze(0).unsqueeze(0)  # [1, 1, n_genes]
                t0 = t0.unsqueeze(0).unsqueeze(0)  # [1, 1, n_genes]
                dt_switching = dt_switching.unsqueeze(0).unsqueeze(0)  # [1, 1, n_genes]
            elif alpha.dim() == 2:  # [num_samples, n_genes]
                alpha = alpha.unsqueeze(1)  # [num_samples, 1, n_genes]
                beta = beta.unsqueeze(1)  # [num_samples, 1, n_genes]
                gamma = gamma.unsqueeze(1)  # [num_samples, 1, n_genes]
                t0 = t0.unsqueeze(1)  # [num_samples, 1, n_genes]
                dt_switching = dt_switching.unsqueeze(1)  # [num_samples, 1, n_genes]

            # Sample gene-specific parameters with the correct shape
            with gene_plate:
                # Create deterministic sites for gene parameters
                alpha = pyro.deterministic("alpha_det", alpha)
                beta = pyro.deterministic("beta_det", beta)
                gamma = pyro.deterministic("gamma_det", gamma)

                # Compute steady state values
                u_inf = alpha / beta  # Shape: [1, 1, n_genes] or [num_samples, 1, n_genes]
                s_inf = alpha / gamma  # Shape: [1, 1, n_genes] or [num_samples, 1, n_genes]

                # Compute switching time
                switching = t0 + dt_switching  # Shape: [1, 1, n_genes] or [num_samples, 1, n_genes]

                # Create deterministic sites for steady state and switching
                u_inf = pyro.deterministic("u_inf", u_inf)
                s_inf = pyro.deterministic("s_inf", s_inf)
                switching = pyro.deterministic("switching", switching)

            # Next, sample cell-specific parameters
            with cell_plate:
                # Generate cell_time if not provided
                cell_time = context.get("cell_time")
                if cell_time is None:
                    # Create a uniform distribution of cell times between 0 and 1
                    cell_time = torch.linspace(0, 1, num_cells).unsqueeze(1)  # [n_cells, 1]
                    cell_time = pyro.sample(
                        "cell_time",
                        pyro.distributions.Delta(cell_time)
                    )  # Shape: [n_cells, 1]
                    context["cell_time"] = cell_time
                elif cell_time.dim() == 1:  # [n_cells]
                    cell_time = cell_time.unsqueeze(1)  # [n_cells, 1]

                # Ensure cell_time has shape [n_cells, 1] or [num_samples, n_cells, 1]
                if cell_time.dim() == 2:  # [n_cells, 1]
                    # If we have multiple samples, expand cell_time
                    if alpha.dim() == 3 and alpha.shape[0] > 1:
                        # Expand to [num_samples, n_cells, 1]
                        cell_time = cell_time.unsqueeze(0).expand(alpha.shape[0], -1, -1)

            # Compute ut and st with proper broadcasting
            # We need to ensure:
            # - alpha, beta, gamma have shape [1, 1, n_genes] or [num_samples, 1, n_genes]
            # - cell_time has shape [n_cells, 1] or [num_samples, n_cells, 1]
            # - ut, st will have shape [n_cells, n_genes] or [num_samples, n_cells, n_genes]

            # Compute ut and st
            ut = u_inf * (1 - torch.exp(-beta * cell_time))
            st = s_inf * (1 - torch.exp(-gamma * cell_time)) - (
                alpha / (gamma - beta + 1e-8)  # Add small epsilon to avoid division by zero
            ) * (torch.exp(-beta * cell_time) - torch.exp(-gamma * cell_time))

            # Create deterministic sites for ut and st
            ut = pyro.deterministic("ut", ut)
            st = pyro.deterministic("st", st)

            # Create deterministic sites for u and s (observed values)
            u = pyro.deterministic("u", ut)
            s = pyro.deterministic("s", st)

            # Add to context
            context["u_inf"] = u_inf
            context["s_inf"] = s_inf
            context["switching"] = switching
            context["ut"] = ut
            context["st"] = st
            context["u"] = u
            context["s"] = s

            # Add expected counts for the likelihood model
            context["u_expected"] = ut
            context["s_expected"] = st

            # Print shapes for debugging
            print(f"LegacyDynamicsModel - cell_time shape: {cell_time.shape}")
            print(f"LegacyDynamicsModel - u_inf shape: {u_inf.shape}")
            print(f"LegacyDynamicsModel - switching shape: {switching.shape}")
            print(f"LegacyDynamicsModel - ut shape: {ut.shape}")
            print(f"LegacyDynamicsModel - st shape: {st.shape}")

            return context
        else:
            # If validation failed, raise an error
            raise ValueError(f"Error in dynamics model forward pass: {validation_result.error}")

    @jaxtyped
    @beartype
    def steady_state(
        self,
        alpha: Union[ParamTensor, torch.Tensor],
        beta: Union[ParamTensor, torch.Tensor],
        gamma: Union[ParamTensor, torch.Tensor],
        **kwargs: Any,
    ) -> Tuple[
        Union[ParamTensor, torch.Tensor],
        Union[ParamTensor, torch.Tensor],
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


@DynamicsModelRegistry.register("nonlinear")
class NonlinearDynamicsModel:
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

    This implementation directly implements the DynamicsModel Protocol.

    Attributes:
        name: The name of the model
        description: A brief description of the model
        shared_time: Whether to use shared time across cells
        t_scale_on: Whether to use time scaling
        cell_specific_kinetics: Type of cell-specific kinetics
        kinetics_num: Number of kinetics
    """

    name: ClassVar[str] = "nonlinear"
    description: ClassVar[
        str
    ] = "Nonlinear RNA velocity dynamics model with saturation effects"

    def __init__(
        self,
        name: str = "nonlinear_dynamics_model",
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
        if torch.all(k_alpha > 1e5) and torch.all(k_beta > 1e5):
            # Use standard model steady state
            u_ss = alpha / beta
            s_ss = alpha / gamma

            # Apply scaling if provided
            if scaling is not None:
                u_ss = u_ss * scaling
                s_ss = s_ss * scaling

            # Expand to match batch size if needed
            if u.dim() > u_ss.dim():
                batch_size = u.shape[0]
                u_ss = u_ss.unsqueeze(0).expand(batch_size, -1)
                s_ss = s_ss.unsqueeze(0).expand(batch_size, -1)

            return u_ss, s_ss

        # Calculate steady state using the nonlinear model
        u_ss, s_ss = self.steady_state(
            alpha, beta, gamma, k_alpha=k_alpha, k_beta=k_beta
        )

        # Apply scaling if provided
        if scaling is not None:
            u_ss = u_ss * scaling
            s_ss = s_ss * scaling

        # Expand to match batch size if needed
        if u.dim() > u_ss.dim():
            batch_size = u.shape[0]
            u_ss = u_ss.unsqueeze(0).expand(batch_size, -1)
            s_ss = s_ss.unsqueeze(0).expand(batch_size, -1)

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
        if torch.all(k_alpha > 1e5) and torch.all(k_beta > 1e5):
            # Use standard model prediction
            # Compute exponentials
            u_current, s_current = current_state
            expu = torch.exp(-beta * time_delta)
            exps = torch.exp(-gamma * time_delta)

            # Compute future unspliced counts using analytical solution
            u_future = u_current * expu + alpha / beta * (1 - expu)

            # Compute future spliced counts using analytical solution
            # Handle the case where gamma == beta
            is_close = torch.isclose(gamma, beta)

            # For gamma != beta
            expus = (
                (alpha - u_current * beta)
                / (gamma - beta + 1e-8)
                * (exps - expu)
            )
            st = s_current * exps + alpha / gamma * (1 - exps) + expus

            # For gamma == beta
            st_gamma_equals_beta = (
                s_current * expu
                + alpha / beta * (1 - expu)
                - (alpha - beta * u_current) * time_delta * expu
            )

            # Use torch.where to select the appropriate formula
            s_future = torch.where(is_close, st_gamma_equals_beta, st)

            # Apply scaling if provided
            if scaling is not None:
                u_future = u_future * scaling
                s_future = s_future * scaling

            return u_future, s_future

        # For the nonlinear model, we need to use numerical integration
        # since there's no simple analytical solution

        # For the nonlinear model, we need to use numerical integration
        # But for now, we'll use a simple Euler method to avoid torchode issues

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

        steps = int(time_delta_value / dt) + 1
        dt = time_delta_value / steps  # Adjust dt for exact time_delta

        u = u_current.clone()
        s = s_current.clone()

        for _ in range(steps):
            u_new = u + dt * dudt(u, s)
            s_new = s + dt * dsdt(u, s)
            u = u_new
            s = s_new

        u_future = u
        s_future = s

        # The code below shows how we would use torchode for numerical integration
        # but we're using Euler method for now to avoid torchode issues

        # # Define the ODE system with saturation effects
        # def rhs(t, state):
        #     u, s = state
        #     dudt = alpha / (1 + u / k_alpha) - beta * u / (k_beta + u)
        #     dsdt = beta * u / (k_beta + u) - gamma * s
        #     return torch.stack([dudt, dsdt])
        #
        # # Set up time points for integration
        # t0 = torch.tensor(0.0)
        # t1 = torch.tensor(float(time_delta))
        #
        # # Initial state
        # y0 = torch.stack([u_current, s_current])
        #
        # # Solve the ODE system using torchode
        # term = to.ODETerm(rhs)
        # step_method = to.Tsit5(term=term)
        # step_size_controller = to.IntegralController(atol=1e-6, rtol=1e-3, term=term)
        # solver = to.AutoDiffAdjoint(step_method, step_size_controller)
        #
        # # Create the initial value problem
        # problem = to.InitialValueProblem(y0=y0, t_start=t0, t_end=t1)
        #
        # # Solve the ODE
        # solution = solver.solve(problem)
        #
        # # Extract results (final state)
        # final_state = solution.ys[-1]
        # u_future = final_state[0]
        # s_future = final_state[1]

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
        **kwargs,
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

        # Check if k_alpha and k_beta are very large
        # If they are, use the standard model simulation
        if torch.all(k_alpha > 1e3) and torch.all(k_beta > 1e3):
            # Use standard model simulation
            # Use analytical solution
            times = torch.linspace(0, t_max, n_steps)

            # Initialize arrays to store results
            u_t = torch.zeros((n_steps, u0.shape[0]))
            s_t = torch.zeros((n_steps, s0.shape[0]))

            # Set initial conditions
            u_t[0] = u0
            s_t[0] = s0

            # Use the analytical solution for each time point
            for i in range(1, n_steps):
                tau = float(times[i] - times[0])  # Time since start
                current_state = (u_t[i - 1], s_t[i - 1])
                # Compute exponentials
                u_current, s_current = current_state
                expu = torch.exp(-beta * tau)
                exps = torch.exp(-gamma * tau)

                # Compute future unspliced counts using analytical solution
                u_future = u_current * expu + alpha / beta * (1 - expu)

                # Compute future spliced counts using analytical solution
                # Handle the case where gamma == beta
                is_close = torch.isclose(gamma, beta)

                # For gamma != beta
                expus = (
                    (alpha - u_current * beta)
                    / (gamma - beta + 1e-8)
                    * (exps - expu)
                )
                st = s_current * exps + alpha / gamma * (1 - exps) + expus

                # For gamma == beta
                st_gamma_equals_beta = (
                    s_current * expu
                    + alpha / beta * (1 - expu)
                    - (alpha - beta * u_current) * tau * expu
                )

                # Use torch.where to select the appropriate formula
                s_future = torch.where(is_close, st_gamma_equals_beta, st)

                # Apply scaling if provided
                if scaling is not None:
                    u_future = u_future * scaling
                    s_future = s_future * scaling

                u_t[i] = u_future
                s_t[i] = s_future

            return times, u_t, s_t

        # Set up time points
        times = torch.linspace(0, t_max, n_steps)

        # Initialize arrays to store results
        u_t = torch.zeros((n_steps, u0.shape[0]))
        s_t = torch.zeros((n_steps, s0.shape[0]))

        # Set initial conditions
        u_t[0] = u0
        s_t[0] = s0

        # For the nonlinear model, we need to use numerical integration
        # But for now, we'll use a simple Euler method to avoid torchode issues

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

        # The code below shows how we would use torchode for numerical integration
        # but we're using Euler method for now to avoid torchode issues

        # # Define the ODE system with saturation effects
        # def rhs(t, state):
        #     u, s = state
        #     dudt = alpha / (1 + u / k_alpha) - beta * u / (k_beta + u)
        #     dsdt = beta * u / (k_beta + u) - gamma * s
        #     return torch.stack([dudt, dsdt])
        #
        # # Solve the ODE system using torchode
        # term = to.ODETerm(rhs)
        # step_method = to.Tsit5(term=term)
        # step_size_controller = to.IntegralController(atol=1e-6, rtol=1e-3, term=term)
        # solver = to.AutoDiffAdjoint(step_method, step_size_controller)
        #
        # # Create the initial value problem
        # y0 = torch.stack([u0, s0])
        # problem = to.InitialValueProblem(y0=y0, t_start=times[0], t_end=times[-1])
        #
        # # Solve the ODE
        # solution = solver.solve(problem)
        #
        # # Extract results
        # ys = solution.ys
        # u_t = ys[:, 0, :]
        # s_t = ys[:, 1, :]

        return times, u_t, s_t

    @jaxtyped
    @beartype
    def steady_state(
        self,
        alpha: Union[ParamTensor, torch.Tensor],
        beta: Union[ParamTensor, torch.Tensor],
        gamma: Union[ParamTensor, torch.Tensor],
        k_alpha: Optional[Union[ParamTensor, torch.Tensor]] = None,
        k_beta: Optional[Union[ParamTensor, torch.Tensor]] = None,
        **kwargs: Any,
    ) -> Tuple[
        Union[ParamTensor, torch.Tensor], Union[ParamTensor, torch.Tensor]
    ]:
        """Implementation of the steady_state method for the nonlinear dynamics model.

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
        if torch.all(k_alpha > 1e5) and torch.all(k_beta > 1e5):
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
