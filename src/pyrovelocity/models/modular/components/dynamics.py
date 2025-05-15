"""
Dynamics model implementations for PyroVelocity's modular architecture.

This module contains dynamics model implementations that directly implement the
DynamicsModel Protocol. These implementations follow the Protocol-First approach,
which embraces composition over inheritance and allows for more flexible component composition.

This module has been simplified to include only the essential components needed for
validation against the legacy implementation:
- StandardDynamicsModel: Standard RNA velocity dynamics model
- LegacyDynamicsModel: Legacy RNA velocity dynamics model that exactly matches the legacy implementation
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
                # Use parameters directly without creating duplicate deterministic nodes
                # This avoids adding extra dimensions to the parameters

                # Compute steady state values
                u_inf = alpha / beta  # Shape: [1, 1, n_genes] or [num_samples, 1, n_genes]
                s_inf = alpha / gamma  # Shape: [1, 1, n_genes] or [num_samples, 1, n_genes]

                # Compute switching time
                switching = t0 + dt_switching  # Shape: [1, 1, n_genes] or [num_samples, 1, n_genes]

                # Create deterministic sites for steady state and switching
                # Match the legacy model exactly by using event_dim=0
                # In the legacy model, these are created within the gene_plate context
                # See _velocity_model.py line 696-700
                u_inf = pyro.deterministic("u_inf", u_inf, event_dim=0)
                s_inf = pyro.deterministic("s_inf", s_inf, event_dim=0)
                switching = pyro.deterministic("switching", switching, event_dim=0)

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

            # Ensure ut and st have the correct shape [num_samples, n_cells, n_genes]
            # First, determine the expected shape
            if alpha.dim() == 3:
                num_samples = alpha.shape[0]
            else:
                num_samples = 1

            # Reshape to ensure we have exactly 3 dimensions with the correct shape
            # This is more robust than using squeeze which can remove dimensions we want to keep
            if ut.dim() > 3:
                # Reshape to [num_samples, n_cells, n_genes]
                ut = ut.reshape(num_samples, num_cells, num_genes)
                st = st.reshape(num_samples, num_cells, num_genes)
            elif ut.dim() < 3:
                # Add missing dimensions
                if ut.dim() == 2:  # [n_cells, n_genes]
                    ut = ut.unsqueeze(0)  # [1, n_cells, n_genes]
                    st = st.unsqueeze(0)  # [1, n_cells, n_genes]
                elif ut.dim() == 1:  # [n_genes]
                    ut = ut.unsqueeze(0).unsqueeze(0)  # [1, 1, n_genes]
                    st = st.unsqueeze(0).unsqueeze(0)  # [1, 1, n_genes]

            # Print shapes after reshaping
                        
            # Ensure ut and st have the correct shape [num_samples, n_cells, n_genes]
            # This is critical for proper shape in posterior samples
            # In the legacy model, ut and st should have shape [num_samples, n_cells, n_genes]
            # We need to be very aggressive about reshaping to match the legacy model exactly

            # First, determine the expected shape
            if alpha.dim() == 3:
                num_samples = alpha.shape[0]
            else:
                num_samples = 1

            # Reshape to ensure we have exactly 3 dimensions with the correct shape
            # This is more robust than using squeeze which can remove dimensions we want to keep
            ut = ut.reshape(num_samples, num_cells, num_genes)
            st = st.reshape(num_samples, num_cells, num_genes)

            # In the legacy model, ut and st are not registered as deterministic nodes here
            # They are registered in the get_likelihood method, which is called within a nested plate context
            # We'll skip creating deterministic nodes here and let the likelihood model handle it

            # Instead, we'll just store the values in the context
            u = ut
            s = st

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



