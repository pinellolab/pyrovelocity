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
    KineticParamTensor,
    LatentCountTensor,
    ParamTensor,
    VelocityTensor,
)
from pyrovelocity.models.modular.registry import DynamicsModelRegistry
from pyrovelocity.models.modular.utils.context_utils import validate_context


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

    @jaxtyped
    @beartype
    def compute_velocity(
        self,
        ut: LatentCountTensor,
        st: LatentCountTensor,
        alpha: KineticParamTensor,
        beta: KineticParamTensor,
        gamma: KineticParamTensor,
        **kwargs: Any,
    ) -> VelocityTensor:
        """
        Compute RNA velocity from latent RNA counts and kinetic parameters.

        This method computes the RNA velocity (ds/dt) based on the legacy dynamics model:
            ds/dt = βu - γs

        This implementation exactly matches the legacy PyroVelocity velocity calculation.

        Args:
            ut: Latent unspliced RNA counts
            st: Latent spliced RNA counts
            alpha: Transcription rate (not used in velocity calculation but kept for interface consistency)
            beta: Splicing rate
            gamma: Degradation rate
            **kwargs: Additional model-specific parameters (e.g., scaling factors)

        Returns:
            RNA velocity tensor with same shape as ut/st

        Raises:
            ValueError: If tensor shapes are incompatible
        """
        # Handle scaling factors if provided (matching legacy implementation)
        u_scale = kwargs.get("u_scale")
        s_scale = kwargs.get("s_scale")

        # For the legacy dynamics model: ds/dt = β * u - γ * s
        # This exactly matches the legacy implementation in compute_mean_vector_field
        if u_scale is not None and s_scale is not None:
            # For models with two scales (Gaussian models)
            # velocity = β * ut / (u_scale / s_scale) - γ * st
            scale = u_scale / s_scale
            velocity = beta * ut / scale - gamma * st
        elif u_scale is not None:
            # For models with one scale (Poisson models)
            # velocity = β * ut / u_scale - γ * st
            velocity = beta * ut / u_scale - gamma * st
        else:
            # For models with no scaling
            # velocity = β * ut - γ * st
            velocity = beta * ut - gamma * st

        return velocity


@DynamicsModelRegistry.register("piecewise_activation")
class PiecewiseActivationDynamicsModel:
    """Piecewise activation dynamics model with corrected dimensional analysis.

    This model implements dimensionless analytical dynamics with piecewise constant
    transcription rates using the corrected parameterization that eliminates
    parameter redundancy. The system has three phases:

    Phase 1 (Off): t* < t*_on
        α*(t*) = 1.0 (fixed reference transcription rate)

    Phase 2 (On): t*_on ≤ t* < t*_on + δ*
        α*(t*) = R_on (fold-change from reference)

    Phase 3 (Return to Off): t* ≥ t*_on + δ*
        α*(t*) = 1.0 (back to reference)

    The dimensionless system is:
        du*/dt* = α*(t*) - u*
        ds*/dt* = u* - γ*s*

    With fixed steady-state initial conditions:
        u*_0 = 1.0, s*_0 = 1.0/γ*

    Key corrections:
    - α*_off = 1.0 (fixed, not inferred) eliminates U₀ᵢ vs α*_off redundancy
    - R_on = α*_on represents fold-change during activation
    - t*_on ~ Normal allows negative values for pre-activation scenarios
    - Initial conditions are fixed, not inferred

    Special handling for γ* = 1 boundary case with τe^(-τ) terms.

    Attributes:
        name: The name of the model
        description: A brief description of the model
        eps: Small epsilon for numerical stability near γ* = 1
    """

    name: ClassVar[str] = "piecewise_activation"
    description: ClassVar[str] = "Piecewise activation dynamics with analytical solutions"

    def __init__(
        self,
        name: str = "piecewise_dynamics_model",
        eps: float = 1e-6,
        **kwargs,
    ):
        """
        Initialize the piecewise activation dynamics model.

        Args:
            name: A unique name for this component instance.
            eps: Small epsilon for numerical stability near γ* = 1.
            **kwargs: Additional keyword arguments.
        """
        self.name = name
        self.eps = eps

    @jaxtyped
    @beartype
    def forward(
        self,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Compute the expected unspliced and spliced RNA counts based on piecewise dynamics.

        This method takes a context dictionary containing observed data and parameters,
        computes the expected unspliced and spliced counts according to the piecewise
        activation dynamics model, and updates the context with the results.

        Args:
            context: Dictionary containing model context with the following required keys:
                - u_obs: Observed unspliced counts (BatchTensor)
                - s_obs: Observed spliced counts (BatchTensor)
                - alpha_off: Basal transcription rate (ParamTensor)
                - alpha_on: Active transcription rate (ParamTensor)
                - gamma_star: Relative degradation rate (ParamTensor)
                - t_on_star: Activation onset time (ParamTensor)
                - delta_star: Activation duration (ParamTensor)
                - t_star: Dimensionless time points (BatchTensor)

        Returns:
            Updated context dictionary with the following additional keys:
                - u_expected: Expected unspliced counts (BatchTensor)
                - s_expected: Expected spliced counts (BatchTensor)
                - ut: Latent unspliced counts (BatchTensor)
                - st: Latent spliced counts (BatchTensor)
        """
        # Validate context (expect R_on instead of alpha_on, alpha_off fixed at 1.0)
        validation_result = validate_context(
            self.__class__.__name__,
            context,
            required_keys=[
                "u_obs", "s_obs", "R_on", "gamma_star",
                "t_on_star", "delta_star", "t_star"
            ],
            tensor_keys=[
                "u_obs", "s_obs", "R_on", "gamma_star",
                "t_on_star", "delta_star", "t_star"
            ],
        )

        if isinstance(validation_result, dict):
            # Extract required values from context
            u_obs = context["u_obs"]
            s_obs = context["s_obs"]
            R_on = context["R_on"]
            gamma_star = context["gamma_star"]
            t_on_star = context["t_on_star"]
            delta_star = context["delta_star"]
            t_star = context["t_star"]

            # Create fixed alpha_off tensor (always 1.0) and compute alpha_on from R_on
            n_genes = R_on.shape[0] if R_on.dim() > 0 else 1
            alpha_off = torch.ones(n_genes, device=R_on.device, dtype=R_on.dtype)
            alpha_on = R_on  # Since alpha_off = 1.0, alpha_on = R_on

            # Compute expected counts using piecewise analytical solutions
            u_expected, s_expected = self._compute_piecewise_solution(
                t_star, alpha_off, alpha_on, gamma_star, t_on_star, delta_star
            )

            # Update context with expected counts
            context["u_expected"] = u_expected
            context["s_expected"] = s_expected

            # Create latent variables ut and st using pyro.deterministic
            import pyro

            # Create latent variables with proper shape handling
            if u_expected.dim() == 2 and u_expected.shape[0] > 1:
                # Add an extra dimension to match legacy shape (num_cells, 1, n_genes)
                u_expected_reshaped = u_expected.unsqueeze(1)
                s_expected_reshaped = s_expected.unsqueeze(1)
            else:
                u_expected_reshaped = u_expected
                s_expected_reshaped = s_expected

            # Create latent variables
            ut = pyro.deterministic("ut", u_expected_reshaped)
            st = pyro.deterministic("st", s_expected_reshaped)

            # Apply ReLU and add small constant for numerical stability
            one = torch.ones_like(ut) * 1e-6
            ut = torch.relu(ut) + one
            st = torch.relu(st) + one

            # Add latent variables to context
            context["ut"] = ut
            context["st"] = st

            return context
        else:
            # If validation failed, raise an error
            raise ValueError(f"Error in piecewise dynamics model forward pass: {validation_result.error}")

    @jaxtyped
    @beartype
    def _compute_piecewise_solution(
        self,
        t_star: BatchTensor,
        alpha_off: ParamTensor,
        alpha_on: ParamTensor,
        gamma_star: ParamTensor,
        t_on_star: ParamTensor,
        delta_star: ParamTensor,
    ) -> Tuple[BatchTensor, BatchTensor]:
        """
        Compute the piecewise analytical solution for dimensionless RNA dynamics.

        This method implements the three-phase analytical solution with steady-state
        initial conditions. The phases are:

        1. Off phase: 0 ≤ t* < t*_on
        2. On phase: t*_on ≤ t* < t*_on + δ*
        3. Return to off phase: t* ≥ t*_on + δ*

        Args:
            t_star: Dimensionless time points [cells] or [cells, genes]
            alpha_off: Basal transcription rate [genes]
            alpha_on: Active transcription rate [genes]
            gamma_star: Relative degradation rate [genes]
            t_on_star: Activation onset time [genes]
            delta_star: Activation duration [genes]

        Returns:
            Tuple of (u_star, s_star) dimensionless concentrations
        """
        # Ensure proper broadcasting shapes
        # t_star: [cells] or [cells, genes]
        # parameters: [genes]

        if t_star.dim() == 1:
            # t_star is [cells], expand to [cells, genes]
            n_genes = alpha_off.shape[0]
            t_star = t_star.unsqueeze(-1).expand(-1, n_genes)  # [cells, genes]

        # Broadcast parameters to match t_star shape
        # All parameters should be [genes] -> [1, genes] for broadcasting
        alpha_off = alpha_off.unsqueeze(0).expand_as(t_star)  # [cells, genes]
        alpha_on = alpha_on.unsqueeze(0).expand_as(t_star)    # [cells, genes]
        gamma_star = gamma_star.unsqueeze(0).expand_as(t_star)  # [cells, genes]
        t_on_star = t_on_star.unsqueeze(0).expand_as(t_star)   # [cells, genes]
        delta_star = delta_star.unsqueeze(0).expand_as(t_star)  # [cells, genes]

        # Initialize output tensors
        u_star = torch.zeros_like(t_star)  # [cells, genes]
        s_star = torch.zeros_like(t_star)  # [cells, genes]

        # Phase 1: Off state (t* < t*_on)
        # System is at steady state with α*_off = 1.0 (fixed reference)
        phase1_mask = t_star < t_on_star
        u_star[phase1_mask] = 1.0  # Fixed reference state
        s_star[phase1_mask] = (1.0 / gamma_star)[phase1_mask]

        # Phase 2: On state (t*_on ≤ t* < t*_on + δ*)
        phase2_mask = (t_star >= t_on_star) & (t_star < t_on_star + delta_star)
        if phase2_mask.any():
            tau_on = t_star - t_on_star  # Time since activation onset
            u_star[phase2_mask], s_star[phase2_mask] = self._compute_on_phase(
                tau_on[phase2_mask],
                alpha_off[phase2_mask],
                alpha_on[phase2_mask],
                gamma_star[phase2_mask],
            )

        # Phase 3: Return to off state (t* ≥ t*_on + δ*)
        phase3_mask = t_star >= t_on_star + delta_star
        if phase3_mask.any():
            tau_off = t_star - (t_on_star + delta_star)  # Time since deactivation
            u_star[phase3_mask], s_star[phase3_mask] = self._compute_off_phase(
                tau_off[phase3_mask],
                alpha_off[phase3_mask],
                alpha_on[phase3_mask],
                gamma_star[phase3_mask],
                delta_star[phase3_mask],
            )

        return u_star, s_star

    @jaxtyped
    @beartype
    def _compute_on_phase(
        self,
        tau_on: torch.Tensor,
        alpha_off: torch.Tensor,
        alpha_on: torch.Tensor,
        gamma_star: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute analytical solution for the ON phase with corrected parameterization.

        Phase 2: t*_on ≤ t* < t*_on + δ*
        Initial conditions: u*_0 = 1.0, s*_0 = 1.0/γ* (fixed)
        Transcription rate: α*_on = R_on (fold-change from reference)

        Args:
            tau_on: Time since activation onset (τ_on = t* - t*_on)
            alpha_off: Fixed reference transcription rate (always 1.0)
            alpha_on: Active transcription rate (R_on fold-change)
            gamma_star: Relative degradation rate

        Returns:
            Tuple of (u_star, s_star) for the ON phase
        """
        # u* solution: u*(τ) = α*_on + (1.0 - α*_on) * exp(-τ)
        # Since alpha_off = 1.0 (fixed reference)
        u_star = alpha_on + (1.0 - alpha_on) * torch.exp(-tau_on)

        # s* solution depends on whether γ* = 1 or γ* ≠ 1
        gamma_near_one = torch.abs(gamma_star - 1.0) < self.eps

        # For γ* ≠ 1 case (using alpha_off = 1.0)
        xi_on = (1.0 - alpha_on) / (gamma_star - 1.0)
        s_star_general = (
            alpha_on / gamma_star +
            (1.0 / gamma_star - xi_on - alpha_on / gamma_star) * torch.exp(-gamma_star * tau_on) +
            xi_on * torch.exp(-tau_on)
        )

        # For γ* = 1 case (special case with τe^(-τ) term, using alpha_off = 1.0)
        s_star_special = (
            alpha_on +
            (1.0 - alpha_on) * torch.exp(-tau_on) +
            (1.0 - alpha_on) * tau_on * torch.exp(-tau_on)
        )

        # Select appropriate solution based on γ* value
        s_star = torch.where(gamma_near_one, s_star_special, s_star_general)

        return u_star, s_star

    @jaxtyped
    @beartype
    def _compute_off_phase(
        self,
        tau_off: torch.Tensor,
        alpha_off: torch.Tensor,
        alpha_on: torch.Tensor,
        gamma_star: torch.Tensor,
        delta_star: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute analytical solution for the return to OFF phase with corrected parameterization.

        Phase 3: t* ≥ t*_on + δ*
        Initial conditions: endpoint values from Phase 2
        Transcription rate: α*_off = 1.0 (fixed reference)

        Args:
            tau_off: Time since deactivation (τ_off = t* - (t*_on + δ*))
            alpha_off: Fixed reference transcription rate (always 1.0)
            alpha_on: Active transcription rate (R_on fold-change)
            gamma_star: Relative degradation rate
            delta_star: Activation duration

        Returns:
            Tuple of (u_star, s_star) for the return to OFF phase
        """
        # Compute initial conditions for Phase 3 (endpoint values from Phase 2)
        u_off_0, s_off_0 = self._compute_phase2_endpoints(
            alpha_off, alpha_on, gamma_star, delta_star
        )

        # u* solution: u*(τ) = 1.0 + (u*_off,0 - 1.0) * exp(-τ)
        # Since alpha_off = 1.0 (fixed reference)
        u_star = 1.0 + (u_off_0 - 1.0) * torch.exp(-tau_off)

        # s* solution depends on whether γ* = 1 or γ* ≠ 1
        gamma_near_one = torch.abs(gamma_star - 1.0) < self.eps

        # For γ* ≠ 1 case (using alpha_off = 1.0)
        xi_off = (u_off_0 - 1.0) / (gamma_star - 1.0)
        s_star_general = (
            1.0 / gamma_star +
            (s_off_0 - xi_off - 1.0 / gamma_star) * torch.exp(-gamma_star * tau_off) +
            xi_off * torch.exp(-tau_off)
        )

        # For γ* = 1 case (special case with τe^(-τ) term, using alpha_off = 1.0)
        s_star_special = (
            1.0 +
            (s_off_0 - 1.0) * torch.exp(-tau_off) +
            (u_off_0 - 1.0) * tau_off * torch.exp(-tau_off)
        )

        # Select appropriate solution based on γ* value
        s_star = torch.where(gamma_near_one, s_star_special, s_star_general)

        return u_star, s_star

    @jaxtyped
    @beartype
    def _compute_phase2_endpoints(
        self,
        alpha_off: torch.Tensor,
        alpha_on: torch.Tensor,
        gamma_star: torch.Tensor,
        delta_star: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the endpoint values of Phase 2 (ON phase) to use as initial conditions for Phase 3.

        These are the values of u* and s* at t* = t*_on + δ*, which become the initial
        conditions for the return to OFF phase.

        Args:
            alpha_off: Basal transcription rate
            alpha_on: Active transcription rate
            gamma_star: Relative degradation rate
            delta_star: Activation duration

        Returns:
            Tuple of (u_off_0, s_off_0) endpoint values from Phase 2
        """
        # Use the ON phase solution evaluated at τ = δ*
        u_off_0, s_off_0 = self._compute_on_phase(
            delta_star, alpha_off, alpha_on, gamma_star
        )
        return u_off_0, s_off_0

    @jaxtyped
    @beartype
    def steady_state(
        self,
        alpha_off: Union[ParamTensor, torch.Tensor],
        gamma_star: Union[ParamTensor, torch.Tensor],
        **kwargs: Any,
    ) -> Tuple[
        Union[ParamTensor, torch.Tensor],
        Union[ParamTensor, torch.Tensor],
    ]:
        """
        Compute the steady-state unspliced and spliced RNA counts for the OFF phase.

        For the corrected piecewise activation model, the steady state corresponds to the
        OFF phase with fixed reference transcription α*_off = 1.0.

        Args:
            alpha_off: Fixed reference transcription rate (always 1.0)
            gamma_star: Relative degradation rate
            **kwargs: Additional model-specific parameters

        Returns:
            Tuple of (steady-state unspliced counts, steady-state spliced counts)
        """
        # At steady state in OFF phase:
        # du*/dt* = 1.0 - u* = 0 => u* = 1.0 (fixed reference)
        u_ss = torch.ones_like(alpha_off)  # Always 1.0

        # ds*/dt* = u* - γ*s* = 0 => s* = u*/γ* = 1.0/γ*
        s_ss = torch.ones_like(alpha_off) / gamma_star

        return u_ss, s_ss

    @beartype
    def compute_velocity(
        self,
        ut: Union[LatentCountTensor, torch.Tensor],
        st: Union[LatentCountTensor, torch.Tensor],
        gamma_star: Union[KineticParamTensor, torch.Tensor],
        **kwargs: Any,
    ) -> Union[VelocityTensor, torch.Tensor]:
        """
        Compute RNA velocity from latent RNA counts and kinetic parameters.

        For the dimensionless piecewise activation model:
            ds*/dt* = u* - γ*s*

        Args:
            ut: Latent unspliced RNA counts (dimensionless)
            st: Latent spliced RNA counts (dimensionless)
            gamma_star: Relative degradation rate
            **kwargs: Additional model-specific parameters

        Returns:
            RNA velocity tensor with same shape as ut/st
        """
        # For the dimensionless piecewise model: ds*/dt* = u* - γ*s*
        velocity = ut - gamma_star * st
        return velocity