"""
Piecewise activation dynamics model for PyroVelocity's modular architecture.

This module implements dimensionless analytical dynamics with latent time and 
piecewise activation for parameter recovery validation. The model follows the
mathematical framework from the "inference in dynamical systems" document
with piecewise constant transcription rates.

The model implements three phases:
1. Off phase: constant basal transcription α*_off
2. On phase: elevated transcription α*_on  
3. Return to off phase: back to basal transcription α*_off

All solutions are analytical with special handling for the γ* = 1 boundary case.
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


@DynamicsModelRegistry.register("piecewise_activation")
class PiecewiseActivationDynamicsModel:
    """Piecewise activation dynamics model for RNA velocity using analytical solutions.

    This model implements dimensionless analytical dynamics with piecewise constant
    transcription rates. The system has three phases:

    Phase 1 (Off): 0 ≤ t* < t*_on
        α*(t*) = α*_off (basal transcription)
        
    Phase 2 (On): t*_on ≤ t* < t*_on + δ*  
        α*(t*) = α*_on (elevated transcription)
        
    Phase 3 (Return to Off): t* ≥ t*_on + δ*
        α*(t*) = α*_off (back to basal)

    The dimensionless system is:
        du*/dt* = α*(t*) - u*
        ds*/dt* = u* - γ*s*

    With steady-state initial conditions:
        u*_0 = α*_off, s*_0 = α*_off/γ*

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
        # Validate context
        validation_result = validate_context(
            self.__class__.__name__,
            context,
            required_keys=[
                "u_obs", "s_obs", "alpha_off", "alpha_on", "gamma_star",
                "t_on_star", "delta_star", "t_star"
            ],
            tensor_keys=[
                "u_obs", "s_obs", "alpha_off", "alpha_on", "gamma_star",
                "t_on_star", "delta_star", "t_star"
            ],
        )

        if isinstance(validation_result, dict):
            # Extract required values from context
            u_obs = context["u_obs"]
            s_obs = context["s_obs"]
            alpha_off = context["alpha_off"]
            alpha_on = context["alpha_on"]
            gamma_star = context["gamma_star"]
            t_on_star = context["t_on_star"]
            delta_star = context["delta_star"]
            t_star = context["t_star"]

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

        # Phase 1: Off state (0 ≤ t* < t*_on)
        # System is at steady state with α*_off
        phase1_mask = t_star < t_on_star
        u_star[phase1_mask] = alpha_off[phase1_mask]
        s_star[phase1_mask] = (alpha_off / gamma_star)[phase1_mask]

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
        Compute analytical solution for the ON phase.

        Phase 2: t*_on ≤ t* < t*_on + δ*
        Initial conditions: u*_0 = α*_off, s*_0 = α*_off/γ*
        Transcription rate: α*_on

        Args:
            tau_on: Time since activation onset (τ_on = t* - t*_on)
            alpha_off: Basal transcription rate
            alpha_on: Active transcription rate
            gamma_star: Relative degradation rate

        Returns:
            Tuple of (u_star, s_star) for the ON phase
        """
        # u* solution: u*(τ) = α*_on + (α*_off - α*_on) * exp(-τ)
        u_star = alpha_on + (alpha_off - alpha_on) * torch.exp(-tau_on)

        # s* solution depends on whether γ* = 1 or γ* ≠ 1
        gamma_near_one = torch.abs(gamma_star - 1.0) < self.eps

        # For γ* ≠ 1 case
        xi_on = (alpha_off - alpha_on) / (gamma_star - 1.0)
        s_star_general = (
            alpha_on / gamma_star +
            (alpha_off / gamma_star - xi_on - alpha_on / gamma_star) * torch.exp(-gamma_star * tau_on) +
            xi_on * torch.exp(-tau_on)
        )

        # For γ* = 1 case (special case with τe^(-τ) term)
        s_star_special = (
            alpha_on +
            (alpha_off - alpha_on) * torch.exp(-tau_on) +
            (alpha_off - alpha_on) * tau_on * torch.exp(-tau_on)
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
        Compute analytical solution for the return to OFF phase.

        Phase 3: t* ≥ t*_on + δ*
        Initial conditions: endpoint values from Phase 2
        Transcription rate: α*_off

        Args:
            tau_off: Time since deactivation (τ_off = t* - (t*_on + δ*))
            alpha_off: Basal transcription rate
            alpha_on: Active transcription rate
            gamma_star: Relative degradation rate
            delta_star: Activation duration

        Returns:
            Tuple of (u_star, s_star) for the return to OFF phase
        """
        # Compute initial conditions for Phase 3 (endpoint values from Phase 2)
        u_off_0, s_off_0 = self._compute_phase2_endpoints(
            alpha_off, alpha_on, gamma_star, delta_star
        )

        # u* solution: u*(τ) = α*_off + (u*_off,0 - α*_off) * exp(-τ)
        u_star = alpha_off + (u_off_0 - alpha_off) * torch.exp(-tau_off)

        # s* solution depends on whether γ* = 1 or γ* ≠ 1
        gamma_near_one = torch.abs(gamma_star - 1.0) < self.eps

        # For γ* ≠ 1 case
        xi_off = (u_off_0 - alpha_off) / (gamma_star - 1.0)
        s_star_general = (
            alpha_off / gamma_star +
            (s_off_0 - xi_off - alpha_off / gamma_star) * torch.exp(-gamma_star * tau_off) +
            xi_off * torch.exp(-tau_off)
        )

        # For γ* = 1 case (special case with τe^(-τ) term)
        s_star_special = (
            alpha_off +
            (s_off_0 - alpha_off) * torch.exp(-tau_off) +
            (u_off_0 - alpha_off) * tau_off * torch.exp(-tau_off)
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

        For the piecewise activation model, the steady state corresponds to the
        OFF phase with basal transcription α*_off.

        Args:
            alpha_off: Basal transcription rate
            gamma_star: Relative degradation rate
            **kwargs: Additional model-specific parameters

        Returns:
            Tuple of (steady-state unspliced counts, steady-state spliced counts)
        """
        # At steady state in OFF phase:
        # du*/dt* = α*_off - u* = 0 => u* = α*_off
        u_ss = alpha_off

        # ds*/dt* = u* - γ*s* = 0 => s* = u*/γ* = α*_off/γ*
        s_ss = alpha_off / gamma_star

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
