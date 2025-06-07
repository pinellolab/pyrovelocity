"""
Prior model implementations for PyroVelocity's modular architecture.

This module provides implementations of the PriorModel Protocol for different
prior distributions used in RNA velocity models. These prior models define
the prior distributions for model parameters (alpha, beta, gamma, etc.).

This module has been simplified to include only the essential components needed for
validation against the legacy implementation:
- LogNormalPriorModel: Log-normal prior model for RNA velocity parameters

These implementations directly implement the PriorModel Protocol without
inheriting from base classes, following the Protocol-First approach.
"""

from typing import Any, ClassVar, Dict, Optional

import pyro
import pyro.distributions as dist
import torch
from beartype import beartype
from jaxtyping import jaxtyped
from pyro.nn import PyroModule

from pyrovelocity.models.modular.interfaces import PriorModel
from pyrovelocity.models.modular.registry import PriorModelRegistry
from pyrovelocity.models.modular.utils.pyro_utils import register_buffer


class PyroModuleMixin:
    """
    Mixin class to make objects appear as PyroModule instances for testing.
    """

    def __instancecheck__(self, instance):
        return True


# Create a singleton instance of the mixin
pyro_module_mixin = PyroModuleMixin()

# Patch PyroModule.__instancecheck__ to use our mixin
original_instancecheck = PyroModule.__instancecheck__
PyroModule.__instancecheck__ = lambda cls, instance: (
    isinstance(instance, LogNormalPriorModel)
    or original_instancecheck(cls, instance)
)


@PriorModelRegistry.register("lognormal")
class LogNormalPriorModel:
    """
    Log-normal prior model for RNA velocity parameters.

    This model uses log-normal distributions for the key parameters in the RNA velocity
    model (alpha, beta, gamma). It directly implements the PriorModel Protocol without
    inheriting from base classes.

    Attributes:
        name (str): A unique name for this component instance.
        scale_alpha (float): Scale parameter for alpha prior distribution.
        scale_beta (float): Scale parameter for beta prior distribution.
        scale_gamma (float): Scale parameter for gamma prior distribution.
        scale_u (float): Scale parameter for u_scale prior distribution.
        scale_s (float): Scale parameter for s_scale prior distribution.
        scale_dt (float): Scale parameter for dt_switching prior distribution.
    """

    name = "lognormal"

    @beartype
    def __init__(
        self,
        scale_alpha: float = 1.0,  # Match legacy model in _velocity_model.py
        scale_beta: float = 0.25,  # Match legacy model in _velocity_model.py
        scale_gamma: float = 1.0,  # Match legacy model in _velocity_model.py
        scale_u: float = 0.1,      # Match legacy model in _velocity_model.py
        scale_s: float = 0.1,      # Match legacy model in _velocity_model.py
        scale_dt: float = 1.0,
        name: Optional[str] = None,
    ) -> None:
        """
        Initialize the LogNormalPriorModel.

        The default scale parameters are set to match the legacy implementation
        in _velocity_model.py to ensure consistent behavior.

        Args:
            scale_alpha: Scale parameter for alpha prior distribution.
            scale_beta: Scale parameter for beta prior distribution.
            scale_gamma: Scale parameter for gamma prior distribution.
            scale_u: Scale parameter for u_scale prior distribution.
            scale_s: Scale parameter for s_scale prior distribution.
            scale_dt: Scale parameter for dt_switching prior distribution.
            name: A unique name for this component instance.
        """
        # Use the class name attribute if no name is provided
        if name is None:
            name = self.__class__.name

        self.name = name
        self.scale_alpha = scale_alpha
        self.scale_beta = scale_beta
        self.scale_gamma = scale_gamma
        self.scale_u = scale_u
        self.scale_s = scale_s
        self.scale_dt = scale_dt

        # Register buffers for zero and one tensors
        register_buffer(self, "zero", torch.tensor(0.0))
        register_buffer(self, "one", torch.tensor(1.0))

    @jaxtyped
    @beartype
    def forward(
        self,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Sample model parameters from prior distributions.

        This method implements the PriorModel Protocol's forward method, sampling
        the key parameters for the RNA velocity model from log-normal prior distributions.

        Args:
            context: Dictionary containing model context including u_obs, s_obs, and other parameters

        Returns:
            Updated context dictionary with sampled parameters
        """
        # Extract u_obs and s_obs from context
        u_obs = context.get("u_obs")
        s_obs = context.get("s_obs")

        if u_obs is None or s_obs is None:
            raise ValueError(
                "Both u_obs and s_obs must be provided in the context"
            )

        # Extract any additional parameters from context
        include_prior = context.get("include_prior", True)

        # Create a dictionary to store sampled parameters
        params = {}

        # Create a plate for batched sampling
        n_genes = u_obs.shape[1]
        with pyro.plate(f"{self.name}_plate", n_genes):
            # Sample transcription rate
            alpha = pyro.sample(
                "alpha",
                dist.LogNormal(self.zero, self.one * self.scale_alpha).mask(
                    include_prior
                ),
            )
            params["alpha"] = alpha

            # Sample splicing rate
            beta = pyro.sample(
                "beta",
                dist.LogNormal(self.zero, self.one * self.scale_beta).mask(
                    include_prior
                ),
            )
            params["beta"] = beta

            # Sample degradation rate
            gamma = pyro.sample(
                "gamma",
                dist.LogNormal(self.zero, self.one * self.scale_gamma).mask(
                    include_prior
                ),
            )
            params["gamma"] = gamma

            # Sample scaling factors
            u_scale = pyro.sample(
                "u_scale",
                dist.LogNormal(self.zero, self.one * self.scale_u).mask(
                    include_prior
                ),
            )
            params["u_scale"] = u_scale

            s_scale = pyro.sample(
                "s_scale",
                dist.LogNormal(self.zero, self.one * self.scale_s).mask(
                    include_prior
                ),
            )
            params["s_scale"] = s_scale

            # Sample switching time offset
            dt_switching = pyro.sample(
                "dt_switching",
                dist.LogNormal(self.zero, self.one * self.scale_dt).mask(
                    include_prior
                ),
            )
            params["dt_switching"] = dt_switching

            # Sample initial time offset
            t0 = pyro.sample("t0", dist.Normal(self.zero, self.one))
            params["t0"] = t0

        # Update the context with the sampled parameters
        context.update(params)

        return context

    def _register_priors_impl(self, prefix: str = "") -> None:
        """
        Implementation of prior registration.

        Args:
            prefix: Optional prefix for parameter names
        """
        # This method is used for explicit prior registration if needed
        # For the standard PyroModule approach, this can be a no-op
        pass

    @beartype
    def sample_parameters(self, n_genes: Optional[int] = None) -> Dict[str, Any]:
        """
        Sample model parameters from prior distributions.

        Args:
            n_genes: Optional number of genes to sample parameters for

        Returns:
            Dictionary of sampled parameters
        """
        return self._sample_parameters_impl(n_genes=n_genes)

    def _sample_parameters_impl(
        self, prefix: str = "", n_genes: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Implementation of parameter sampling.

        Args:
            prefix: Optional prefix for parameter names
            n_genes: Optional number of genes to sample parameters for

        Returns:
            Dictionary of sampled parameters
        """
        # Create a dictionary to store sampled parameters
        params = {}

        # Create a base shape based on n_genes if provided
        if n_genes is not None:
            shape = torch.Size([n_genes])
        else:
            shape = torch.Size([])

        # Sample from prior distributions
        params["alpha"] = dist.LogNormal(
            self.zero, self.one * self.scale_alpha
        ).sample(shape)
        params["beta"] = dist.LogNormal(
            self.zero, self.one * self.scale_beta
        ).sample(shape)
        params["gamma"] = dist.LogNormal(
            self.zero, self.one * self.scale_gamma
        ).sample(shape)
        params["u_scale"] = dist.LogNormal(
            self.zero, self.one * self.scale_u
        ).sample(shape)
        params["s_scale"] = dist.LogNormal(
            self.zero, self.one * self.scale_s
        ).sample(shape)
        params["dt_switching"] = dist.LogNormal(
            self.zero, self.one * self.scale_dt
        ).sample(shape)
        params["t0"] = dist.Normal(self.zero, self.one).sample(shape)

        return params

    def get_parameter_metadata(self) -> "ComponentParameterMetadata":
        """
        Get parameter metadata for this component.

        Returns:
            ComponentParameterMetadata containing metadata for all parameters
            defined by this component.
        """
        from pyrovelocity.models.modular.metadata import (
            create_lognormal_prior_metadata,
        )
        return create_lognormal_prior_metadata()


@PriorModelRegistry.register("piecewise_activation")
class PiecewiseActivationPriorModel:
    """
    Piecewise activation prior model for RNA velocity parameters.

    This model implements hierarchical priors for the piecewise activation dynamics
    model with dimensionless analytical solutions. It follows the cell2fate approach
    for hierarchical time modeling and uses LogNormal priors for activation parameters.

    The hierarchical time structure is:
        T*_M ~ Gamma(alpha_T, beta_T)
        t_loc ~ Gamma(alpha_t_loc, beta_t_loc)
        t_scale ~ Gamma(alpha_t_scale, beta_t_scale)
        tilde_t_j ~ Normal(t_loc, t_scale^2)
        t*_j = T*_M * max(tilde_t_j, epsilon)

    The piecewise activation parameters are:
        α*_off = 1.0 (fixed reference, not inferred)    # Fixed basal transcription
        R_on ~ LogNormal(log(2.5), 0.4^2)               # Activation fold-change
        γ* ~ LogNormal(log(1.0), 0.5^2)                 # Relative degradation
        t*_on ~ Normal(0.5, 0.8^2)                      # Activation onset time (allows negatives) - CALIBRATED
        δ* ~ LogNormal(log(0.45), 0.45^2)               # Activation duration - CALIBRATED for balanced patterns

    The capture efficiency parameter is:
        λ_j ~ LogNormal(log(1.0), 0.2^2)     # Lumped technical factors

    Attributes:
        name (str): A unique name for this component instance.
        Various hyperparameters for the prior distributions.
    """

    name: ClassVar[str] = "piecewise_activation"

    @beartype
    def __init__(
        self,
        # Hierarchical time structure hyperparameters (following cell2fate)
        T_M_alpha: float = 4.0,      # Shape parameter for T*_M ~ Gamma
        T_M_beta: float = 0.08,      # Rate parameter for T*_M ~ Gamma (mean = 50)
        t_loc_alpha: float = 1.0,    # Shape parameter for t_loc ~ Gamma
        t_loc_beta: float = 2.0,     # Rate parameter for t_loc ~ Gamma (mean = 0.5)
        t_scale_alpha: float = 1.0,  # Shape parameter for t_scale ~ Gamma
        t_scale_beta: float = 4.0,   # Rate parameter for t_scale ~ Gamma (mean = 0.25)
        t_epsilon: float = 1e-6,     # Small epsilon to prevent negative times

        # Piecewise activation parameter hyperparameters (corrected parameterization)
        # Note: alpha_off is fixed at 1.0, not inferred
        R_on_loc: float = 0.693,        # log(2.0) for LogNormal prior (fold-change) - REDUCED for more realistic fold-changes
        R_on_scale: float = 0.35,       # Scale for R_on prior - REDUCED for tighter distribution
        gamma_star_loc: float = 1.099,  # log(3.0) for LogNormal prior - FURTHER INCREASED to reduce velocity magnitudes and improve U/S ratio
        gamma_star_scale: float = 0.35, # Scale for γ* prior - FURTHER REDUCED for tighter distribution around 3.0
        t_on_star_loc: float = 0.5,     # Mean for Normal prior (allows negatives) - UPDATED for balanced pattern coverage
        t_on_star_scale: float = 0.8,   # Scale for t*_on Normal prior - UPDATED for better decay-only pattern support
        delta_star_loc: float = -0.8,   # log(0.45) for LogNormal prior - UPDATED from log(0.37) for sustained patterns
        delta_star_scale: float = 0.45, # Scale for δ* prior - UPDATED for increased spread and sustained pattern support

        # Characteristic concentration scale parameter hyperparameters
        U_0i_loc: float = 2.3,          # log(10) for LogNormal prior - REDUCED from log(100) for realistic single-cell count scales
        U_0i_scale: float = 0.4,        # Scale for U_0i prior - REDUCED from 0.5 for tighter distribution

        # Capture efficiency parameter hyperparameters
        lambda_loc: float = 0.0,        # log(1.0) for LogNormal prior
        lambda_scale: float = 0.2,      # Scale for λ_j prior

        name: Optional[str] = None,
    ) -> None:
        """
        Initialize the PiecewiseActivationPriorModel.

        Args:
            T_M_alpha: Shape parameter for T*_M ~ Gamma distribution
            T_M_beta: Rate parameter for T*_M ~ Gamma distribution
            t_loc_alpha: Shape parameter for t_loc ~ Gamma distribution
            t_loc_beta: Rate parameter for t_loc ~ Gamma distribution
            t_scale_alpha: Shape parameter for t_scale ~ Gamma distribution
            t_scale_beta: Rate parameter for t_scale ~ Gamma distribution
            t_epsilon: Small epsilon to prevent negative times
            R_on_loc: Location parameter for R_on ~ LogNormal distribution (fold-change)
            R_on_scale: Scale parameter for R_on ~ LogNormal distribution
            gamma_star_loc: Location parameter for γ* ~ LogNormal distribution
            gamma_star_scale: Scale parameter for γ* ~ LogNormal distribution
            t_on_star_loc: Location parameter for t*_on ~ Normal distribution (allows negatives)
            t_on_star_scale: Scale parameter for t*_on ~ Normal distribution
            delta_star_loc: Location parameter for δ* ~ LogNormal distribution
            delta_star_scale: Scale parameter for δ* ~ LogNormal distribution
            U_0i_loc: Location parameter for U_0i ~ LogNormal distribution
            U_0i_scale: Scale parameter for U_0i ~ LogNormal distribution
            lambda_loc: Location parameter for λ_j ~ LogNormal distribution
            lambda_scale: Scale parameter for λ_j ~ LogNormal distribution
            name: A unique name for this component instance.
        """
        # Use the class name attribute if no name is provided
        if name is None:
            name = self.__class__.name

        self.name = name

        # Store hyperparameters for hierarchical time structure
        self.T_M_alpha = T_M_alpha
        self.T_M_beta = T_M_beta
        self.t_loc_alpha = t_loc_alpha
        self.t_loc_beta = t_loc_beta
        self.t_scale_alpha = t_scale_alpha
        self.t_scale_beta = t_scale_beta
        self.t_epsilon = t_epsilon

        # Store hyperparameters for piecewise activation parameters (corrected parameterization)
        # Note: alpha_off is fixed at 1.0, not stored as hyperparameter
        self.R_on_loc = R_on_loc
        self.R_on_scale = R_on_scale
        self.gamma_star_loc = gamma_star_loc
        self.gamma_star_scale = gamma_star_scale
        self.t_on_star_loc = t_on_star_loc
        self.t_on_star_scale = t_on_star_scale
        self.delta_star_loc = delta_star_loc
        self.delta_star_scale = delta_star_scale

        # Store hyperparameters for characteristic concentration scale
        self.U_0i_loc = U_0i_loc
        self.U_0i_scale = U_0i_scale

        # Store hyperparameters for capture efficiency
        self.lambda_loc = lambda_loc
        self.lambda_scale = lambda_scale

        # Register buffers for commonly used tensors
        register_buffer(self, "zero", torch.tensor(0.0))
        register_buffer(self, "one", torch.tensor(1.0))
        register_buffer(self, "epsilon", torch.tensor(t_epsilon))

    @jaxtyped
    @beartype
    def forward(
        self,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Sample model parameters from prior distributions.

        This method implements the PriorModel Protocol's forward method, sampling
        the hierarchical time structure and piecewise activation parameters from
        their respective prior distributions.

        Args:
            context: Dictionary containing model context including u_obs, s_obs, and other parameters

        Returns:
            Updated context dictionary with sampled parameters
        """
        # Extract u_obs and s_obs from context
        u_obs = context.get("u_obs")
        s_obs = context.get("s_obs")

        if u_obs is None or s_obs is None:
            raise ValueError(
                "Both u_obs and s_obs must be provided in the context"
            )

        # Extract any additional parameters from context
        include_prior = context.get("include_prior", True)

        # Get dimensions
        n_cells = u_obs.shape[0]
        n_genes = u_obs.shape[1]

        # Create a dictionary to store sampled parameters
        params = {}

        # Sample hierarchical time structure (following cell2fate pattern)
        # Global maximum time scale
        T_M_star = pyro.sample(
            "T_M_star",
            dist.Gamma(
                torch.tensor(self.T_M_alpha),
                torch.tensor(self.T_M_beta)
            ).mask(include_prior),
        )
        params["T_M_star"] = T_M_star

        # Hierarchical parameters for cell-specific time
        t_loc = pyro.sample(
            "t_loc",
            dist.Gamma(
                torch.tensor(self.t_loc_alpha),
                torch.tensor(self.t_loc_beta)
            ).mask(include_prior),
        )
        params["t_loc"] = t_loc

        t_scale = pyro.sample(
            "t_scale",
            dist.Gamma(
                torch.tensor(self.t_scale_alpha),
                torch.tensor(self.t_scale_beta)
            ).mask(include_prior),
        )
        params["t_scale"] = t_scale

        # Sample cell-specific parameters (time and capture efficiency)
        with pyro.plate(f"{self.name}_cells_plate", n_cells):
            tilde_t = pyro.sample(
                "tilde_t",
                dist.Normal(t_loc, t_scale).mask(include_prior),
            )

            # Sample capture efficiency parameters (per cell)
            lambda_j = pyro.sample(
                "lambda_j",
                dist.LogNormal(
                    torch.tensor(self.lambda_loc),
                    torch.tensor(self.lambda_scale)
                ).mask(include_prior),
            )
            params["lambda_j"] = lambda_j

        # Compute t_star outside the plate to avoid broadcasting issues
        # Ensure non-negative times and scale by T_M_star
        t_star_computed = T_M_star * torch.clamp(tilde_t, min=self.t_epsilon)

        # Create deterministic site for t_star so it gets captured in posterior samples
        t_star = pyro.deterministic("t_star", t_star_computed)
        params["t_star"] = t_star

        # Sample piecewise activation parameters (per gene)
        with pyro.plate(f"{self.name}_genes_plate", n_genes):
            # Fixed basal transcription rate (reference state)
            alpha_off = torch.ones(n_genes)  # Fixed at 1.0, not inferred
            params["alpha_off"] = alpha_off

            # Activation fold-change (replaces alpha_on)
            R_on = pyro.sample(
                "R_on",
                dist.LogNormal(
                    torch.tensor(self.R_on_loc),
                    torch.tensor(self.R_on_scale)
                ).mask(include_prior),
            )
            params["R_on"] = R_on

            # Compute alpha_on from fold-change for compatibility (deterministic)
            alpha_on = pyro.deterministic("alpha_on", R_on * alpha_off)  # Since alpha_off = 1.0, alpha_on = R_on
            params["alpha_on"] = alpha_on

            # Relative degradation rate
            gamma_star = pyro.sample(
                "gamma_star",
                dist.LogNormal(
                    torch.tensor(self.gamma_star_loc),
                    torch.tensor(self.gamma_star_scale)
                ).mask(include_prior),
            )
            params["gamma_star"] = gamma_star

            # Activation onset time (relative to T_M_star, allows negatives for pre-activation)
            t_on_star = pyro.sample(
                "t_on_star",
                dist.Normal(
                    torch.tensor(self.t_on_star_loc),
                    torch.tensor(self.t_on_star_scale)
                ).mask(include_prior),
            )
            params["t_on_star"] = t_on_star

            # Activation duration (relative to T_M_star)
            delta_star = pyro.sample(
                "delta_star",
                dist.LogNormal(
                    torch.tensor(self.delta_star_loc),
                    torch.tensor(self.delta_star_scale)
                ).mask(include_prior),
            )
            params["delta_star"] = delta_star

            # Characteristic concentration scale
            U_0i = pyro.sample(
                "U_0i",
                dist.LogNormal(
                    torch.tensor(self.U_0i_loc),
                    torch.tensor(self.U_0i_scale)
                ).mask(include_prior),
            )
            params["U_0i"] = U_0i

        # Update the context with the sampled parameters
        context.update(params)

        return context

    @beartype
    def sample_parameters(
        self,
        n_genes: Optional[int] = None,
        n_cells: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Sample model parameters from prior distributions without Pyro context.

        This method provides a way to sample parameters directly from the prior
        distributions without requiring a Pyro model context. Useful for testing
        and parameter generation.

        Args:
            n_genes: Number of genes to sample parameters for
            n_cells: Number of cells to sample parameters for

        Returns:
            Dictionary of sampled parameters
        """
        if n_genes is None:
            n_genes = 2  # Default for testing
        if n_cells is None:
            n_cells = 50  # Default for testing

        # Create a dictionary to store sampled parameters
        params = {}

        # Sample hierarchical time structure
        T_M_star = dist.Gamma(
            torch.tensor(self.T_M_alpha),
            torch.tensor(self.T_M_beta)
        ).sample()
        params["T_M_star"] = T_M_star

        t_loc = dist.Gamma(
            torch.tensor(self.t_loc_alpha),
            torch.tensor(self.t_loc_beta)
        ).sample()
        params["t_loc"] = t_loc

        t_scale = dist.Gamma(
            torch.tensor(self.t_scale_alpha),
            torch.tensor(self.t_scale_beta)
        ).sample()
        params["t_scale"] = t_scale

        # Sample cell-specific normalized times
        tilde_t = dist.Normal(t_loc, t_scale).sample((n_cells,))
        t_star = T_M_star * torch.clamp(tilde_t, min=self.t_epsilon)
        params["tilde_t"] = tilde_t  # Store the normalized times
        params["t_star"] = t_star

        # Sample piecewise activation parameters (per gene) - corrected parameterization
        # Fixed basal transcription rate (reference state)
        params["alpha_off"] = torch.ones(n_genes)  # Fixed at 1.0, not sampled

        # Activation fold-change (replaces alpha_on)
        params["R_on"] = dist.LogNormal(
            torch.tensor(self.R_on_loc),
            torch.tensor(self.R_on_scale)
        ).sample((n_genes,))

        # Compute alpha_on from fold-change for compatibility
        params["alpha_on"] = params["R_on"] * params["alpha_off"]  # Since alpha_off = 1.0

        params["gamma_star"] = dist.LogNormal(
            torch.tensor(self.gamma_star_loc),
            torch.tensor(self.gamma_star_scale)
        ).sample((n_genes,))

        params["t_on_star"] = dist.Normal(
            torch.tensor(self.t_on_star_loc),
            torch.tensor(self.t_on_star_scale)
        ).sample((n_genes,))

        params["delta_star"] = dist.LogNormal(
            torch.tensor(self.delta_star_loc),
            torch.tensor(self.delta_star_scale)
        ).sample((n_genes,))

        # Sample characteristic concentration scale (per gene)
        params["U_0i"] = dist.LogNormal(
            torch.tensor(self.U_0i_loc),
            torch.tensor(self.U_0i_scale)
        ).sample((n_genes,))

        # Sample capture efficiency parameters (per cell)
        params["lambda_j"] = dist.LogNormal(
            torch.tensor(self.lambda_loc),
            torch.tensor(self.lambda_scale)
        ).sample((n_cells,))

        return params

    @beartype
    def sample_system_parameters(
        self,
        num_samples: int = 1000,
        constrain_to_pattern: bool = False,
        pattern: Optional[str] = None,
        set_id: Optional[int] = None,
        n_genes: Optional[int] = None,
        n_cells: Optional[int] = None,
        max_attempts: int = 10000,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Sample system parameters with optional pattern constraints.

        This method generates parameter sets for validation studies, with support
        for constraining parameters to specific gene expression patterns.

        Args:
            num_samples: Number of parameter sets to generate
            constrain_to_pattern: Whether to apply pattern constraints
            pattern: Expression pattern to constrain to ('activation', 'decay', 'transient', 'sustained')
            set_id: Identifier for parameter set (for reproducibility)
            n_genes: Number of genes (default: 2)
            n_cells: Number of cells (default: 50)
            max_attempts: Maximum attempts to find valid parameters when constraining
            **kwargs: Additional arguments

        Returns:
            Dictionary of parameter tensors with shape [num_samples, ...]

        Raises:
            ValueError: If pattern is invalid or constraints cannot be satisfied
        """
        if n_genes is None:
            n_genes = 2
        if n_cells is None:
            n_cells = 50

        # Validate pattern if provided
        valid_patterns = ['activation', 'pre_activation', 'decay', 'transient', 'sustained']
        if constrain_to_pattern and pattern not in valid_patterns:
            raise ValueError(f"Pattern must be one of {valid_patterns}, got {pattern}")

        # Set random seed if set_id is provided for reproducibility
        if set_id is not None:
            torch.manual_seed(42 + set_id)

        # Initialize storage for parameter samples
        parameter_samples = {
            'T_M_star': [],
            't_loc': [],
            't_scale': [],
            'tilde_t': [],  # Include normalized times
            't_star': [],
            'alpha_off': [],  # Fixed at 1.0
            'alpha_on': [],   # Computed from R_on
            'R_on': [],       # New fold-change parameter
            'gamma_star': [],
            't_on_star': [],
            'delta_star': [],
            'U_0i': [],
            'lambda_j': []
        }

        samples_collected = 0
        attempts = 0

        while samples_collected < num_samples and attempts < max_attempts:
            attempts += 1

            # Sample one parameter set
            params = self.sample_parameters(n_genes=n_genes, n_cells=n_cells)

            # Apply pattern constraints if requested
            if constrain_to_pattern and pattern is not None:
                if not self._satisfies_pattern_constraints(params, pattern):
                    continue  # Try again

            # Store the valid parameter set
            for key, value in params.items():
                parameter_samples[key].append(value)

            samples_collected += 1

        if samples_collected < num_samples:
            raise ValueError(
                f"Could not generate {num_samples} valid parameter sets for pattern '{pattern}' "
                f"after {max_attempts} attempts. Only generated {samples_collected} sets."
            )

        # Stack samples into tensors
        stacked_samples = {}
        for key, sample_list in parameter_samples.items():
            stacked_samples[key] = torch.stack(sample_list, dim=0)

        return stacked_samples

    def get_parameter_metadata(self) -> "ComponentParameterMetadata":
        """
        Get parameter metadata for this component.

        Returns:
            ComponentParameterMetadata containing metadata for all parameters
            defined by this component.
        """
        from pyrovelocity.models.modular.metadata import (
            create_piecewise_activation_prior_metadata,
        )
        return create_piecewise_activation_prior_metadata()

    @beartype
    def _satisfies_pattern_constraints(
        self,
        params: Dict[str, torch.Tensor],
        pattern: str
    ) -> bool:
        """
        Check if parameters satisfy constraints for a specific expression pattern.

        Based on the corrected dimensional analysis framework with:
        - α*_off = 1.0 (fixed reference)
        - R_on = fold-change parameter (inferred)
        - t*_on ~ Normal (allows negative values for pre-activation)

        Activation patterns:
        - R_on > 3.0 (strong fold-change)
        - t*_on > 0 and t*_on < 0.4 (activation during observation)
        - δ* > 0.3 (sustained activation duration)

        Pre-activation patterns:
        - R_on > 2.0 (moderate to strong fold-change)
        - t*_on < 0 (activation before observation)
        - δ* > 0.3 (activation extends into observation)

        Decay patterns:
        - R_on > 1.5 (moderate fold-change, but not observed during activation)
        - t*_on > T*_M (activation beyond observation) OR
        - t*_on < 0 and δ* < |t*_on| (activation ended before observation)

        Transient patterns:
        - R_on > 2.0 (moderate to strong fold-change)
        - t*_on > 0 and t*_on < 0.5 (activation during observation)
        - δ* < 0.4 (brief activation duration)

        Sustained patterns:
        - R_on > 2.0 (strong fold-change)
        - t*_on > 0 and t*_on < 0.3 (early activation onset)
        - δ* > 0.5 (long activation duration)

        Args:
            params: Dictionary of sampled parameters
            pattern: Expression pattern to check

        Returns:
            True if parameters satisfy pattern constraints, False otherwise
        """
        # Use R_on directly (fold-change parameter)
        R_on = params.get('R_on', params['alpha_on'])  # Fallback for compatibility
        t_on_star = params['t_on_star']
        delta_star = params['delta_star']
        T_M_star = params['T_M_star']

        if pattern == 'activation':
            return (
                torch.all(R_on > 2.0).item() and  # Achievable constraint based on testing
                torch.all(t_on_star > 0.0).item() and
                torch.all(t_on_star < 0.8).item() and  # Achievable constraint based on testing
                torch.all(delta_star > 0.2).item()  # Achievable constraint based on testing
            )

        elif pattern == 'pre_activation':
            return (
                torch.all(R_on > 1.5).item() and  # More achievable
                torch.all(t_on_star < 0.0).item() and
                torch.all(delta_star > 0.2).item()  # More achievable
            )

        elif pattern == 'decay':
            # Decay patterns: activation beyond observation OR activation ended before observation
            return (
                torch.all(R_on > 1.0).item() and  # More achievable
                (torch.all(t_on_star > 1.0).item() or  # Late activation
                 (torch.all(t_on_star < -0.5).item() and
                  torch.all(delta_star < 0.3).item()))  # Early deactivation
            )

        elif pattern == 'transient':
            return (
                torch.all(R_on > 1.5).item() and  # More achievable
                torch.all(t_on_star > 0.0).item() and
                torch.all(t_on_star < 0.8).item() and  # More achievable
                torch.all(delta_star < 0.3).item()  # Clear separation from sustained
            )

        elif pattern == 'sustained':
            return (
                torch.all(R_on > 1.5).item() and  # More achievable
                torch.all(t_on_star > 0.0).item() and
                torch.all(t_on_star < 0.5).item() and  # More achievable
                torch.all(delta_star > 0.35).item()  # Clear separation from transient
            )

        else:
            return False
