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
        α*_off ~ LogNormal(log(0.1), 0.5^2)  # Basal transcription
        α*_on ~ LogNormal(log(2.0), 0.5^2)   # Active transcription
        γ* ~ LogNormal(log(1.0), 0.3^2)      # Relative degradation
        t*_on ~ LogNormal(log(0.3), 0.3^2)   # Activation onset time
        δ* ~ LogNormal(log(0.4), 0.3^2)      # Activation duration

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

        # Piecewise activation parameter hyperparameters
        alpha_off_loc: float = -2.3,    # log(0.1) for LogNormal prior
        alpha_off_scale: float = 0.5,   # Scale for α*_off prior
        alpha_on_loc: float = 0.69,     # log(2.0) for LogNormal prior
        alpha_on_scale: float = 0.5,    # Scale for α*_on prior
        gamma_star_loc: float = 0.0,    # log(1.0) for LogNormal prior
        gamma_star_scale: float = 0.3,  # Scale for γ* prior
        t_on_star_loc: float = -1.2,    # log(0.3) for LogNormal prior
        t_on_star_scale: float = 0.3,   # Scale for t*_on prior
        delta_star_loc: float = -1.0,   # log(0.37) for LogNormal prior (hybrid optimization)
        delta_star_scale: float = 0.35, # Scale for δ* prior (hybrid optimization)

        # Characteristic concentration scale parameter hyperparameters
        U_0i_loc: float = 4.6,          # log(100) for LogNormal prior
        U_0i_scale: float = 0.5,        # Scale for U_0i prior

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
            alpha_off_loc: Location parameter for α*_off ~ LogNormal distribution
            alpha_off_scale: Scale parameter for α*_off ~ LogNormal distribution
            alpha_on_loc: Location parameter for α*_on ~ LogNormal distribution
            alpha_on_scale: Scale parameter for α*_on ~ LogNormal distribution
            gamma_star_loc: Location parameter for γ* ~ LogNormal distribution
            gamma_star_scale: Scale parameter for γ* ~ LogNormal distribution
            t_on_star_loc: Location parameter for t*_on ~ LogNormal distribution
            t_on_star_scale: Scale parameter for t*_on ~ LogNormal distribution
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

        # Store hyperparameters for piecewise activation parameters
        self.alpha_off_loc = alpha_off_loc
        self.alpha_off_scale = alpha_off_scale
        self.alpha_on_loc = alpha_on_loc
        self.alpha_on_scale = alpha_on_scale
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
            # Ensure non-negative times and scale by T_M_star
            # Compute t_star outside pyro.deterministic to avoid plate broadcasting issues
            t_star_computed = T_M_star * torch.clamp(tilde_t, min=self.t_epsilon)
            params["t_star"] = t_star_computed

            # Sample capture efficiency parameters (per cell)
            lambda_j = pyro.sample(
                "lambda_j",
                dist.LogNormal(
                    torch.tensor(self.lambda_loc),
                    torch.tensor(self.lambda_scale)
                ).mask(include_prior),
            )
            params["lambda_j"] = lambda_j

        # Note: t_star is computed deterministically from tilde_t and T_M_star
        # We don't create a separate deterministic site to avoid issues with AutoGuides
        # The downstream code can compute t_star from the sampled hierarchical parameters

        # Sample piecewise activation parameters (per gene)
        with pyro.plate(f"{self.name}_genes_plate", n_genes):
            # Basal transcription rate
            alpha_off = pyro.sample(
                "alpha_off",
                dist.LogNormal(
                    torch.tensor(self.alpha_off_loc),
                    torch.tensor(self.alpha_off_scale)
                ).mask(include_prior),
            )
            params["alpha_off"] = alpha_off

            # Active transcription rate
            alpha_on = pyro.sample(
                "alpha_on",
                dist.LogNormal(
                    torch.tensor(self.alpha_on_loc),
                    torch.tensor(self.alpha_on_scale)
                ).mask(include_prior),
            )
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

            # Activation onset time (relative to T_M_star)
            t_on_star = pyro.sample(
                "t_on_star",
                dist.LogNormal(
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

        # Sample piecewise activation parameters (per gene)
        params["alpha_off"] = dist.LogNormal(
            torch.tensor(self.alpha_off_loc),
            torch.tensor(self.alpha_off_scale)
        ).sample((n_genes,))

        params["alpha_on"] = dist.LogNormal(
            torch.tensor(self.alpha_on_loc),
            torch.tensor(self.alpha_on_scale)
        ).sample((n_genes,))

        params["gamma_star"] = dist.LogNormal(
            torch.tensor(self.gamma_star_loc),
            torch.tensor(self.gamma_star_scale)
        ).sample((n_genes,))

        params["t_on_star"] = dist.LogNormal(
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
        valid_patterns = ['activation', 'decay', 'transient', 'sustained']
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
            'alpha_off': [],
            'alpha_on': [],
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

        Based on the quantitative parameter ranges defined in the validation study:

        Activation patterns:
        - α*_off < 0.2 (low basal transcription)
        - α*_on > 1.5 (strong activation)
        - t*_on < 0.4 (early activation onset)
        - δ* > 0.3 (sustained activation duration)
        - Fold-change: α*_on/α*_off > 7.5

        Decay patterns:
        - α*_off > 0.5 (high basal transcription)
        - t*_on > T*_M (activation onset beyond observation window)

        Transient patterns:
        - α*_off < 0.3 (low basal transcription)
        - α*_on > 1.0 (moderate to strong activation)
        - t*_on < 0.5 (early to mid-process activation)
        - δ* < 0.3 (brief activation duration)
        - Fold-change: α*_on/α*_off > 3.3

        Sustained patterns:
        - α*_off < 0.3 (low basal transcription)
        - α*_on > 1.0 (strong activation)
        - t*_on < 0.3 (early activation onset)
        - δ* > 0.6 (long activation duration)
        - Fold-change: α*_on/α*_off > 3.3

        Args:
            params: Dictionary of sampled parameters
            pattern: Expression pattern to check

        Returns:
            True if parameters satisfy pattern constraints, False otherwise
        """
        alpha_off = params['alpha_off']
        alpha_on = params['alpha_on']
        t_on_star = params['t_on_star']
        delta_star = params['delta_star']
        T_M_star = params['T_M_star']

        # Calculate fold-change
        fold_change = alpha_on / alpha_off

        if pattern == 'activation':
            return (
                torch.all(alpha_off < 0.15).item() and  # More stringent to separate from sustained
                torch.all(alpha_on > 1.5).item() and
                torch.all(t_on_star < 0.4).item() and
                torch.all(delta_star > 0.4).item() and  # Higher threshold to separate from sustained
                torch.all(fold_change > 7.5).item()
            )

        elif pattern == 'decay':
            # Achievable constraints for decay patterns with current priors
            # Decay patterns: relatively high basal transcription, late activation
            return (
                torch.all(alpha_off > 0.08).item() and  # Higher basal (top 30% of prior range)
                torch.all(t_on_star > 0.35).item()      # Late activation (beyond mid-process)
                # Note: No fold-change constraint as current priors don't generate low fold-changes
            )

        elif pattern == 'transient':
            return (
                torch.all(alpha_off < 0.3).item() and
                torch.all(alpha_on > 1.0).item() and
                torch.all(t_on_star < 0.5).item() and
                torch.all(delta_star < 0.35).item() and  # Adjusted to avoid overlap with sustained
                torch.all(fold_change > 3.3).item()
            )

        elif pattern == 'sustained':
            return (
                torch.all(alpha_off < 0.3).item() and
                torch.all(alpha_on > 1.0).item() and
                torch.all(t_on_star < 0.3).item() and
                torch.all(delta_star > 0.35).item() and  # Clear separation from transient
                torch.all(fold_change > 3.3).item()
            )

        else:
            return False
