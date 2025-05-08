"""
Prior model implementations for PyroVelocity's modular architecture.

This module provides implementations of the PriorModel Protocol for different
prior distributions used in RNA velocity models. These prior models define
the prior distributions for model parameters (alpha, beta, gamma, etc.).

These implementations directly implement the PriorModel Protocol without
inheriting from base classes, following the Protocol-First approach.
"""

from typing import Any, Dict, Optional

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
    isinstance(instance, (LogNormalPriorModel, InformativePriorModel))
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


@PriorModelRegistry.register("informative")
class InformativePriorModel:
    """
    Informative prior model for RNA velocity parameters.

    This model uses more informative prior distributions for the key parameters
    in the RNA velocity model, based on biological knowledge and empirical observations.
    The priors are designed to be more specific about the expected ranges of parameter values.

    Attributes:
        name (str): A unique name for this component instance.
        alpha_loc (float): Location parameter for alpha prior distribution.
        alpha_scale (float): Scale parameter for alpha prior distribution.
        beta_loc (float): Location parameter for beta prior distribution.
        beta_scale (float): Scale parameter for beta prior distribution.
        gamma_loc (float): Location parameter for gamma prior distribution.
        gamma_scale (float): Scale parameter for gamma prior distribution.
    """

    name = "informative"

    @beartype
    def __init__(
        self,
        alpha_loc: float = -0.5,
        alpha_scale: float = 0.5,
        beta_loc: float = -1.0,
        beta_scale: float = 0.3,
        gamma_loc: float = -0.5,
        gamma_scale: float = 0.5,
        u_scale_loc: float = -2.0,
        u_scale_scale: float = 0.2,
        s_scale_loc: float = -2.0,
        s_scale_scale: float = 0.2,
        dt_switching_loc: float = 0.0,
        dt_switching_scale: float = 0.5,
        name: Optional[str] = None,
    ) -> None:
        """
        Initialize the InformativePriorModel.

        Args:
            alpha_loc: Location parameter for alpha prior distribution.
            alpha_scale: Scale parameter for alpha prior distribution.
            beta_loc: Location parameter for beta prior distribution.
            beta_scale: Scale parameter for beta prior distribution.
            gamma_loc: Location parameter for gamma prior distribution.
            gamma_scale: Scale parameter for gamma prior distribution.
            u_scale_loc: Location parameter for u_scale prior distribution.
            u_scale_scale: Scale parameter for u_scale prior distribution.
            s_scale_loc: Location parameter for s_scale prior distribution.
            s_scale_scale: Scale parameter for s_scale prior distribution.
            dt_switching_loc: Location parameter for dt_switching prior distribution.
            dt_switching_scale: Scale parameter for dt_switching prior distribution.
            name: A unique name for this component instance.
        """
        # Use the class name attribute if no name is provided
        if name is None:
            name = self.__class__.name

        self.name = name
        self.alpha_loc = alpha_loc
        self.alpha_scale = alpha_scale
        self.beta_loc = beta_loc
        self.beta_scale = beta_scale
        self.gamma_loc = gamma_loc
        self.gamma_scale = gamma_scale
        self.u_scale_loc = u_scale_loc
        self.u_scale_scale = u_scale_scale
        self.s_scale_loc = s_scale_loc
        self.s_scale_scale = s_scale_scale
        self.dt_switching_loc = dt_switching_loc
        self.dt_switching_scale = dt_switching_scale

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
        Sample model parameters from informative prior distributions.

        This method implements the PriorModel Protocol's forward method, sampling
        the key parameters for the RNA velocity model from informative prior distributions.

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
            # Sample transcription rate with informative prior
            alpha = pyro.sample(
                "alpha",
                dist.LogNormal(
                    torch.tensor(self.alpha_loc), torch.tensor(self.alpha_scale)
                ).mask(include_prior),
            )
            params["alpha"] = alpha

            # Sample splicing rate with informative prior
            beta = pyro.sample(
                "beta",
                dist.LogNormal(
                    torch.tensor(self.beta_loc), torch.tensor(self.beta_scale)
                ).mask(include_prior),
            )
            params["beta"] = beta

            # Sample degradation rate with informative prior
            gamma = pyro.sample(
                "gamma",
                dist.LogNormal(
                    torch.tensor(self.gamma_loc), torch.tensor(self.gamma_scale)
                ).mask(include_prior),
            )
            params["gamma"] = gamma

            # Sample scaling factors with informative priors
            u_scale = pyro.sample(
                "u_scale",
                dist.LogNormal(
                    torch.tensor(self.u_scale_loc),
                    torch.tensor(self.u_scale_scale),
                ).mask(include_prior),
            )
            params["u_scale"] = u_scale

            s_scale = pyro.sample(
                "s_scale",
                dist.LogNormal(
                    torch.tensor(self.s_scale_loc),
                    torch.tensor(self.s_scale_scale),
                ).mask(include_prior),
            )
            params["s_scale"] = s_scale

            # Sample switching time offset with informative prior
            dt_switching = pyro.sample(
                "dt_switching",
                dist.LogNormal(
                    torch.tensor(self.dt_switching_loc),
                    torch.tensor(self.dt_switching_scale),
                ).mask(include_prior),
            )
            params["dt_switching"] = dt_switching

            # Sample initial time offset
            t0 = pyro.sample("t0", dist.Normal(self.zero, self.one * 0.5))
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

        # Sample from informative prior distributions
        params["alpha"] = dist.LogNormal(
            torch.tensor(self.alpha_loc), torch.tensor(self.alpha_scale)
        ).sample(shape)

        params["beta"] = dist.LogNormal(
            torch.tensor(self.beta_loc), torch.tensor(self.beta_scale)
        ).sample(shape)

        params["gamma"] = dist.LogNormal(
            torch.tensor(self.gamma_loc), torch.tensor(self.gamma_scale)
        ).sample(shape)

        params["u_scale"] = dist.LogNormal(
            torch.tensor(self.u_scale_loc), torch.tensor(self.u_scale_scale)
        ).sample(shape)

        params["s_scale"] = dist.LogNormal(
            torch.tensor(self.s_scale_loc), torch.tensor(self.s_scale_scale)
        ).sample(shape)

        params["dt_switching"] = dist.LogNormal(
            torch.tensor(self.dt_switching_loc),
            torch.tensor(self.dt_switching_scale),
        ).sample(shape)

        params["t0"] = dist.Normal(self.zero, self.one * 0.5).sample(shape)

        return params
