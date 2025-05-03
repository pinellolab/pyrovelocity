"""
Protocol-First prior model implementations for PyroVelocity's modular architecture.

This module contains prior model implementations that directly implement the
PriorModel Protocol without inheriting from BasePriorModel. These implementations
follow the Protocol-First approach, which embraces composition over inheritance
and allows for more flexible component composition.
"""

from typing import Any, Dict, Optional

import pyro
import pyro.distributions as dist
import torch
from beartype import beartype
from jaxtyping import jaxtyped

from pyrovelocity.models.modular.interfaces import PriorModel
from pyrovelocity.models.modular.registry import PriorModelRegistry
from pyrovelocity.models.modular.utils.pyro_utils import register_buffer


@PriorModelRegistry.register("lognormal_direct")
class LogNormalPriorModelDirect:
    """
    Log-normal prior model for RNA velocity parameters.

    This model uses log-normal distributions for the key parameters in the RNA velocity
    model (alpha, beta, gamma). It directly implements the PriorModel Protocol without
    inheriting from BasePriorModel.

    Attributes:
        name (str): A unique name for this component instance.
        scale_alpha (float): Scale parameter for alpha prior distribution.
        scale_beta (float): Scale parameter for beta prior distribution.
        scale_gamma (float): Scale parameter for gamma prior distribution.
        scale_u (float): Scale parameter for u_scale prior distribution.
        scale_s (float): Scale parameter for s_scale prior distribution.
        scale_dt (float): Scale parameter for dt_switching prior distribution.
    """

    name = "lognormal_direct"

    @beartype
    def __init__(
        self,
        scale_alpha: float = 1.0,
        scale_beta: float = 0.25,
        scale_gamma: float = 1.0,
        scale_u: float = 0.1,
        scale_s: float = 0.1,
        scale_dt: float = 1.0,
        name: Optional[str] = None,
    ) -> None:
        """
        Initialize the LogNormalPriorModelDirect.

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
        
        # Intentional duplication: Register buffers for constants
        # This functionality is extracted to a utility function
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

        # Check if we should include priors (for guide execution)
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

    def sample_parameters(
        self, n_genes: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Sample parameters from prior distributions without using Pyro's sampling mechanism.

        This method is useful for testing and initialization.

        Args:
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
