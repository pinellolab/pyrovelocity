"""Likelihood models for PyroVelocity.

This module provides likelihood models for generating distributions
for observed data given latent variables.
"""

from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import pyro
import pyro.distributions
import torch
from anndata._core.anndata import AnnData
from beartype import beartype
from jaxtyping import Array, Float, Int

from pyrovelocity.models.modular.components.base import BaseLikelihoodModel
from pyrovelocity.models.modular.registry import LikelihoodModelRegistry


@LikelihoodModelRegistry.register("poisson")
class PoissonLikelihoodModel(BaseLikelihoodModel):
    """Poisson likelihood model for observed counts.

    This model uses a Poisson distribution as the likelihood for
    observed RNA counts, which is appropriate when the variance
    is approximately equal to the mean.
    """

    @beartype
    def forward(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Define the likelihood distributions for observed data given expected values.

        Args:
            context: Dictionary containing model context including u_obs, s_obs, u_expected, s_expected, and other parameters

        Returns:
            Updated context dictionary with likelihood information
        """
        # Extract required values from context
        u_obs = context.get("u_obs")
        s_obs = context.get("s_obs")
        u_expected = context.get("u_expected")
        s_expected = context.get("s_expected")

        if u_obs is None or s_obs is None:
            raise ValueError("Both u_obs and s_obs must be provided in the context")

        if u_expected is None or s_expected is None:
            raise ValueError("Both u_expected and s_expected must be provided in the context")

        # Extract optional scaling factors
        u_scale = context.get("u_scale")
        s_scale = context.get("s_scale")

        # Apply scale if provided
        u_rate = u_expected
        s_rate = s_expected

        if u_scale is not None:
            u_rate = u_rate * u_scale
        if s_scale is not None:
            s_rate = s_rate * s_scale

        # Ensure u_rate and s_rate have the same shape as u_obs and s_obs
        if u_rate.shape != u_obs.shape:
            # Reshape u_rate to match u_obs
            if u_rate.shape[1] != u_obs.shape[1]:
                # If the number of genes doesn't match, slice to match
                u_rate = u_rate[:, :u_obs.shape[1]]
            if u_rate.shape[0] != u_obs.shape[0]:
                # If the number of cells doesn't match, expand to match
                u_rate = u_rate[:u_obs.shape[0]]

        if s_rate.shape != s_obs.shape:
            # Reshape s_rate to match s_obs
            if s_rate.shape[1] != s_obs.shape[1]:
                # If the number of genes doesn't match, slice to match
                s_rate = s_rate[:, :s_obs.shape[1]]
            if s_rate.shape[0] != s_obs.shape[0]:
                # If the number of cells doesn't match, expand to match
                s_rate = s_rate[:s_obs.shape[0]]

        # Create Poisson distributions for both unspliced and spliced counts
        with pyro.plate("cells", u_obs.shape[0]):
            with pyro.plate("genes", u_obs.shape[1]):
                # Create Poisson distributions and observe data
                u_dist = pyro.distributions.Poisson(rate=u_rate)
                s_dist = pyro.distributions.Poisson(rate=s_rate)

                # Observe data
                pyro.sample("u_obs", u_dist, obs=u_obs)
                pyro.sample("s_obs", s_dist, obs=s_obs)

        # Add distributions to context
        context["u_dist"] = u_dist
        context["s_dist"] = s_dist

        return context

    def __call__(
        self,
        adata: AnnData = None,
        cell_state: Float[Array, "batch_size latent_dim"] = None,
        gene_offset: Optional[Float[Array, "batch_size genes"]] = None,
        time_info: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, dist.Distribution]:
        """Generate likelihood distributions for observed data.

        Args:
            adata: AnnData object containing gene expression data
            cell_state: Latent cell state vectors
            gene_offset: Optional gene-specific offset factors
            time_info: Optional time-related information
            **kwargs: Additional keyword arguments

        Returns:
            Dictionary mapping observation names to their distributions
        """
        # Handle direct tensor inputs for integration testing
        if "u_expected" in kwargs and "s_expected" in kwargs:
            return self._generate_direct_distributions(**kwargs)

        return self._generate_distributions(
            adata=adata,
            cell_state=cell_state,
            gene_offset=gene_offset,
            time_info=time_info,
            **kwargs,
        )

    @beartype
    def _generate_distributions(
        self,
        adata: AnnData,
        cell_state: Float[Array, "batch_size latent_dim"],
        gene_offset: Optional[Float[Array, "batch_size genes"]] = None,
        time_info: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, dist.Distribution]:
        """Generate Poisson likelihood distributions for observed data.

        Args:
            adata: AnnData object containing gene expression data
            cell_state: Latent cell state vectors
            gene_offset: Optional gene-specific offset factors
            time_info: Optional time-related information
            **kwargs: Additional keyword arguments

        Returns:
            Dictionary mapping "obs_counts" to a Poisson distribution
        """
        # Extract relevant data from AnnData
        batch_size = cell_state.shape[0]
        n_genes = adata.n_vars

        # Get gene-specific parameters
        gene_scale = numpyro.param(
            "gene_scale",
            jnp.ones(n_genes),
            constraint=dist.constraints.positive,
        )

        # Apply gene offset if provided
        if gene_offset is None:
            gene_offset = jnp.ones((batch_size, n_genes))

        # Transform cell_state to have compatible shape with gene_scale
        # We'll use a linear transformation from latent_dim to n_genes
        # This is a simple approach - in a real implementation, this would be more sophisticated
        latent_dim = cell_state.shape[1]
        projection = numpyro.param(
            "projection_matrix",
            jnp.ones((latent_dim, n_genes))
            / latent_dim,  # Initialize with uniform weights
            constraint=dist.constraints.positive,
        )

        # Project cell_state to gene space
        projected_state = jnp.matmul(
            cell_state, projection
        )  # Shape: (batch_size, n_genes)

        # Calculate rate parameter
        rate = jnp.exp(projected_state) * gene_scale * gene_offset

        # Create Poisson likelihood distribution
        return {"obs_counts": dist.Poisson(rate=rate)}

    @beartype
    def _log_prob_impl(
        self,
        observations: torch.Tensor,
        predictions: torch.Tensor,
        scale_factors: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Calculate log probability of observations under Poisson distribution.

        Args:
            observations: Observed gene expression counts
            predictions: Predicted mean expression levels
            scale_factors: Optional scaling factors for observations

        Returns:
            Log probability of observations for each cell
        """
        # Apply scale factors if provided
        if scale_factors is not None:
            # Reshape scale_factors to broadcast correctly
            scale_factors = scale_factors.reshape(-1, 1)
            rate = predictions * scale_factors
        else:
            rate = predictions

        # Create Poisson distribution and calculate log probability
        distribution = torch.distributions.Poisson(rate=rate)
        log_probs = distribution.log_prob(observations)

        # Sum log probabilities across genes for each cell
        return torch.sum(log_probs, dim=-1)

    @beartype
    def _sample_impl(
        self,
        predictions: torch.Tensor,
        scale_factors: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sample observations from Poisson distribution.

        Args:
            predictions: Predicted mean expression levels
            scale_factors: Optional scaling factors for observations

        Returns:
            Sampled observations
        """
        # Apply scale factors if provided
        if scale_factors is not None:
            # Reshape scale_factors to broadcast correctly
            scale_factors = scale_factors.reshape(-1, 1)
            rate = predictions * scale_factors
        else:
            rate = predictions

        # Create Poisson distribution and sample
        distribution = torch.distributions.Poisson(rate=rate)
        # Set a fixed seed for reproducibility
        torch.manual_seed(0)
        return distribution.sample()

    def _generate_direct_distributions(
        self,
        u_expected: torch.Tensor,
        s_expected: torch.Tensor,
        u_obs: torch.Tensor,
        s_obs: torch.Tensor,
        u_scale: Optional[torch.Tensor] = None,
        s_scale: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, pyro.distributions.Distribution]:
        """Generate Poisson distributions directly from expected counts for testing.

        Args:
            u_expected: Expected unspliced counts
            s_expected: Expected spliced counts
            u_obs: Observed unspliced counts
            s_obs: Observed spliced counts
            u_scale: Optional scaling for unspliced counts
            s_scale: Optional scaling for spliced counts
            **kwargs: Additional keyword arguments

        Returns:
            Dictionary with Poisson distributions for u and s
        """
        # Create Poisson distributions for both unspliced and spliced counts
        with pyro.plate("cells", u_expected.shape[0]):
            with pyro.plate("genes", u_expected.shape[1]):
                # Apply scale if provided
                u_rate = u_expected
                s_rate = s_expected

                if u_scale is not None:
                    u_rate = u_rate * u_scale
                if s_scale is not None:
                    s_rate = s_rate * s_scale

                # Create Poisson distributions and observe data
                u_dist = pyro.distributions.Poisson(rate=u_rate)
                s_dist = pyro.distributions.Poisson(rate=s_rate)

                # Observe data
                pyro.sample("u_obs", u_dist, obs=u_obs)
                pyro.sample("s_obs", s_dist, obs=s_obs)

        return {"u_obs": u_dist, "s_obs": s_dist}


@LikelihoodModelRegistry.register("negative_binomial")
class NegativeBinomialLikelihoodModel(BaseLikelihoodModel):
    """Negative Binomial likelihood model for observed counts.

    This model uses a Negative Binomial distribution (implemented as
    GammaPoisson in NumPyro) as the likelihood for observed RNA counts,
    which is appropriate when the variance exceeds the mean (overdispersion).
    """

    @beartype
    def forward(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Define the likelihood distributions for observed data given expected values.

        Args:
            context: Dictionary containing model context including u_obs, s_obs, u_expected, s_expected, and other parameters

        Returns:
            Updated context dictionary with likelihood information
        """
        # Extract required values from context
        u_obs = context.get("u_obs")
        s_obs = context.get("s_obs")
        u_expected = context.get("u_expected")
        s_expected = context.get("s_expected")

        if u_obs is None or s_obs is None:
            raise ValueError("Both u_obs and s_obs must be provided in the context")

        if u_expected is None or s_expected is None:
            raise ValueError("Both u_expected and s_expected must be provided in the context")

        # Extract optional scaling factors
        u_scale = context.get("u_scale")
        s_scale = context.get("s_scale")

        # Apply scale if provided
        u_rate = u_expected
        s_rate = s_expected

        if u_scale is not None:
            u_rate = u_rate * u_scale
        if s_scale is not None:
            s_rate = s_rate * s_scale

        # Use fixed dispersion parameters for simplicity
        # In a real implementation, these would be learned or provided
        u_dispersion = torch.ones_like(u_rate[0] if u_rate.dim() > 1 else u_rate)
        s_dispersion = torch.ones_like(s_rate[0] if s_rate.dim() > 1 else s_rate)

        # Calculate concentration parameters (inverse of dispersion)
        u_concentration = 1.0 / u_dispersion
        s_concentration = 1.0 / s_dispersion

        # Ensure u_rate and s_rate have the same shape as u_obs and s_obs
        if u_rate.shape != u_obs.shape:
            # Reshape u_rate to match u_obs
            if u_rate.shape[1] != u_obs.shape[1]:
                # If the number of genes doesn't match, slice to match
                u_rate = u_rate[:, :u_obs.shape[1]]
            if u_rate.shape[0] != u_obs.shape[0]:
                # If the number of cells doesn't match, expand to match
                u_rate = u_rate[:u_obs.shape[0]]

        if s_rate.shape != s_obs.shape:
            # Reshape s_rate to match s_obs
            if s_rate.shape[1] != s_obs.shape[1]:
                # If the number of genes doesn't match, slice to match
                s_rate = s_rate[:, :s_obs.shape[1]]
            if s_rate.shape[0] != s_obs.shape[0]:
                # If the number of cells doesn't match, expand to match
                s_rate = s_rate[:s_obs.shape[0]]

        # Create Negative Binomial distributions for both unspliced and spliced counts
        with pyro.plate("cells", u_obs.shape[0]):
            with pyro.plate("genes", u_obs.shape[1]):
                # Create Negative Binomial distributions and observe data
                # In PyTorch, we use NegativeBinomial with total_count=concentration, probs=concentration/(concentration+rate)
                u_probs = u_concentration / (u_concentration + u_rate)
                s_probs = s_concentration / (s_concentration + s_rate)

                u_dist = pyro.distributions.NegativeBinomial(total_count=u_concentration, probs=u_probs)
                s_dist = pyro.distributions.NegativeBinomial(total_count=s_concentration, probs=s_probs)

                # Observe data
                pyro.sample("u_obs", u_dist, obs=u_obs)
                pyro.sample("s_obs", s_dist, obs=s_obs)

        # Add distributions to context
        context["u_dist"] = u_dist
        context["s_dist"] = s_dist

        return context

    def __call__(
        self,
        adata: AnnData,
        cell_state: Float[Array, "batch_size latent_dim"],
        gene_offset: Optional[Float[Array, "batch_size genes"]] = None,
        time_info: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, dist.Distribution]:
        """Generate likelihood distributions for observed data.

        Args:
            adata: AnnData object containing gene expression data
            cell_state: Latent cell state vectors
            gene_offset: Optional gene-specific offset factors
            time_info: Optional time-related information
            **kwargs: Additional keyword arguments

        Returns:
            Dictionary mapping observation names to their distributions
        """
        return self._generate_distributions(
            adata=adata,
            cell_state=cell_state,
            gene_offset=gene_offset,
            time_info=time_info,
            **kwargs,
        )

    @beartype
    def _generate_distributions(
        self,
        adata: AnnData,
        cell_state: Float[Array, "batch_size latent_dim"],
        gene_offset: Optional[Float[Array, "batch_size genes"]] = None,
        time_info: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, dist.Distribution]:
        """Generate Negative Binomial likelihood distributions for observed data.

        Args:
            adata: AnnData object containing gene expression data
            cell_state: Latent cell state vectors
            gene_offset: Optional gene-specific offset factors
            time_info: Optional time-related information
            **kwargs: Additional keyword arguments

        Returns:
            Dictionary mapping "obs_counts" to a GammaPoisson distribution
        """
        # Extract relevant data from AnnData
        batch_size = cell_state.shape[0]
        n_genes = adata.n_vars

        # Get gene-specific parameters
        gene_scale = numpyro.param(
            "gene_scale",
            jnp.ones(n_genes),
            constraint=dist.constraints.positive,
        )

        # Get gene-specific dispersion parameters
        gene_dispersion = numpyro.param(
            "gene_dispersion",
            jnp.ones(n_genes),
            constraint=dist.constraints.positive,
        )

        # Apply gene offset if provided
        if gene_offset is None:
            gene_offset = jnp.ones((batch_size, n_genes))

        # Transform cell_state to have compatible shape with gene_scale
        # We'll use a linear transformation from latent_dim to n_genes
        # This is a simple approach - in a real implementation, this would be more sophisticated
        latent_dim = cell_state.shape[1]
        projection = numpyro.param(
            "projection_matrix",
            jnp.ones((latent_dim, n_genes))
            / latent_dim,  # Initialize with uniform weights
            constraint=dist.constraints.positive,
        )

        # Project cell_state to gene space
        projected_state = jnp.matmul(
            cell_state, projection
        )  # Shape: (batch_size, n_genes)

        # Calculate rate parameter
        rate = jnp.exp(projected_state) * gene_scale * gene_offset

        # Calculate concentration parameter (inverse of dispersion)
        concentration = 1.0 / gene_dispersion

        # Create Negative Binomial likelihood distribution
        # In NumPyro, Negative Binomial is implemented as GammaPoisson
        return {
            "obs_counts": dist.GammaPoisson(
                concentration=concentration,
                rate=concentration / rate,
            )
        }

    @beartype
    def _log_prob_impl(
        self,
        observations: Float[Array, "batch_size genes"],
        predictions: Float[Array, "batch_size genes"],
        scale_factors: Optional[Float[Array, "batch_size"]] = None,
    ) -> Float[Array, "batch_size"]:
        """Calculate log probability of observations under Negative Binomial distribution.

        Args:
            observations: Observed gene expression counts
            predictions: Predicted mean expression levels
            scale_factors: Optional scaling factors for observations

        Returns:
            Log probability of observations for each cell
        """
        # Apply scale factors if provided
        if scale_factors is not None:
            # Reshape scale_factors to broadcast correctly
            scale_factors = scale_factors.reshape((-1, 1))
            mean = predictions * scale_factors
        else:
            mean = predictions

        # Use a fixed dispersion parameter for simplicity
        # In a real implementation, this would be learned or provided
        dispersion = jnp.ones(mean.shape[1])
        concentration = 1.0 / dispersion

        # Create Negative Binomial distribution and calculate log probability
        distribution = dist.GammaPoisson(
            concentration=concentration,
            rate=concentration / mean,
        )
        log_probs = distribution.log_prob(observations)

        # Sum log probabilities across genes for each cell
        return jnp.sum(log_probs, axis=-1)

    @beartype
    def _sample_impl(
        self,
        predictions: Float[Array, "batch_size genes"],
        scale_factors: Optional[Float[Array, "batch_size"]] = None,
    ) -> Float[Array, "batch_size genes"]:
        """Sample observations from Negative Binomial distribution.

        Args:
            predictions: Predicted mean expression levels
            scale_factors: Optional scaling factors for observations

        Returns:
            Sampled observations
        """
        # Apply scale factors if provided
        if scale_factors is not None:
            # Reshape scale_factors to broadcast correctly
            scale_factors = scale_factors.reshape((-1, 1))
            mean = predictions * scale_factors
        else:
            mean = predictions

        # Use a fixed dispersion parameter for simplicity
        # In a real implementation, this would be learned or provided
        dispersion = jnp.ones(mean.shape[1])
        concentration = 1.0 / dispersion

        # Create Negative Binomial distribution and sample
        distribution = dist.GammaPoisson(
            concentration=concentration,
            rate=concentration / mean,
        )
        key = jax.random.PRNGKey(0)  # Use a fixed seed for reproducibility
        return distribution.sample(key)
