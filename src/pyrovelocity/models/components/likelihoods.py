"""Likelihood models for PyroVelocity.

This module provides likelihood models for generating distributions
for observed data given latent variables.
"""

from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from beartype import beartype
from jaxtyping import Array, Float, Int
from anndata._core.anndata import AnnData

from pyrovelocity.models.components.base import BaseLikelihoodModel
from pyrovelocity.models.registry import LikelihoodModelRegistry


@LikelihoodModelRegistry.register("poisson")
class PoissonLikelihoodModel(BaseLikelihoodModel):
    """Poisson likelihood model for observed counts.

    This model uses a Poisson distribution as the likelihood for
    observed RNA counts, which is appropriate when the variance
    is approximately equal to the mean.
    """

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
        observations: Float[Array, "batch_size genes"],
        predictions: Float[Array, "batch_size genes"],
        scale_factors: Optional[Float[Array, "batch_size"]] = None,
    ) -> Float[Array, "batch_size"]:
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
            scale_factors = scale_factors.reshape((-1, 1))
            rate = predictions * scale_factors
        else:
            rate = predictions

        # Create Poisson distribution and calculate log probability
        distribution = dist.Poisson(rate=rate)
        log_probs = distribution.log_prob(observations)

        # Sum log probabilities across genes for each cell
        return jnp.sum(log_probs, axis=-1)

    @beartype
    def _sample_impl(
        self,
        predictions: Float[Array, "batch_size genes"],
        scale_factors: Optional[Float[Array, "batch_size"]] = None,
    ) -> Float[Array, "batch_size genes"]:
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
            scale_factors = scale_factors.reshape((-1, 1))
            rate = predictions * scale_factors
        else:
            rate = predictions

        # Create Poisson distribution and sample
        distribution = dist.Poisson(rate=rate)
        key = jax.random.PRNGKey(0)  # Use a fixed seed for reproducibility
        return distribution.sample(key)


@LikelihoodModelRegistry.register("negative_binomial")
class NegativeBinomialLikelihoodModel(BaseLikelihoodModel):
    """Negative Binomial likelihood model for observed counts.

    This model uses a Negative Binomial distribution (implemented as
    GammaPoisson in NumPyro) as the likelihood for observed RNA counts,
    which is appropriate when the variance exceeds the mean (overdispersion).
    """

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
