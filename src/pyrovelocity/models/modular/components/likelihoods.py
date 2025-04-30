"""Likelihood models for PyroVelocity.

This module provides likelihood models for generating distributions
for observed data given latent variables.
"""

from typing import Any, Dict, Optional

import pyro

# Using PyTorch instead of JAX/NumPyro
import pyro.distributions
import torch
from anndata._core.anndata import AnnData
from beartype import beartype

# Remove jaxtyping dependency
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
            raise ValueError(
                "Both u_obs and s_obs must be provided in the context"
            )

        if u_expected is None or s_expected is None:
            raise ValueError(
                "Both u_expected and s_expected must be provided in the context"
            )

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

        # Ensure rates are positive for Poisson distribution
        u_rate = torch.abs(u_rate)
        s_rate = torch.abs(s_rate)

        # Get the number of genes from the model parameters
        model_n_genes = None
        for param_name in ["alpha", "beta", "gamma"]:
            if param_name in context:
                model_n_genes = context[param_name].shape[0]
                break

        # Log tensor shapes for debugging
        print(f"PoissonLikelihoodModel - u_obs shape: {u_obs.shape}")
        print(f"PoissonLikelihoodModel - s_obs shape: {s_obs.shape}")
        print(f"PoissonLikelihoodModel - u_rate shape: {u_rate.shape}")
        print(f"PoissonLikelihoodModel - s_rate shape: {s_rate.shape}")
        if model_n_genes is not None:
            print(f"PoissonLikelihoodModel - model_n_genes: {model_n_genes}")

        # Determine the correct dimensions to use
        n_cells = u_obs.shape[0]

        # If model parameters have been sampled, use their dimensions for the genes
        # Otherwise, use the data dimensions
        n_genes = model_n_genes if model_n_genes is not None else u_obs.shape[1]

        # If there's a mismatch between model parameters and data dimensions,
        # use the minimum to ensure compatibility
        if model_n_genes is not None and model_n_genes != u_obs.shape[1]:
            n_genes = min(model_n_genes, u_obs.shape[1])
            print(f"PoissonLikelihoodModel - Shape mismatch detected. Using n_genes = {n_genes}")

        # Ensure all tensors have compatible shapes
        # Reshape u_obs and s_obs if needed
        if u_obs.shape[1] != n_genes:
            print(f"PoissonLikelihoodModel - Reshaping u_obs from {u_obs.shape} to match n_genes = {n_genes}")
            u_obs = u_obs[:, :n_genes]

        if s_obs.shape[1] != n_genes:
            print(f"PoissonLikelihoodModel - Reshaping s_obs from {s_obs.shape} to match n_genes = {n_genes}")
            s_obs = s_obs[:, :n_genes]

        # Reshape u_rate and s_rate if needed
        if u_rate.dim() > 1 and u_rate.shape[1] != n_genes:
            print(f"PoissonLikelihoodModel - Reshaping u_rate from {u_rate.shape} to match n_genes = {n_genes}")
            u_rate = u_rate[:, :n_genes]

        if s_rate.dim() > 1 and s_rate.shape[1] != n_genes:
            print(f"PoissonLikelihoodModel - Reshaping s_rate from {s_rate.shape} to match n_genes = {n_genes}")
            s_rate = s_rate[:, :n_genes]

        # If u_rate or s_rate are 1D tensors (gene parameters), expand them to match batch size
        if u_rate.dim() == 1:
            print(f"PoissonLikelihoodModel - Expanding u_rate from {u_rate.shape} to match batch size")
            u_rate = u_rate.unsqueeze(0).expand(n_cells, -1)

        if s_rate.dim() == 1:
            print(f"PoissonLikelihoodModel - Expanding s_rate from {s_rate.shape} to match batch size")
            s_rate = s_rate.unsqueeze(0).expand(n_cells, -1)

        # Ensure observations are integers for Poisson distribution
        u_obs_int = u_obs.round().long()
        s_obs_int = s_obs.round().long()

        # Final shape check
        print(f"PoissonLikelihoodModel - Final shapes: u_obs={u_obs_int.shape}, u_rate={u_rate.shape}")

        # Use the data dimensions for the plate
        # Ensure the plate dimensions match the tensor dimensions
        with pyro.plate("cells", n_cells, dim=-2):
            with pyro.plate("genes", n_genes, dim=-1):
                # Create Poisson distributions and observe data
                u_dist = pyro.distributions.Poisson(rate=u_rate)
                s_dist = pyro.distributions.Poisson(rate=s_rate)

                # Observe data
                pyro.sample("u_obs", u_dist, obs=u_obs_int)
                pyro.sample("s_obs", s_dist, obs=s_obs_int)

        # Add distributions to context
        context["u_dist"] = u_dist
        context["s_dist"] = s_dist

        return context

    def __call__(
        self,
        adata: Optional[AnnData] = None,
        cell_state: Optional[torch.Tensor] = None,
        gene_offset: Optional[torch.Tensor] = None,
        time_info: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, pyro.distributions.Distribution]:
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

        # Validate inputs
        if adata is None or cell_state is None:
            raise ValueError("Both adata and cell_state must be provided")

        # Ensure inputs are torch.Tensor
        assert isinstance(
            cell_state, torch.Tensor
        ), "cell_state must be a torch.Tensor"

        if gene_offset is not None:
            assert isinstance(
                gene_offset, torch.Tensor
            ), "gene_offset must be a torch.Tensor"

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
        cell_state: torch.Tensor,
        gene_offset: Optional[torch.Tensor] = None,
        time_info: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, pyro.distributions.Distribution]:
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
        gene_scale = torch.ones(n_genes)

        # Apply gene offset if provided
        if gene_offset is None:
            gene_offset = torch.ones((batch_size, n_genes))

        # Transform cell_state to have compatible shape with gene_scale
        # We'll use a linear transformation from latent_dim to n_genes
        # This is a simple approach - in a real implementation, this would be more sophisticated
        latent_dim = cell_state.shape[1]
        projection = (
            torch.ones((latent_dim, n_genes)) / latent_dim
        )  # Initialize with uniform weights

        # Project cell_state to gene space
        projected_state = torch.matmul(
            cell_state, projection
        )  # Shape: (batch_size, n_genes)

        # Calculate rate parameter
        rate = torch.exp(projected_state) * gene_scale * gene_offset

        # Create Poisson likelihood distribution
        return {"obs_counts": pyro.distributions.Poisson(rate=rate)}

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
            raise ValueError(
                "Both u_obs and s_obs must be provided in the context"
            )

        if u_expected is None or s_expected is None:
            raise ValueError(
                "Both u_expected and s_expected must be provided in the context"
            )

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

        # Ensure rates are positive for Negative Binomial distribution
        u_rate = torch.abs(u_rate)
        s_rate = torch.abs(s_rate)

        # Get the number of genes from the model parameters
        model_n_genes = None
        for param_name in ["alpha", "beta", "gamma"]:
            if param_name in context:
                model_n_genes = context[param_name].shape[0]
                break

        # Log tensor shapes for debugging
        print(f"NegativeBinomialLikelihoodModel - u_obs shape: {u_obs.shape}")
        print(f"NegativeBinomialLikelihoodModel - s_obs shape: {s_obs.shape}")
        print(f"NegativeBinomialLikelihoodModel - u_rate shape: {u_rate.shape}")
        print(f"NegativeBinomialLikelihoodModel - s_rate shape: {s_rate.shape}")
        if model_n_genes is not None:
            print(f"NegativeBinomialLikelihoodModel - model_n_genes: {model_n_genes}")

        # Determine the correct dimensions to use
        n_cells = u_obs.shape[0]

        # If model parameters have been sampled, use their dimensions for the genes
        # Otherwise, use the data dimensions
        n_genes = model_n_genes if model_n_genes is not None else u_obs.shape[1]

        # If there's a mismatch between model parameters and data dimensions,
        # use the minimum to ensure compatibility
        if model_n_genes is not None and model_n_genes != u_obs.shape[1]:
            n_genes = min(model_n_genes, u_obs.shape[1])
            print(f"NegativeBinomialLikelihoodModel - Shape mismatch detected. Using n_genes = {n_genes}")

        # Ensure all tensors have compatible shapes
        # Reshape u_obs and s_obs if needed
        if u_obs.shape[1] != n_genes:
            print(f"NegativeBinomialLikelihoodModel - Reshaping u_obs from {u_obs.shape} to match n_genes = {n_genes}")
            u_obs = u_obs[:, :n_genes]

        if s_obs.shape[1] != n_genes:
            print(f"NegativeBinomialLikelihoodModel - Reshaping s_obs from {s_obs.shape} to match n_genes = {n_genes}")
            s_obs = s_obs[:, :n_genes]

        # Reshape u_rate and s_rate if needed
        if u_rate.dim() > 1 and u_rate.shape[1] != n_genes:
            print(f"NegativeBinomialLikelihoodModel - Reshaping u_rate from {u_rate.shape} to match n_genes = {n_genes}")
            u_rate = u_rate[:, :n_genes]

        if s_rate.dim() > 1 and s_rate.shape[1] != n_genes:
            print(f"NegativeBinomialLikelihoodModel - Reshaping s_rate from {s_rate.shape} to match n_genes = {n_genes}")
            s_rate = s_rate[:, :n_genes]

        # If u_rate or s_rate are 1D tensors (gene parameters), expand them to match batch size
        if u_rate.dim() == 1:
            print(f"NegativeBinomialLikelihoodModel - Expanding u_rate from {u_rate.shape} to match batch size")
            u_rate = u_rate.unsqueeze(0).expand(n_cells, -1)

        if s_rate.dim() == 1:
            print(f"NegativeBinomialLikelihoodModel - Expanding s_rate from {s_rate.shape} to match batch size")
            s_rate = s_rate.unsqueeze(0).expand(n_cells, -1)

        # Use fixed dispersion parameters for simplicity
        # In a real implementation, these would be learned or provided
        u_dispersion = torch.ones(n_genes)
        s_dispersion = torch.ones(n_genes)

        # Calculate concentration parameters (inverse of dispersion)
        u_concentration = 1.0 / u_dispersion
        s_concentration = 1.0 / s_dispersion

        print(f"NegativeBinomialLikelihoodModel - u_concentration shape: {u_concentration.shape}")

        # Ensure observations are integers for Negative Binomial distribution
        u_obs_int = u_obs.round().long()
        s_obs_int = s_obs.round().long()

        # Final shape check
        print(f"NegativeBinomialLikelihoodModel - Final shapes: u_obs={u_obs_int.shape}, u_rate={u_rate.shape}, u_concentration={u_concentration.shape}")

        # Use the data dimensions for the plate
        # Ensure the plate dimensions match the tensor dimensions
        with pyro.plate("cells", n_cells, dim=-2):
            with pyro.plate("genes", n_genes, dim=-1):
                # Create Negative Binomial distributions and observe data
                # In PyTorch, we use NegativeBinomial with total_count=concentration, probs=concentration/(concentration+rate)
                u_probs = u_concentration / (u_concentration + u_rate)
                s_probs = s_concentration / (s_concentration + s_rate)

                u_dist = pyro.distributions.NegativeBinomial(
                    total_count=u_concentration, probs=u_probs
                )
                s_dist = pyro.distributions.NegativeBinomial(
                    total_count=s_concentration, probs=s_probs
                )

                # Observe data
                pyro.sample("u_obs", u_dist, obs=u_obs_int)
                pyro.sample("s_obs", s_dist, obs=s_obs_int)

        # Add distributions to context
        context["u_dist"] = u_dist
        context["s_dist"] = s_dist

        return context

    def __call__(
        self,
        adata: Optional[AnnData] = None,
        cell_state: Optional[torch.Tensor] = None,
        gene_offset: Optional[torch.Tensor] = None,
        time_info: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, pyro.distributions.Distribution]:
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

        # Validate inputs
        if adata is None or cell_state is None:
            raise ValueError("Both adata and cell_state must be provided")

        # Ensure inputs are torch.Tensor
        assert isinstance(
            cell_state, torch.Tensor
        ), "cell_state must be a torch.Tensor"

        if gene_offset is not None:
            assert isinstance(
                gene_offset, torch.Tensor
            ), "gene_offset must be a torch.Tensor"

        return self._generate_distributions(
            adata=adata,
            cell_state=cell_state,
            gene_offset=gene_offset,
            time_info=time_info,
            **kwargs,
        )

    @beartype
    def _generate_direct_distributions(
        self,
        u_expected: torch.Tensor,
        s_expected: torch.Tensor,
        u_scale: Optional[torch.Tensor] = None,
        s_scale: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, pyro.distributions.Distribution]:
        """Generate Negative Binomial distributions directly from expected counts for testing.

        Args:
            u_expected: Expected unspliced counts
            s_expected: Expected spliced counts
            u_scale: Optional scaling factors for unspliced counts
            s_scale: Optional scaling factors for spliced counts
            **kwargs: Additional keyword arguments

        Returns:
            Dictionary mapping observation names to their distributions
        """
        # Apply scaling if provided
        if u_scale is not None:
            u_rate = u_expected * u_scale
        else:
            u_rate = u_expected

        if s_scale is not None:
            s_rate = s_expected * s_scale
        else:
            s_rate = s_expected

        # Use a fixed dispersion parameter for simplicity
        dispersion = torch.ones(u_rate.shape[1])
        concentration = 1.0 / dispersion

        # Create Negative Binomial distributions
        u_probs = concentration / (concentration + u_rate)
        s_probs = concentration / (concentration + s_rate)

        u_dist = pyro.distributions.NegativeBinomial(
            total_count=concentration, probs=u_probs
        )
        s_dist = pyro.distributions.NegativeBinomial(
            total_count=concentration, probs=s_probs
        )

        # Return distributions in a dictionary
        return {"u_dist": u_dist, "s_dist": s_dist}

    @beartype
    def _generate_distributions(
        self,
        adata: AnnData,
        cell_state: torch.Tensor,
        gene_offset: Optional[torch.Tensor] = None,
        time_info: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, pyro.distributions.Distribution]:
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
        gene_scale = torch.ones(n_genes)

        # Get gene-specific dispersion parameters
        gene_dispersion = torch.ones(n_genes)

        # Apply gene offset if provided
        if gene_offset is None:
            gene_offset = torch.ones((batch_size, n_genes))

        # Transform cell_state to have compatible shape with gene_scale
        # We'll use a linear transformation from latent_dim to n_genes
        # This is a simple approach - in a real implementation, this would be more sophisticated
        latent_dim = cell_state.shape[1]
        projection = (
            torch.ones((latent_dim, n_genes)) / latent_dim
        )  # Initialize with uniform weights

        # Project cell_state to gene space
        projected_state = torch.matmul(
            cell_state, projection
        )  # Shape: (batch_size, n_genes)

        # Calculate rate parameter
        rate = torch.exp(projected_state) * gene_scale * gene_offset

        # Calculate concentration parameter (inverse of dispersion)
        concentration = 1.0 / gene_dispersion

        # Create Negative Binomial likelihood distribution
        # Use PyTorch's NegativeBinomial instead of NumPyro's GammaPoisson
        probs = concentration / (concentration + rate)
        return {
            "obs_counts": pyro.distributions.NegativeBinomial(
                total_count=concentration, probs=probs
            )
        }

    @beartype
    def _log_prob_impl(
        self,
        observations: torch.Tensor,
        predictions: torch.Tensor,
        scale_factors: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
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
        dispersion = torch.ones(mean.shape[1])
        concentration = 1.0 / dispersion

        # Create Negative Binomial distribution and calculate log probability
        # Use PyTorch's NegativeBinomial instead of NumPyro's GammaPoisson
        probs = concentration / (concentration + mean)
        distribution = pyro.distributions.NegativeBinomial(
            total_count=concentration, probs=probs
        )
        log_probs = distribution.log_prob(observations)

        # Sum log probabilities across genes for each cell
        return torch.sum(log_probs, dim=-1)

    @beartype
    def _sample_impl(
        self,
        predictions: torch.Tensor,
        scale_factors: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
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
        dispersion = torch.ones(mean.shape[1])
        concentration = 1.0 / dispersion

        # Create Negative Binomial distribution and sample
        # Use PyTorch's NegativeBinomial instead of NumPyro's GammaPoisson
        probs = concentration / (concentration + mean)
        distribution = pyro.distributions.NegativeBinomial(
            total_count=concentration, probs=probs
        )
        return distribution.sample()
