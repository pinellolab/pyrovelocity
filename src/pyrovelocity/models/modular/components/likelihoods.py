"""
Protocol-First likelihood model implementations for PyroVelocity's modular architecture.

This module contains likelihood model implementations that directly implement the
LikelihoodModel Protocol. These implementations follow the Protocol-First approach,
which embraces composition over inheritance and allows for more flexible component composition.

This module has been simplified to include only the essential components needed for
validation against the legacy implementation:
- PoissonLikelihoodModel: Poisson likelihood model for observed counts
- LegacyLikelihoodModel: Legacy likelihood model for observed counts that exactly matches the legacy implementation
"""

from typing import Any, Dict, Optional

import pyro
import torch
from anndata import AnnData
from beartype import beartype

from pyrovelocity.models.modular.interfaces import LikelihoodModel
from pyrovelocity.models.modular.registry import LikelihoodModelRegistry
from pyrovelocity.models.modular.utils.context_utils import validate_context


@LikelihoodModelRegistry.register("poisson")
class PoissonLikelihoodModel:
    """Poisson likelihood model for observed counts.

    This model uses a Poisson distribution as the likelihood for
    observed RNA counts, which is appropriate when the variance
    is approximately equal to the mean.

    This implementation directly implements the LikelihoodModel Protocol.
    """

    def __init__(self, name: str = "poisson_likelihood"):
        """
        Initialize the PoissonLikelihoodModel.

        Args:
            name: A unique name for this component instance.
        """
        self.name = name

    @beartype
    def forward(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Define the likelihood distributions for observed data given expected values.

        Args:
            context: Dictionary containing model context including u_obs, s_obs, u_expected, s_expected, and other parameters

        Returns:
            Updated context dictionary with likelihood information
        """
        # Validate context
        validation_result = validate_context(
            self.__class__.__name__,
            context,
            required_keys=["u_obs", "s_obs", "u_expected", "s_expected"],
            tensor_keys=["u_obs", "s_obs", "u_expected", "s_expected"],
        )

        if isinstance(validation_result, dict):
            # Extract required values from context
            u_obs = context["u_obs"]
            s_obs = context["s_obs"]
            u_expected = context["u_expected"]
            s_expected = context["s_expected"]

                                                            
            # Extract optional scaling factors
            u_scale = context.get("u_scale")
            s_scale = context.get("s_scale")

            # Apply scaling factors if provided
            u_rate = u_expected
            s_rate = s_expected

            if u_scale is not None:
                                u_rate = u_rate * u_scale
            if s_scale is not None:
                                s_rate = s_rate * s_scale

            # Get model dimensions if available
            model_n_genes = None
            for param in ["alpha", "beta", "gamma"]:
                if param in context and isinstance(context[param], torch.Tensor):
                    model_n_genes = context[param].shape[-1]
                    break

            # Determine the correct dimensions to use
            n_cells = u_obs.shape[0]

            # If model parameters have been sampled, use their dimensions for the genes
            # Otherwise, use the data dimensions
            n_genes = model_n_genes if model_n_genes is not None else u_obs.shape[1]

            # If there's a mismatch between model parameters and data dimensions,
            # use the minimum to ensure compatibility
            if model_n_genes is not None and model_n_genes != u_obs.shape[1]:
                n_genes = min(model_n_genes, u_obs.shape[1])
                print(f"PoissonLikelihoodModel - Using {n_genes} genes (min of model and data)")

            # Ensure all tensors have compatible shapes
            # Reshape u_obs and s_obs if needed
            if u_obs.shape[1] != n_genes:
                u_obs = u_obs[:, :n_genes]
                
            if s_obs.shape[1] != n_genes:
                s_obs = s_obs[:, :n_genes]
                
            # Ensure observations are integers for Poisson distribution
            u_obs_int = u_obs.round().long()
            s_obs_int = s_obs.round().long()

            # Handle different shapes for u_rate and s_rate
            # Check if we have batch dimensions (from posterior samples)
            has_batch_dim = False
            batch_size = 1

            # Check if u_rate has a batch dimension
            if u_rate.dim() == 3:  # Shape: [batch_size, num_cells, num_genes]
                has_batch_dim = True
                batch_size = u_rate.shape[0]
                print(f"PoissonLikelihoodModel - Detected batch dimension: {batch_size}")
            elif u_rate.dim() == 2:  # Shape: [num_cells, num_genes]
                # No batch dimension
                pass
            elif u_rate.dim() == 1:  # Shape: [num_genes]
                # Gene parameters only, need to expand to cells
                u_rate = u_rate.unsqueeze(0).expand(n_cells, -1)
                s_rate = s_rate.unsqueeze(0).expand(n_cells, -1)
                
            # Reshape u_rate and s_rate if needed to match n_genes
            if u_rate.dim() > 1 and u_rate.shape[-1] != n_genes:
                if u_rate.dim() == 3:  # Shape: [batch_size, num_cells, num_genes]
                    u_rate = u_rate[:, :, :n_genes]
                    s_rate = s_rate[:, :, :n_genes]
                else:  # Shape: [num_cells, num_genes]
                    u_rate = u_rate[:, :n_genes]
                    s_rate = s_rate[:, :n_genes]
                
            # Ensure all rate values are positive (required for Poisson distribution)
            # Use a small positive value (epsilon) as the minimum rate
            epsilon = 1e-6
            u_rate = torch.maximum(u_rate, torch.tensor(epsilon))
            s_rate = torch.maximum(s_rate, torch.tensor(epsilon))
            print(f"PoissonLikelihoodModel - Ensured positive rate values with min: {epsilon}")

            # Create Poisson distributions
            u_dist = pyro.distributions.Poisson(rate=u_rate)
            s_dist = pyro.distributions.Poisson(rate=s_rate)

            # Handle observations based on whether we have batch dimensions
            if has_batch_dim:
                # We need to expand observations to match the batch dimension
                u_obs_expanded = u_obs_int.unsqueeze(0).expand(batch_size, -1, -1)
                s_obs_expanded = s_obs_int.unsqueeze(0).expand(batch_size, -1, -1)

                                
                # Use plates with consistent dimensions
                with pyro.plate("batch", batch_size, dim=-3):
                    with pyro.plate("cells_likelihood", n_cells, dim=-2):
                        with pyro.plate("genes_likelihood", n_genes, dim=-1):
                            # Observe data
                            pyro.sample("u_obs", u_dist, obs=u_obs_expanded)
                            pyro.sample("s_obs", s_dist, obs=s_obs_expanded)
            else:
                # No batch dimension, use standard plates
                
                # Use plates with consistent dimensions
                with pyro.plate("cells_likelihood", n_cells, dim=-2):
                    with pyro.plate("genes_likelihood", n_genes, dim=-1):
                        # Observe data
                        pyro.sample("u_obs", u_dist, obs=u_obs_int)
                        pyro.sample("s_obs", s_dist, obs=s_obs_int)

            # Add distributions to context
            context["u_dist"] = u_dist
            context["s_dist"] = s_dist

            return context
        else:
            # If validation failed, raise an error
            raise ValueError(f"Error in likelihood model forward pass: {validation_result.error}")

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
            Dictionary of likelihood distributions
        """
        if adata is None and cell_state is None:
            raise ValueError("Either adata or cell_state must be provided")

        if adata is not None:
            return self._generate_distributions_from_adata(
                adata, gene_offset, time_info, **kwargs
            )
        else:
            return self._generate_distributions_from_cell_state(
                cell_state, gene_offset, time_info, **kwargs
            )

    def _generate_distributions_from_adata(
        self,
        adata: AnnData,
        gene_offset: Optional[torch.Tensor] = None,
        time_info: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, pyro.distributions.Distribution]:
        """Generate Poisson likelihood distributions from AnnData.

        Args:
            adata: AnnData object containing gene expression data
            gene_offset: Optional gene-specific offset factors
            time_info: Optional time-related information
            **kwargs: Additional keyword arguments

        Returns:
            Dictionary of likelihood distributions
        """
        # Check if cell_state is provided in kwargs
        cell_state = kwargs.get("cell_state")

        if cell_state is not None:
            # If cell_state is provided, use it to generate distributions
            return self._generate_distributions_from_cell_state(
                cell_state, gene_offset, time_info, n_genes=adata.n_vars, **kwargs
            )

        # Otherwise, use AnnData to generate distributions
        # Extract counts from AnnData
        if "X" not in adata.layers:
            raise ValueError("AnnData object must have 'X' in layers")

        # Convert to torch tensor
        # Handle both sparse and dense arrays
        layer_data = adata.layers["X"]
        if hasattr(layer_data, 'toarray'):
            # For sparse matrices
            counts = torch.tensor(layer_data.toarray(), dtype=torch.float32)
        else:
            # For numpy arrays
            counts = torch.tensor(layer_data, dtype=torch.float32)

        # Use mean expression as rate parameter
        rate = torch.mean(counts, dim=0)

        # Check if we need to expand the rate to match a batch dimension
        # This is needed for the tests that expect a batch dimension
        batch_size = kwargs.get("batch_size")
        if batch_size is not None:
            rate = rate.unsqueeze(0).expand(batch_size, -1)

        # Apply gene offset if provided
        if gene_offset is not None:
            # If gene_offset has a batch dimension, use it directly
            if gene_offset.dim() > 1:
                rate = rate.expand(gene_offset.shape[0], -1) * gene_offset
            else:
                rate = rate * gene_offset

        # Create Poisson likelihood distribution
        return {"obs_counts": pyro.distributions.Poisson(rate=rate)}

    def _generate_distributions_from_cell_state(
        self,
        cell_state: torch.Tensor,
        gene_offset: Optional[torch.Tensor] = None,
        time_info: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, pyro.distributions.Distribution]:
        """Generate Poisson likelihood distributions from cell state.

        Args:
            cell_state: Latent cell state vectors
            gene_offset: Optional gene-specific offset factors
            time_info: Optional time-related information
            **kwargs: Additional keyword arguments

        Returns:
            Dictionary of likelihood distributions
        """
        # Get projection matrix from kwargs or create a random one
        projection = kwargs.get("projection")
        if projection is None:
            # Create a random projection matrix
            n_latent = cell_state.shape[1]
            n_genes = kwargs.get("n_genes", 100)
            projection = torch.randn(n_latent, n_genes)

        # Get gene scale from kwargs or use default
        gene_scale = kwargs.get("gene_scale", torch.ones(projection.shape[1]))

        # Get batch size from cell_state
        batch_size = cell_state.shape[0]

        # Apply gene offset if not provided
        if gene_offset is None:
            gene_offset = torch.ones((batch_size, projection.shape[1]))
        elif gene_offset.dim() == 1:
            # If gene_offset is 1D, expand it to match batch size
            gene_offset = gene_offset.unsqueeze(0).expand(batch_size, -1)

        # Ensure gene_scale has the right shape
        if gene_scale.dim() == 1:
            gene_scale = gene_scale.unsqueeze(0).expand(batch_size, -1)

        # Project cell_state to gene space
        projected_state = torch.matmul(
            cell_state, projection
        )  # Shape: (batch_size, n_genes)

        # Calculate rate parameter
        rate = torch.exp(projected_state) * gene_scale * gene_offset

        # Create Poisson likelihood distribution
        return {"obs_counts": pyro.distributions.Poisson(rate=rate)}

    def log_prob(
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

    def sample(
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


@LikelihoodModelRegistry.register("legacy")
class LegacyLikelihoodModel:
    """Legacy likelihood model for observed counts.

    This model is specifically designed to work with the LegacyDynamicsModel
    and handle the shape mismatches that can occur between the legacy and
    modular implementations.

    This implementation directly implements the LikelihoodModel Protocol.
    """

    def __init__(self, name: str = "legacy_likelihood"):
        """
        Initialize the LegacyLikelihoodModel.

        Args:
            name: A unique name for this component instance.
        """
        self.name = name

    @beartype
    def forward(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Define the likelihood distributions for observed data given expected values.

        This method is specifically designed to handle the shape mismatches that
        can occur between the legacy and modular implementations.

        Args:
            context: Dictionary containing model context including u_obs, s_obs, u_expected, s_expected, and other parameters

        Returns:
            Updated context dictionary with likelihood information
        """
        # Validate context
        validation_result = validate_context(
            self.__class__.__name__,
            context,
            required_keys=["u_obs", "s_obs", "u_expected", "s_expected"],
            tensor_keys=["u_obs", "s_obs", "u_expected", "s_expected"],
        )

        if isinstance(validation_result, dict):
            # Extract required values from context
            u_obs = context["u_obs"]
            s_obs = context["s_obs"]
            u_expected = context["u_expected"]
            s_expected = context["s_expected"]

            # Extract optional scaling factors
            u_scale = context.get("u_scale")
            s_scale = context.get("s_scale")

            # Apply scaling factors if provided
            u_rate = u_expected
            s_rate = s_expected

            if u_scale is not None:
                u_rate = u_rate * u_scale
            if s_scale is not None:
                s_rate = s_rate * s_scale

            # Ensure all rate values are positive
            u_rate = torch.abs(u_rate) + 1e-6
            s_rate = torch.abs(s_rate) + 1e-6

            # Get dimensions
            n_cells = u_obs.shape[0]
            n_genes = u_obs.shape[1]

            # Reshape u_rate and s_rate to match the observations
            # This is critical for the legacy model compatibility
            if u_rate.dim() == 3:  # Shape: [batch_size, num_cells, num_genes]
                # Already in the right shape, no need to reshape
                pass
            elif u_rate.dim() == 2:  # Shape: [batch_size, num_genes]
                # Reshape to [batch_size, num_cells, num_genes]
                u_rate = u_rate.unsqueeze(1).expand(-1, n_cells, -1)
                s_rate = s_rate.unsqueeze(1).expand(-1, n_cells, -1)
            elif u_rate.dim() == 1:  # Shape: [num_genes]
                # Reshape to [1, num_cells, num_genes]
                u_rate = u_rate.unsqueeze(0).unsqueeze(0).expand(1, n_cells, -1)
                s_rate = s_rate.unsqueeze(0).unsqueeze(0).expand(1, n_cells, -1)

            # Ensure observations are integers for Poisson distribution
            u_obs_int = u_obs.round().long()
            s_obs_int = s_obs.round().long()

            # Create Poisson distributions
            u_dist = pyro.distributions.Poisson(rate=u_rate)
            s_dist = pyro.distributions.Poisson(rate=s_rate)

            # Observe data
            # We need to handle the case where u_rate and s_rate have batch dimensions
            # but u_obs and s_obs don't
            if u_rate.dim() == 3:
                # u_rate shape: [batch_size, num_cells, num_genes]
                # u_obs shape: [num_cells, num_genes]
                # We need to expand u_obs to match u_rate
                batch_size = u_rate.shape[0]
                u_obs_expanded = u_obs_int.unsqueeze(0).expand(batch_size, -1, -1)
                s_obs_expanded = s_obs_int.unsqueeze(0).expand(batch_size, -1, -1)

                # Use pyro.plate to handle the dimensions
                with pyro.plate("batch", batch_size, dim=-3):
                    with pyro.plate("cells_likelihood", n_cells, dim=-2):
                        with pyro.plate("genes_likelihood", n_genes, dim=-1):
                            # Observe data
                            pyro.sample("u_obs", u_dist, obs=u_obs_expanded)
                            pyro.sample("s_obs", s_dist, obs=s_obs_expanded)
            else:
                # Use pyro.plate to handle the dimensions
                with pyro.plate("cells_likelihood", n_cells, dim=-2):
                    with pyro.plate("genes_likelihood", n_genes, dim=-1):
                        # Observe data
                        pyro.sample("u_obs", u_dist, obs=u_obs_int)
                        pyro.sample("s_obs", s_dist, obs=s_obs_int)

            # Add distributions to context
            context["u_dist"] = u_dist
            context["s_dist"] = s_dist

            return context
        else:
            # If validation failed, raise an error
            raise ValueError(f"Error in likelihood model forward pass: {validation_result.error}")

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
            Dictionary of likelihood distributions
        """
        if adata is None and cell_state is None:
            raise ValueError("Either adata or cell_state must be provided")

        if adata is not None:
            return self._generate_distributions_from_adata(
                adata, gene_offset, time_info, **kwargs
            )
        else:
            return self._generate_distributions_from_cell_state(
                cell_state, gene_offset, time_info, **kwargs
            )

    def _generate_distributions_from_adata(
        self,
        adata: AnnData,
        gene_offset: Optional[torch.Tensor] = None,
        time_info: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, pyro.distributions.Distribution]:
        """Generate Poisson likelihood distributions from AnnData.

        Args:
            adata: AnnData object containing gene expression data
            gene_offset: Optional gene-specific offset factors
            time_info: Optional time-related information
            **kwargs: Additional keyword arguments

        Returns:
            Dictionary of likelihood distributions
        """
        # Check if cell_state is provided in kwargs
        cell_state = kwargs.get("cell_state")

        if cell_state is not None:
            # If cell_state is provided, use it to generate distributions
            return self._generate_distributions_from_cell_state(
                cell_state, gene_offset, time_info, n_genes=adata.n_vars, **kwargs
            )

        # Otherwise, use AnnData to generate distributions
        # Extract counts from AnnData
        if "X" not in adata.layers:
            raise ValueError("AnnData object must have 'X' in layers")

        # Convert to torch tensor
        # Handle both sparse and dense arrays
        layer_data = adata.layers["X"]
        if hasattr(layer_data, 'toarray'):
            # For sparse matrices
            counts = torch.tensor(layer_data.toarray(), dtype=torch.float32)
        else:
            # For numpy arrays
            counts = torch.tensor(layer_data, dtype=torch.float32)

        # Use mean expression as rate parameter
        rate = torch.mean(counts, dim=0)

        # Check if we need to expand the rate to match a batch dimension
        # This is needed for the tests that expect a batch dimension
        batch_size = kwargs.get("batch_size")
        if batch_size is not None:
            rate = rate.unsqueeze(0).expand(batch_size, -1)

        # Apply gene offset if provided
        if gene_offset is not None:
            # If gene_offset has a batch dimension, use it directly
            if gene_offset.dim() > 1:
                rate = rate.expand(gene_offset.shape[0], -1) * gene_offset
            else:
                rate = rate * gene_offset

        # Create Poisson likelihood distribution
        return {"obs_counts": pyro.distributions.Poisson(rate=rate)}

    def _generate_distributions_from_cell_state(
        self,
        cell_state: torch.Tensor,
        gene_offset: Optional[torch.Tensor] = None,
        time_info: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, pyro.distributions.Distribution]:
        """Generate Poisson likelihood distributions from cell state.

        Args:
            cell_state: Latent cell state vectors
            gene_offset: Optional gene-specific offset factors
            time_info: Optional time-related information
            **kwargs: Additional keyword arguments

        Returns:
            Dictionary of likelihood distributions
        """
        # Get projection matrix from kwargs or create a random one
        projection = kwargs.get("projection")
        if projection is None:
            # Create a random projection matrix
            n_latent = cell_state.shape[1]
            n_genes = kwargs.get("n_genes", 100)
            projection = torch.randn(n_latent, n_genes)

        # Get gene scale from kwargs or use default
        gene_scale = kwargs.get("gene_scale", torch.ones(projection.shape[1]))

        # Get batch size from cell_state
        batch_size = cell_state.shape[0]

        # Apply gene offset if not provided
        if gene_offset is None:
            gene_offset = torch.ones((batch_size, projection.shape[1]))
        elif gene_offset.dim() == 1:
            # If gene_offset is 1D, expand it to match batch size
            gene_offset = gene_offset.unsqueeze(0).expand(batch_size, -1)

        # Ensure gene_scale has the right shape
        if gene_scale.dim() == 1:
            gene_scale = gene_scale.unsqueeze(0).expand(batch_size, -1)

        # Project cell_state to gene space
        projected_state = torch.matmul(
            cell_state, projection
        )  # Shape: (batch_size, n_genes)

        # Calculate rate parameter
        rate = torch.exp(projected_state) * gene_scale * gene_offset

        # Create Poisson likelihood distribution
        return {"obs_counts": pyro.distributions.Poisson(rate=rate)}



