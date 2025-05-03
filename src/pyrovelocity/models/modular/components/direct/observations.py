"""
Protocol-First observation model implementations for PyroVelocity's modular architecture.

This module contains observation model implementations that directly implement the
ObservationModel Protocol without inheriting from BaseObservationModel. These implementations
follow the Protocol-First approach, which embraces composition over inheritance
and allows for more flexible component composition.
"""

from typing import Any, Dict, Tuple

import numpy as np
import pyro
import pyro.distributions as dist
import torch
from anndata import AnnData
from beartype import beartype
from jaxtyping import Array, Float, Int

from pyrovelocity.models.modular.interfaces import ObservationModel
from pyrovelocity.models.modular.registry import observation_model_registry
from pyrovelocity.models.modular.utils.context_utils import validate_context


@observation_model_registry.register("standard_direct")
class StandardObservationModelDirect:
    """Standard observation model for RNA velocity.

    This model defines the likelihood of observed data (spliced and unspliced counts)
    given the latent variables. It uses negative binomial distributions for both
    spliced and unspliced counts, with appropriate transformations of the latent
    variables to generate the parameters of these distributions.

    This implementation directly implements the ObservationModel Protocol
    without inheriting from BaseObservationModel.

    Attributes:
        use_observed_lib_size: Whether to use observed library size as a scaling factor.
        transform_batch: Whether to transform batch effects.
        batch_size: Batch size for data loaders.
        name: A unique name for this component instance.
    """

    def __init__(
        self,
        use_observed_lib_size: bool = True,
        transform_batch: bool = True,
        batch_size: int = 128,
        name: str = "observation_model_direct",
        **kwargs: Any,
    ) -> None:
        """Initialize the StandardObservationModelDirect.

        Args:
            use_observed_lib_size: Whether to use observed library size as a scaling factor.
            transform_batch: Whether to transform batch effects.
            batch_size: Batch size for data loaders.
            name: A unique name for this component instance.
            **kwargs: Additional keyword arguments.
        """
        self.name = name
        self.use_observed_lib_size = use_observed_lib_size
        self.transform_batch = transform_batch
        self.batch_size = batch_size
        
        # Initialize instance variables that will be set later
        self.u_obs = None
        self.s_obs = None
        self.u_scale = None
        self.s_scale = None

    @beartype
    def forward(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform observed data for the model.

        Args:
            context: Dictionary containing model context including u_obs, s_obs, and other parameters

        Returns:
            Updated context dictionary with transformed data
        """
        # Intentional duplication: Context validation
        # This functionality might be extracted to a utility function in the future
        validation_result = validate_context(
            self.__class__.__name__,
            context,
            required_keys=["u_obs", "s_obs"],
            tensor_keys=["u_obs", "s_obs"],
        )
        
        if isinstance(validation_result, dict):
            # Extract required values from context
            u_obs = context["u_obs"]
            s_obs = context["s_obs"]
            
            # Make a copy of the context to avoid modifying the original
            context_copy = {k: v for k, v in context.items() if k not in ["u_obs", "s_obs"]}
            
            # Calculate library size
            u_lib_size = u_obs.sum(1).unsqueeze(1).float()  # Convert to float
            s_lib_size = s_obs.sum(1).unsqueeze(1).float()  # Convert to float
            
            # Normalize by library size if specified
            if self.use_observed_lib_size:
                u_scale = u_lib_size / u_lib_size.mean()
                s_scale = s_lib_size / s_lib_size.mean()
            else:
                u_scale = torch.ones_like(u_lib_size)
                s_scale = torch.ones_like(s_lib_size)
            
            # Check if there's a shape mismatch between model parameters and data
            model_n_genes = None
            for param in ["alpha", "beta", "gamma"]:
                if param in context and isinstance(context[param], torch.Tensor):
                    model_n_genes = context[param].shape[-1]
                    break
            
            # If there's a shape mismatch between model parameters and data,
            # handle it by reshaping the data to match the model parameters
            if model_n_genes is not None and model_n_genes != u_obs.shape[1]:
                # Determine the minimum number of genes to use
                n_genes = min(model_n_genes, u_obs.shape[1])
                
                # Reshape u_obs and s_obs to match the model parameters
                if u_obs.shape[1] > n_genes:
                    u_obs = u_obs[:, :n_genes]
                
                if s_obs.shape[1] > n_genes:
                    s_obs = s_obs[:, :n_genes]
            
            # Store the observations and scaling factors
            self.u_obs = u_obs
            self.s_obs = s_obs
            self.u_scale = u_scale
            self.s_scale = s_scale
            
            # Update the context with the transformed data
            context["u_obs"] = u_obs
            context["s_obs"] = s_obs
            context["u_scale"] = u_scale
            context["s_scale"] = s_scale
            
            return context
        else:
            # If validation failed, raise an error
            raise ValueError(f"Error in observation model forward pass: {validation_result.error}")

    @beartype
    def setup_observation(
        self,
        adata: AnnData,
        num_cells: int,
        num_genes: int,
        plate_size: int,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Set up the observation model parameters.

        Args:
            adata: AnnData object containing the data.
            num_cells: Number of cells.
            num_genes: Number of genes.
            plate_size: Size of the plate for Pyro's plate context.
            **kwargs: Additional keyword arguments.

        Returns:
            Dictionary of observation model parameters.
        """
        # Extract necessary data from AnnData
        u_obs = torch.tensor(
            adata.layers["unspliced"].toarray()
            if isinstance(adata.layers["unspliced"], np.ndarray) == False
            else adata.layers["unspliced"]
        )
        s_obs = torch.tensor(
            adata.layers["spliced"].toarray()
            if isinstance(adata.layers["spliced"], np.ndarray) == False
            else adata.layers["spliced"]
        )
        
        # Calculate library size
        u_lib_size = u_obs.sum(1).unsqueeze(1).float()  # Convert to float
        s_lib_size = s_obs.sum(1).unsqueeze(1).float()  # Convert to float
        
        # Normalize by library size if specified
        if self.use_observed_lib_size:
            u_scale = u_lib_size / u_lib_size.mean()
            s_scale = s_lib_size / s_lib_size.mean()
        else:
            u_scale = torch.ones_like(u_lib_size)
            s_scale = torch.ones_like(s_lib_size)
        
        # Store parameters
        self.u_obs = u_obs
        self.s_obs = s_obs
        self.u_scale = u_scale
        self.s_scale = s_scale
        
        return {
            "u_obs": u_obs,
            "s_obs": s_obs,
            "u_scale": u_scale,
            "s_scale": s_scale,
        }

    @beartype
    def model_obs(
        self,
        idx: Int[Array, "batch"],
        u_scale: Float[torch.Tensor, "num_cells 1"],
        s_scale: Float[torch.Tensor, "num_cells 1"],
        u_log_rate: Float[torch.Tensor, "num_cells num_genes"],
        s_log_rate: Float[torch.Tensor, "num_cells num_genes"],
        u_log_r: Float[torch.Tensor, "num_genes"],
        s_log_r: Float[torch.Tensor, "num_genes"],
        cell_plate: pyro.plate,
        gene_plate: pyro.plate,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Define the observation model within the Pyro model context.

        Args:
            idx: Indices of cells in the mini-batch.
            u_scale: Scaling factors for unspliced counts.
            s_scale: Scaling factors for spliced counts.
            u_log_rate: Log rates for unspliced counts.
            s_log_rate: Log rates for spliced counts.
            u_log_r: Log dispersion parameters for unspliced counts.
            s_log_r: Log dispersion parameters for spliced counts.
            cell_plate: Pyro plate for cells.
            gene_plate: Pyro plate for genes.
            **kwargs: Additional keyword arguments.

        Returns:
            Tuple of observed unspliced and spliced counts.
        """
        with cell_plate as ind:
            # Get cell indices for the mini-batch
            i = ind[idx]
            
            # Apply scaling factors
            u_rate = torch.exp(u_log_rate[i]) * u_scale[i]
            s_rate = torch.exp(s_log_rate[i]) * s_scale[i]
            
            with gene_plate:
                # Define negative binomial distributions for unspliced and spliced counts
                u_dist = dist.GammaPoisson(
                    concentration=torch.exp(u_log_r),
                    rate=torch.exp(u_log_r) / u_rate,
                )
                s_dist = dist.GammaPoisson(
                    concentration=torch.exp(s_log_r),
                    rate=torch.exp(s_log_r) / s_rate,
                )
                
                # Sample from the distributions
                u = pyro.sample("u", u_dist, obs=self.u_obs[i])
                s = pyro.sample("s", s_dist, obs=self.s_obs[i])
        
        return u, s

    @beartype
    def guide_obs(
        self,
        idx: Int[Array, "batch"],
        cell_plate: pyro.plate,
        gene_plate: pyro.plate,
        **kwargs: Any,
    ) -> None:
        """Define the observation guide within the Pyro guide context.

        For the standard observation model, the guide is empty as the parameters
        are learned through the AutoGuide mechanism.

        Args:
            idx: Indices of cells in the mini-batch.
            cell_plate: Pyro plate for cells.
            gene_plate: Pyro plate for genes.
            **kwargs: Additional keyword arguments.
        """
        # The standard observation model doesn't need a custom guide
        pass

    @beartype
    def prepare_data(
        self,
        adata: AnnData,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Prepare data for the model.

        Args:
            adata: AnnData object containing the data.
            **kwargs: Additional keyword arguments.

        Returns:
            Dictionary of prepared data.
        """
        # Extract necessary data from AnnData
        u_obs = torch.tensor(
            adata.layers["unspliced"].toarray()
            if isinstance(adata.layers["unspliced"], np.ndarray) == False
            else adata.layers["unspliced"]
        )
        s_obs = torch.tensor(
            adata.layers["spliced"].toarray()
            if isinstance(adata.layers["spliced"], np.ndarray) == False
            else adata.layers["spliced"]
        )
        
        # Calculate library size
        u_lib_size = u_obs.sum(1).unsqueeze(1).float()  # Convert to float
        s_lib_size = s_obs.sum(1).unsqueeze(1).float()  # Convert to float
        
        # Normalize by library size if specified
        if self.use_observed_lib_size:
            u_scale = u_lib_size / u_lib_size.mean()
            s_scale = s_lib_size / s_lib_size.mean()
        else:
            u_scale = torch.ones_like(u_lib_size)
            s_scale = torch.ones_like(s_lib_size)
        
        # Create cell indices
        cell_indices = torch.arange(u_obs.shape[0])
        
        return {
            "u_obs": u_obs,
            "s_obs": s_obs,
            "u_scale": u_scale,
            "s_scale": s_scale,
            "cell_indices": cell_indices,
        }
