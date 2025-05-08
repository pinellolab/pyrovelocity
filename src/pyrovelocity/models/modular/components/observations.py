"""Observation models for PyroVelocity.

This module contains implementations of observation models that define how
latent variables are mapped to observed data in the PyroVelocity framework.
Observation models are responsible for defining the likelihood of observed
data given the latent variables.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

# Remove JAX dependency
import numpy as np
import pyro
import pyro.distributions as dist
import torch
from anndata import AnnData
from beartype import beartype
from jaxtyping import Array, Float, Int
from torch.utils.data import DataLoader, TensorDataset

from pyrovelocity.models.modular.constants import CELLS_DIM, GENES_DIM
from pyrovelocity.models.modular.interfaces import ObservationModel
from pyrovelocity.models.modular.registry import observation_model_registry


@observation_model_registry.register("standard")
class StandardObservationModel:
    """Standard observation model for RNA velocity.

    This model defines the likelihood of observed data (spliced and unspliced counts)
    given the latent variables. It uses negative binomial distributions for both
    spliced and unspliced counts, with appropriate transformations of the latent
    variables to generate the parameters of these distributions.

    Attributes:
        use_observed_lib_size: Whether to use observed library size as a scaling factor.
        transform_batch: Whether to transform batch effects.
    """

    @beartype
    def forward(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Forward pass through the observation model.

        Args:
            context: Dictionary containing model context

        Returns:
            Updated context dictionary with observation model outputs
        """
        # Extract u_obs and s_obs from context
        u_obs = context.get("u_obs")
        s_obs = context.get("s_obs")

        if u_obs is None or s_obs is None:
            # If x is provided but not u_obs and s_obs, extract u_obs and s_obs from x
            x = context.get("x")
            if x is not None:
                if isinstance(x, dict):
                    # If x is a dictionary, extract u_obs and s_obs directly
                    if "u_obs" in x and "s_obs" in x:
                        u_obs = x["u_obs"]
                        s_obs = x["s_obs"]
                        context["u_obs"] = u_obs
                        context["s_obs"] = s_obs
                    else:
                        raise ValueError(
                            "If x is a dictionary, it must contain 'u_obs' and 's_obs' keys"
                        )
                else:
                    # If x is a tensor, assume first half of features are u_obs and second half are s_obs
                    n_genes = x.shape[1] // 2
                    u_obs = x[:, :n_genes]
                    s_obs = x[:, n_genes:]
                    context["u_obs"] = u_obs
                    context["s_obs"] = s_obs
            else:
                raise ValueError(
                    "Either u_obs and s_obs or x must be provided in the context"
                )

        # Log tensor shapes for debugging
        print(f"StandardObservationModel - u_obs shape: {u_obs.shape}")
        print(f"StandardObservationModel - s_obs shape: {s_obs.shape}")

        # Check for model parameters in context that might have different shapes
        model_n_genes = None
        for param_name in ["alpha", "beta", "gamma"]:
            if param_name in context:
                model_n_genes = context[param_name].shape[0]
                print(f"StandardObservationModel - Found {param_name} with shape: {context[param_name].shape}")
                break

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

        # If there's a shape mismatch between model parameters and data,
        # handle it by reshaping the data to match the model parameters
        if model_n_genes is not None and model_n_genes != u_obs.shape[1]:
            print(f"WARNING: Shape mismatch between model parameters ({model_n_genes} genes) and data ({u_obs.shape[1]} genes)")

            # Determine the minimum number of genes to use
            n_genes = min(model_n_genes, u_obs.shape[1])
            print(f"StandardObservationModel - Reshaping data to use {n_genes} genes")

            # Reshape u_obs and s_obs to match the model parameters
            if u_obs.shape[1] > n_genes:
                print(f"StandardObservationModel - Reshaping u_obs from {u_obs.shape} to use {n_genes} genes")
                u_obs = u_obs[:, :n_genes]

            if s_obs.shape[1] > n_genes:
                print(f"StandardObservationModel - Reshaping s_obs from {s_obs.shape} to use {n_genes} genes")
                s_obs = s_obs[:, :n_genes]

            print(f"StandardObservationModel - After reshaping: u_obs shape: {u_obs.shape}, s_obs shape: {s_obs.shape}")

        # Store the observations and scaling factors
        self.u_obs = u_obs
        self.s_obs = s_obs
        self.u_scale = u_scale
        self.s_scale = s_scale

        # Calculate read depths (library sizes) for use in dynamics model
        u_read_depth = u_lib_size
        s_read_depth = s_lib_size

        # Update the context with the transformed data
        context["u_obs"] = u_obs
        context["s_obs"] = s_obs
        context["u_scale"] = u_scale
        context["s_scale"] = s_scale
        context["u_read_depth"] = u_read_depth
        context["s_read_depth"] = s_read_depth

        return context

    @beartype
    def _forward_impl(
        self, u_obs: torch.Tensor, s_obs: torch.Tensor, **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Implementation of the forward transformation.

        Args:
            u_obs: Raw observed unspliced RNA counts
            s_obs: Raw observed spliced RNA counts
            **kwargs: Additional model-specific parameters

        Returns:
            Dictionary with transformed data
        """
        # Remove u_obs and s_obs from kwargs to avoid duplicate arguments
        if "u_obs" in kwargs:
            del kwargs["u_obs"]
        if "s_obs" in kwargs:
            del kwargs["s_obs"]
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

        # Store the observations and scaling factors
        self.u_obs = u_obs
        self.s_obs = s_obs
        self.u_scale = u_scale
        self.s_scale = s_scale

        # Return the transformed data
        return {
            "u_obs": u_obs,
            "s_obs": s_obs,
            "u_scale": u_scale,
            "s_scale": s_scale,
            "u_read_depth": u_lib_size,
            "s_read_depth": s_lib_size,
        }

    @beartype
    def __init__(
        self,
        use_observed_lib_size: bool = True,
        transform_batch: bool = True,
        batch_size: int = 128,
        **kwargs: Any,
    ) -> None:
        """Initialize the StandardObservationModel.

        Args:
            use_observed_lib_size: Whether to use observed library size as a scaling factor.
            transform_batch: Whether to transform batch effects.
            batch_size: Batch size for data loaders.
            **kwargs: Additional keyword arguments.
        """
        self.name = kwargs.get("name", "standard_observation_model")
        self.use_observed_lib_size = use_observed_lib_size
        self.transform_batch = transform_batch
        self.batch_size = batch_size

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
            "u_read_depth": u_lib_size,
            "s_read_depth": s_lib_size,
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
            u_scale: Scaling factor for unspliced counts.
            s_scale: Scaling factor for spliced counts.
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
        # Print shapes for debugging
        print(f"model_obs - idx shape: {idx.shape if hasattr(idx, 'shape') else 'scalar'}")
        print(f"model_obs - u_scale shape: {u_scale.shape}")
        print(f"model_obs - u_log_rate shape: {u_log_rate.shape}")
        print(f"model_obs - u_log_r shape: {u_log_r.shape}")

        # Validate that cell_plate and gene_plate use the correct dimensions
        if hasattr(cell_plate, 'dim') and cell_plate.dim != CELLS_DIM:
            print(f"WARNING: cell_plate dimension {cell_plate.dim} does not match CELLS_DIM {CELLS_DIM}")

        if hasattr(gene_plate, 'dim') and gene_plate.dim != GENES_DIM:
            print(f"WARNING: gene_plate dimension {gene_plate.dim} does not match GENES_DIM {GENES_DIM}")

        # Get cell indices for the mini-batch
        # Use a single cell_plate with dim=CELLS_DIM for cells
        with cell_plate as ind:
            i = ind[idx]

            # Apply scaling factors to rates
            # These operations should maintain proper dimensions
            u_rate = torch.exp(u_log_rate[i]) * u_scale[i]  # Shape: [batch_size, num_genes]
            s_rate = torch.exp(s_log_rate[i]) * s_scale[i]  # Shape: [batch_size, num_genes]

            # Use gene_plate with dim=GENES_DIM for genes
            # This ensures proper broadcasting between cell and gene dimensions
            with gene_plate:
                # Define negative binomial distributions for unspliced and spliced counts
                # The concentration parameter is gene-specific (dim=GENES_DIM)
                # The rate parameter combines cell and gene information
                u_dist = dist.GammaPoisson(
                    concentration=torch.exp(u_log_r),  # Shape: [num_genes]
                    rate=torch.exp(u_log_r) / u_rate,  # Shape: [batch_size, num_genes]
                )
                s_dist = dist.GammaPoisson(
                    concentration=torch.exp(s_log_r),  # Shape: [num_genes]
                    rate=torch.exp(s_log_r) / s_rate,  # Shape: [batch_size, num_genes]
                )

                # Sample from the distributions
                # The observations should have shape [batch_size, num_genes]
                u = pyro.sample("u_obs", u_dist, obs=self.u_obs[i])
                s = pyro.sample("s_obs", s_dist, obs=self.s_obs[i])

                # Print shapes for debugging
                print(f"model_obs - u shape: {u.shape}")
                print(f"model_obs - s shape: {s.shape}")

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
    def _prepare_data_impl(
        self, adata: AnnData, **kwargs: Any
    ) -> Dict[str, torch.Tensor]:
        """
        Implementation of data preparation.

        This method prepares data from an AnnData object for use in the model.
        It extracts unspliced and spliced counts and calculates scaling factors.

        Args:
            adata: AnnData object containing the data
            **kwargs: Additional keyword arguments

        Returns:
            Dictionary of prepared data
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
            "u_read_depth": u_lib_size,
            "s_read_depth": s_lib_size,
            "cell_indices": cell_indices,
        }

    @beartype
    def _create_dataloaders_impl(
        self, data: Dict[str, torch.Tensor], **kwargs: Any
    ) -> Dict[str, torch.utils.data.DataLoader]:
        """
        Implementation of data loader creation.

        This method creates data loaders from prepared data for use in training.

        Args:
            data: Dictionary of prepared data
            **kwargs: Additional keyword arguments

        Returns:
            Dictionary of data loaders
        """
        # Extract data
        u_obs = data["u_obs"]
        s_obs = data["s_obs"]
        u_scale = data["u_scale"]
        s_scale = data["s_scale"]
        u_read_depth = data["u_read_depth"]
        s_read_depth = data["s_read_depth"]
        cell_indices = data["cell_indices"]

        # Create dataset
        dataset = TensorDataset(cell_indices, u_obs, s_obs, u_scale, s_scale, u_read_depth, s_read_depth)

        # Create data loader
        batch_size = kwargs.get("batch_size", self.batch_size)
        shuffle = kwargs.get("shuffle", True)

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=False,
        )

        return {"train": dataloader}

    @beartype
    def _preprocess_batch_impl(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Implementation of batch preprocessing.

        This method preprocesses a batch of data for use in the model.

        Args:
            batch: Dictionary containing batch data

        Returns:
            Preprocessed batch data
        """
        # Extract batch data
        cell_indices = batch.get("cell_indices")
        u_obs = batch.get("u_obs")
        s_obs = batch.get("s_obs")
        u_scale = batch.get("u_scale")
        s_scale = batch.get("s_scale")
        u_read_depth = batch.get("u_read_depth")
        s_read_depth = batch.get("s_read_depth")

        # If batch is a tuple from DataLoader, unpack it
        if isinstance(batch, tuple) or isinstance(batch, list):
            cell_indices, u_obs, s_obs, u_scale, s_scale, u_read_depth, s_read_depth = batch

        return {
            "cell_indices": cell_indices,
            "u_obs": u_obs,
            "s_obs": s_obs,
            "u_scale": u_scale,
            "s_scale": s_scale,
            "u_read_depth": u_read_depth,
            "s_read_depth": s_read_depth,
        }
