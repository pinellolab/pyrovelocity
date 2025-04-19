"""Observation models for PyroVelocity.

This module contains implementations of observation models that define how
latent variables are mapped to observed data in the PyroVelocity framework.
Observation models are responsible for defining the likelihood of observed
data given the latent variables.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import jax.numpy as jnp
import numpy as np
import pyro
import pyro.distributions as dist
import torch
from anndata import AnnData
from beartype import beartype
from jaxtyping import Array, Float, Int
from torch.utils.data import DataLoader, TensorDataset

from pyrovelocity.models.components.base import BaseObservationModel
from pyrovelocity.models.interfaces import ObservationModel
from pyrovelocity.models.registry import observation_model_registry


@observation_model_registry.register("standard")
class StandardObservationModel(BaseObservationModel):
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
        super().__init__(**kwargs)
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
    def _prepare_data_impl(
        self, adata: AnnData, **kwargs: Any
    ) -> Dict[str, Union[torch.Tensor, jnp.ndarray]]:
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
        cell_indices = data["cell_indices"]

        # Create dataset
        dataset = TensorDataset(cell_indices, u_obs, s_obs, u_scale, s_scale)

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
    ) -> Dict[str, Union[torch.Tensor, jnp.ndarray]]:
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

        # If batch is a tuple from DataLoader, unpack it
        if isinstance(batch, tuple) or isinstance(batch, list):
            cell_indices, u_obs, s_obs, u_scale, s_scale = batch

        return {
            "cell_indices": cell_indices,
            "u_obs": u_obs,
            "s_obs": s_obs,
            "u_scale": u_scale,
            "s_scale": s_scale,
        }
