"""
Protocol-First inference guide implementations for PyroVelocity's modular architecture.

This module contains inference guide implementations that directly implement the
InferenceGuide Protocol. These implementations follow the Protocol-First approach,
which embraces composition over inheritance and allows for more flexible component composition.

This module has been simplified to include only the essential components needed for
validation against the legacy implementation:
- AutoGuideFactory: Factory for creating AutoGuides for PyroVelocity models
- LegacyAutoGuideFactory: Factory for creating AutoGuideList guides that replicate the legacy PyroVelocity guide
"""

from typing import Any, Callable, Dict, List, Optional, Union

import pyro
import torch
from beartype import beartype
from pyro import poutine
from pyro.infer import Predictive, autoguide
from pyro.infer.autoguide import (
    AutoLowRankMultivariateNormal,
    AutoNormal,
    init_to_median,
)
from pyro.infer.autoguide.guides import AutoGuideList

from pyrovelocity.models.modular.interfaces import InferenceGuide
from pyrovelocity.models.modular.registry import inference_guide_registry


@inference_guide_registry.register("auto")
class AutoGuideFactory:
    """Factory for creating AutoGuides for PyroVelocity models.

    This class provides a factory for creating different types of AutoGuides
    based on the specified guide type. It supports various AutoGuide types
    provided by Pyro, such as AutoNormal, AutoDiagonalNormal, etc.

    This implementation directly implements the InferenceGuide Protocol.

    Attributes:
        guide_type: The type of AutoGuide to create.
        init_loc_fn: Function to initialize the location parameters of the guide.
        init_scale: Initial scale for the guide parameters.
        name: A unique name for this component instance.
    """

    @beartype
    def __init__(
        self,
        guide_type: str = "AutoLowRankMultivariateNormal",
        init_loc_fn: Optional[Callable] = None,
        init_scale: float = 0.1,  # Match legacy model's default scale
        name: str = "inference_guide",
        **kwargs: Any,
    ) -> None:
        """Initialize the AutoGuideFactory.

        The default parameters are set to match the legacy implementation
        to ensure consistent behavior.

        Args:
            guide_type: The type of AutoGuide to create.
            init_loc_fn: Function to initialize the location parameters of the guide.
            init_scale: Initial scale for the guide parameters.
            name: A unique name for this component instance.
            **kwargs: Additional keyword arguments.
        """
        self.name = name
        self.guide_type = guide_type
        self.init_loc_fn = init_loc_fn or init_to_median
        self.init_scale = init_scale
        self._guide = None
        self._model = None

    @beartype
    def create_guide(
        self, model: Union[Callable, Any], **kwargs: Any
    ) -> autoguide.AutoGuide:
        """Create an AutoGuide for the given model.

        Args:
            model: The model function for which to create a guide.
            **kwargs: Additional keyword arguments.

        Returns:
            An AutoGuide instance.

        Raises:
            ValueError: If the specified guide_type is not supported.
        """
        # Store the model for later use
        self._model = model

        # Map guide_type to the corresponding AutoGuide class
        guide_classes = {
            "AutoNormal": autoguide.AutoNormal,
            "AutoDiagonalNormal": autoguide.AutoDiagonalNormal,
            "AutoMultivariateNormal": autoguide.AutoMultivariateNormal,
            "AutoLowRankMultivariateNormal": autoguide.AutoLowRankMultivariateNormal,
            "AutoIAFNormal": autoguide.AutoIAFNormal,
            "AutoLaplaceApproximation": autoguide.AutoLaplaceApproximation,
            "AutoDelta": autoguide.AutoDelta,
        }

        if self.guide_type not in guide_classes:
            raise ValueError(
                f"Unsupported guide type: {self.guide_type}. "
                f"Supported types are: {list(guide_classes.keys())}"
            )

        # Create the guide
        guide_cls = guide_classes[self.guide_type]

        # Different guide types accept different parameters
        if self.guide_type == "AutoDelta":
            self._guide = guide_cls(
                model,
                init_loc_fn=self.init_loc_fn,
            )
        else:
            self._guide = guide_cls(
                model,
                init_loc_fn=self.init_loc_fn,
                init_scale=self.init_scale,
            )
        return self._guide

    @beartype
    def get_guide(self) -> autoguide.AutoGuide:
        """Get the created guide.

        Returns:
            The created AutoGuide instance.

        Raises:
            RuntimeError: If the guide has not been created yet.
        """
        if self._guide is None:
            raise RuntimeError(
                "Guide has not been created yet. Call create_guide first."
            )
        return self._guide

    @beartype
    def get_posterior(self) -> Dict[str, torch.Tensor]:
        """Get the posterior distribution.

        Returns:
            Dictionary of posterior samples.

        Raises:
            RuntimeError: If the guide has not been created yet.
        """
        if self._guide is None:
            raise RuntimeError(
                "Guide has not been created yet. Call create_guide first."
            )
        return self.sample_posterior()

    @beartype
    def sample_posterior(
        self,
        model: Optional[Callable] = None,
        guide: Optional[Callable] = None,
        num_samples: int = 1000,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """Sample from the posterior distribution.

        Args:
            model: Optional model function (uses stored model if None)
            guide: Optional guide function (uses stored guide if None)
            num_samples: Number of samples to draw
            **kwargs: Additional keyword arguments

        Returns:
            Dictionary of posterior samples

        Raises:
            RuntimeError: If guide has not been created yet
        """
        if self._guide is None and guide is None:
            raise RuntimeError(
                "Guide has not been created yet. Call create_guide first."
            )

        # Use stored model/guide if not provided
        if model is None:
            model = self._model
        if guide is None:
            guide = self._guide

        # Create a predictive object
        predictive = Predictive(
            model=model,
            guide=guide,
            num_samples=num_samples,
            return_sites=kwargs.get("return_sites", None),
        )

        # Sample from the posterior
        posterior_samples = predictive()

        # Filter out non-sample sites
        posterior_samples = {
            k: v
            for k, v in posterior_samples.items()
            if not k.startswith("_") and k != "obs"
        }

        return posterior_samples

    def __call__(self, *args: Any, **kwargs: Any) -> Callable:
        """
        Create a guide function for the given model.

        This method is called by Pyro's SVI when the guide is used directly
        in svi.step() as the guide parameter. It should delegate to the guide
        object created by create_guide.

        Args:
            *args: Additional positional arguments (first is model when used with SVI)
            **kwargs: Additional keyword arguments

        Returns:
            A guide function compatible with the model
        """
        # If being used directly in SVI, the first argument will be the model
        if len(args) > 0 and callable(args[0]) and self._guide is None:
            model = args[0]
            self.create_guide(model)
            self._model = model

        # If we have a guide, delegate to it
        if self._guide is not None:
            return self._guide(*args, **kwargs)

        # Otherwise, raise an error
        raise RuntimeError(
            "Guide has not been created yet. Call create_guide first."
        )





@inference_guide_registry.register("legacy_auto")
class LegacyAutoGuideFactory:
    """Factory for creating AutoGuideList guides that replicate the legacy PyroVelocity guide.

    This class creates an AutoGuideList with two components:
    1. AutoNormal for cell-specific parameters (cell_time, u_read_depth, s_read_depth, etc.)
    2. AutoLowRankMultivariateNormal for gene-specific parameters (alpha, beta, gamma, etc.)

    It uses poutine.block to isolate parameters for each guide component, exactly matching
    the approach used in the legacy PyroVelocity implementation.

    Attributes:
        add_offset: Whether to include offset parameters in the guide
        init_scale: Initial scale for the guide parameters
        rank: Rank for the AutoLowRankMultivariateNormal guide
        name: A unique name for this component instance
    """

    @beartype
    def __init__(
        self,
        add_offset: bool = False,
        init_scale: float = 0.1,
        rank: int = 10,
        name: str = "legacy_auto_guide",
        **kwargs: Any,
    ) -> None:
        """Initialize the LegacyAutoGuideFactory.

        Args:
            add_offset: Whether to include offset parameters in the guide
            init_scale: Initial scale for the guide parameters
            rank: Rank for the AutoLowRankMultivariateNormal guide
            name: A unique name for this component instance
            **kwargs: Additional keyword arguments
        """
        self.name = name
        self.add_offset = add_offset
        self.init_scale = init_scale
        self.rank = rank
        self._guide = None
        self._model = None

    @beartype
    def forward(
        self,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Create a guide function for the model.

        This method implements the InferenceGuide Protocol's forward method.
        It takes a context dictionary containing the model function,
        creates an AutoGuideList guide, and updates the context with the guide function.

        Args:
            context: Dictionary containing model context with the following required keys:
                - model: The model function

        Returns:
            Updated context dictionary with the following additional keys:
                - guide: The guide function
        """
        # Extract model from context
        model = context.get("model")
        if model is None:
            raise ValueError("Model must be provided in the context")

        # Create the guide
        guide = self.create_guide(model)

        # Update context with guide function
        context["guide"] = guide

        return context

    @beartype
    def create_guide(
        self, model: Union[Callable, Any], **kwargs: Any
    ) -> AutoGuideList:
        """Create an AutoGuideList for the given model.

        This method creates an AutoGuideList with two components:
        1. AutoNormal for cell-specific parameters
        2. AutoLowRankMultivariateNormal for gene-specific parameters

        Args:
            model: The model function for which to create a guide
            **kwargs: Additional keyword arguments

        Returns:
            An AutoGuideList instance
        """
        # Store the model for later use
        self._model = model

        # Create the AutoGuideList
        guide = AutoGuideList(model, create_plates=getattr(model, "create_plates", None))

        # First guide component - AutoNormal for cell-specific parameters
        guide.append(
            AutoNormal(
                poutine.block(
                    model,
                    expose=[
                        "cell_time",
                        "u_read_depth",
                        "s_read_depth",
                        "kinetics_prob",
                        "kinetics_weights",
                    ],
                ),
                init_scale=self.init_scale,
            )
        )

        # Second guide component - AutoLowRankMultivariateNormal for gene-specific parameters
        if self.add_offset:
            guide.append(
                AutoLowRankMultivariateNormal(
                    poutine.block(
                        model,
                        expose=[
                            "alpha",
                            "beta",
                            "gamma",
                            "dt_switching",
                            "t0",
                            "u_scale",
                            "s_scale",
                            "u_offset",
                            "s_offset",
                        ],
                    ),
                    rank=self.rank,
                    init_scale=self.init_scale,
                )
            )
        else:
            guide.append(
                AutoLowRankMultivariateNormal(
                    poutine.block(
                        model,
                        expose=[
                            "alpha",
                            "beta",
                            "gamma",
                            "dt_switching",
                            "t0",
                            "u_scale",
                            "s_scale",
                        ],
                    ),
                    rank=self.rank,
                    init_scale=self.init_scale,
                )
            )

        # Modify the guide to match the legacy model's parameter shapes
        # This is a critical step to ensure compatibility with the legacy model
        # The legacy model uses shapes like (num_samples, 1, num_genes) for gene parameters
        # and (num_samples, num_cells, 1) for cell parameters

        # We need to modify the guide to match these shapes
        # This is done by wrapping the guide's __call__ method
        original_call = guide.__call__

        def wrapped_call(*args, **kwargs):
            # Call the original guide to get the samples
            result = original_call(*args, **kwargs)

            # Reshape the samples to match the legacy model's shapes
            # For gene parameters, reshape from (num_samples, num_genes) to (num_samples, 1, num_genes)
            # For cell parameters, reshape from (num_samples, num_cells) to (num_samples, num_cells, 1)

            # Get the number of samples and genes
            num_samples = next(iter(result.values())).shape[0]
            num_genes = 0
            num_cells = 0

            # Find the number of genes and cells from the result
            for key, value in result.items():
                if key in ["alpha", "beta", "gamma", "dt_switching", "t0", "u_scale", "s_scale"]:
                    if len(value.shape) > 1:
                        num_genes = value.shape[1]
                elif key in ["cell_time", "u_read_depth", "s_read_depth"]:
                    if len(value.shape) > 1:
                        num_cells = value.shape[1]

            # Reshape gene parameters
            for key in ["alpha", "beta", "gamma", "dt_switching", "t0", "u_scale", "s_scale"]:
                if key in result and num_genes > 0:
                    # Reshape from (num_samples, num_genes) to (num_samples, 1, num_genes)
                    result[key] = result[key].reshape(num_samples, 1, num_genes)

            # Reshape cell parameters
            for key in ["cell_time", "u_read_depth", "s_read_depth"]:
                if key in result and num_cells > 0:
                    # Reshape from (num_samples, num_cells) to (num_samples, num_cells, 1)
                    result[key] = result[key].reshape(num_samples, num_cells, 1)

            return result

        # Replace the guide's __call__ method with our wrapped version
        guide.__call__ = wrapped_call

        self._guide = guide
        return guide

    @beartype
    def get_guide(self) -> AutoGuideList:
        """Get the created guide.

        Returns:
            The created AutoGuideList instance.

        Raises:
            RuntimeError: If the guide has not been created yet.
        """
        if self._guide is None and self._model is not None:
            # If we have a model but no guide, create the guide
            self.create_guide(self._model)

        if self._guide is None:
            raise RuntimeError(
                "Guide has not been created yet. Call create_guide first."
            )
        return self._guide

    @beartype
    def get_posterior(self) -> Dict[str, torch.Tensor]:
        """Get the posterior distribution.

        Returns:
            Dictionary of posterior samples.

        Raises:
            RuntimeError: If the guide has not been created yet.
        """
        if self._guide is None:
            raise RuntimeError(
                "Guide has not been created yet. Call create_guide first."
            )
        return self.sample_posterior()

    @beartype
    def sample_posterior(
        self,
        model: Optional[Callable] = None,
        guide: Optional[Callable] = None,
        num_samples: int = 1000,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """Sample from the posterior distribution.

        Args:
            model: Optional model function (uses stored model if None)
            guide: Optional guide function (uses stored guide if None)
            num_samples: Number of samples to draw
            **kwargs: Additional keyword arguments

        Returns:
            Dictionary of posterior samples

        Raises:
            RuntimeError: If guide has not been created yet
        """
        if self._guide is None and guide is None and self._model is not None:
            # If we have a model but no guide, create the guide
            self.create_guide(self._model)

        if self._guide is None and guide is None:
            raise RuntimeError(
                "Guide has not been created yet. Call create_guide first."
            )

        # Use stored model/guide if not provided
        if model is None:
            model = self._model
        if guide is None:
            guide = self._guide

        # Create a predictive object
        predictive = Predictive(
            model=model,
            guide=guide,
            num_samples=num_samples,
            return_sites=kwargs.get("return_sites", None),
        )

        # Sample from the posterior
        posterior_samples = predictive()

        # Filter out non-sample sites
        posterior_samples = {
            k: v
            for k, v in posterior_samples.items()
            if not k.startswith("_") and k != "obs"
        }

        # Reshape the samples to match the legacy model's shapes
        # For gene parameters, reshape from (num_samples, num_genes) to (num_samples, 1, num_genes)
        # For cell parameters, reshape from (num_samples, num_cells) to (num_samples, num_cells, 1)

        # Get the number of genes and cells from the result
        num_genes = 0
        num_cells = 0

        # Find the number of genes and cells from the result
        for key, value in posterior_samples.items():
            if key in ["alpha", "beta", "gamma", "dt_switching", "t0", "u_scale", "s_scale"]:
                if len(value.shape) > 1:
                    num_genes = value.shape[1]
            elif key in ["cell_time", "u_read_depth", "s_read_depth"]:
                if len(value.shape) > 1:
                    num_cells = value.shape[1]

        # Reshape gene parameters
        for key in ["alpha", "beta", "gamma", "dt_switching", "t0", "u_scale", "s_scale"]:
            if key in posterior_samples and num_genes > 0:
                # Reshape from (num_samples, num_genes) to (num_samples, 1, num_genes)
                posterior_samples[key] = posterior_samples[key].reshape(num_samples, 1, num_genes)

        # Reshape cell parameters
        for key in ["cell_time", "u_read_depth", "s_read_depth"]:
            if key in posterior_samples and num_cells > 0:
                # Reshape from (num_samples, num_cells) to (num_samples, num_cells, 1)
                posterior_samples[key] = posterior_samples[key].reshape(num_samples, num_cells, 1)

        # Add deterministic sites for ut and st
        # These are critical for velocity calculation in the legacy model
        if "alpha" in posterior_samples and "beta" in posterior_samples and "gamma" in posterior_samples:
            # Get the parameters
            alpha = posterior_samples["alpha"]
            beta = posterior_samples["beta"]
            gamma = posterior_samples["gamma"]

            # Calculate steady state values
            u_inf = alpha / beta
            s_inf = alpha / gamma

            # Add to posterior samples
            posterior_samples["u_inf"] = u_inf
            posterior_samples["s_inf"] = s_inf

            # If cell_time is available, calculate ut and st
            if "cell_time" in posterior_samples:
                cell_time = posterior_samples["cell_time"]

                # Calculate switching time (t0 + dt_switching)
                t0 = posterior_samples.get("t0", torch.zeros_like(alpha))
                dt_switching = posterior_samples.get("dt_switching", torch.zeros_like(alpha))
                switching = t0 + dt_switching

                # Add switching to posterior samples
                posterior_samples["switching"] = switching

                # Calculate ut and st based on the transcription model
                # This is a simplified version of the calculation in the legacy model
                # For cells before switching time
                before_switching = cell_time < switching

                # Initialize ut and st with zeros
                ut = torch.zeros_like(cell_time.expand(-1, -1, num_genes))
                st = torch.zeros_like(cell_time.expand(-1, -1, num_genes))

                # Calculate ut and st for cells before switching
                ut = u_inf * (1 - torch.exp(-beta * cell_time))
                st = s_inf * (1 - torch.exp(-gamma * cell_time)) - (
                    alpha / (gamma - beta)
                ) * (torch.exp(-beta * cell_time) - torch.exp(-gamma * cell_time))

                # Add ut and st to posterior samples
                posterior_samples["ut"] = ut
                posterior_samples["st"] = st

                # Add u and s (observed values) to posterior samples
                # These are just copies of ut and st for now
                posterior_samples["u"] = ut
                posterior_samples["s"] = st

        return posterior_samples

    def __call__(self, *args: Any, **kwargs: Any) -> Callable:
        """
        Create a guide function for the given model.

        This method is called by Pyro's SVI when the guide is used directly
        in svi.step() as the guide parameter. It should delegate to the guide
        object created by create_guide.

        Args:
            *args: Additional positional arguments (first is model when used with SVI)
            **kwargs: Additional keyword arguments

        Returns:
            A guide function compatible with the model
        """
        # If being used directly in SVI, the first argument will be the model
        if len(args) > 0 and callable(args[0]) and self._guide is None:
            model = args[0]
            self.create_guide(model)
            self._model = model

        # If we have a guide, delegate to it
        if self._guide is not None:
            return self._guide(*args, **kwargs)

        # Otherwise, raise an error
        raise RuntimeError(
            "Guide has not been created yet. Call create_guide first."
        )