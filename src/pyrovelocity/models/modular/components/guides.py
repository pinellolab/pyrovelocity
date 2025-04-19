"""Inference guides for PyroVelocity.

This module contains implementations of inference guides that define how
to perform approximate Bayesian inference in the PyroVelocity framework.
Guides are responsible for defining the variational distribution used to
approximate the posterior distribution of the model parameters.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pyro
import pyro.distributions as dist
import pyro.infer as infer
import torch
from anndata import AnnData
from beartype import beartype
from jaxtyping import Array, Float, Int

from pyrovelocity.models.modular.components.base import BaseInferenceGuide
from pyrovelocity.models.modular.interfaces import InferenceGuide
from pyrovelocity.models.modular.registry import inference_guide_registry


@inference_guide_registry.register("auto")
class AutoGuideFactory(BaseInferenceGuide):
    """Factory for creating AutoGuides for PyroVelocity models.

    This class provides a factory for creating different types of AutoGuides
    based on the specified guide type. It supports various AutoGuide types
    provided by Pyro, such as AutoNormal, AutoDiagonalNormal, etc.

    Attributes:
        guide_type: The type of AutoGuide to create.
        init_loc_fn: Function to initialize the location parameters of the guide.
        init_scale: Initial scale for the guide parameters.
    """

    @beartype
    def __init__(
        self,
        guide_type: str = "AutoNormal",
        init_loc_fn: Optional[Callable] = None,
        init_scale: float = 0.1,
        **kwargs: Any,
    ) -> None:
        """Initialize the AutoGuideFactory.

        Args:
            guide_type: The type of AutoGuide to create.
            init_loc_fn: Function to initialize the location parameters of the guide.
            init_scale: Initial scale for the guide parameters.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.guide_type = guide_type
        self.init_loc_fn = init_loc_fn
        self.init_scale = init_scale
        self._guide = None

    @beartype
    def create_guide(
        self, model: Callable, **kwargs: Any
    ) -> infer.autoguide.AutoGuide:
        """Create an AutoGuide for the given model.

        Args:
            model: The model function for which to create a guide.
            **kwargs: Additional keyword arguments.

        Returns:
            An AutoGuide instance.

        Raises:
            ValueError: If the specified guide_type is not supported.
        """
        # Map guide_type to the corresponding AutoGuide class
        guide_classes = {
            "AutoNormal": infer.autoguide.AutoNormal,
            "AutoDiagonalNormal": infer.autoguide.AutoDiagonalNormal,
            "AutoMultivariateNormal": infer.autoguide.AutoMultivariateNormal,
            "AutoLowRankMultivariateNormal": infer.autoguide.AutoLowRankMultivariateNormal,
            "AutoIAFNormal": infer.autoguide.AutoIAFNormal,
            "AutoLaplaceApproximation": infer.autoguide.AutoLaplaceApproximation,
            "AutoDelta": infer.autoguide.AutoDelta,
        }

        if self.guide_type not in guide_classes:
            raise ValueError(
                f"Unsupported guide type: {self.guide_type}. "
                f"Supported types are: {list(guide_classes.keys())}"
            )

        # Create the guide
        guide_cls = guide_classes[self.guide_type]
        self._guide = guide_cls(
            model,
            init_loc_fn=self.init_loc_fn,
            init_scale=self.init_scale,
        )
        return self._guide

    @beartype
    def get_guide(self) -> infer.autoguide.AutoGuide:
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
        """Get the posterior distribution parameters.

        Returns:
            Dictionary of posterior distribution parameters.

        Raises:
            RuntimeError: If the guide has not been created yet.
        """
        if self._guide is None:
            raise RuntimeError(
                "Guide has not been created yet. Call create_guide first."
            )
        return self._guide.get_posterior()

    @beartype
    def _setup_guide_impl(self, model: Callable, **kwargs) -> None:
        """
        Implementation of guide setup.

        This method creates an AutoGuide for the given model.

        Args:
            model: Model function to guide
            **kwargs: Additional keyword arguments
        """
        self.create_guide(model, **kwargs)

    @beartype
    def _sample_posterior_impl(
        self, model: Callable, guide: Callable, **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Implementation of posterior sampling.

        This method samples from the posterior distribution using the guide.

        Args:
            model: Model function
            guide: Guide function
            **kwargs: Additional keyword arguments

        Returns:
            Dictionary of posterior samples
        """
        # Get the number of samples
        num_samples = kwargs.get("num_samples", 1000)

        # Create a predictive object
        predictive = pyro.infer.Predictive(
            model, guide=guide, num_samples=num_samples
        )

        # Get samples
        samples = predictive()

        # Filter out observed values
        posterior_samples = {
            k: v for k, v in samples.items() if not k.startswith("_")
        }

        return posterior_samples


@inference_guide_registry.register("normal")
class NormalGuide(BaseInferenceGuide):
    """Normal guide for PyroVelocity models.

    This guide uses a multivariate normal distribution with diagonal covariance
    to approximate the posterior distribution of the model parameters.

    Attributes:
        init_scale: Initial scale for the guide parameters.
    """

    @beartype
    def __init__(
        self,
        init_scale: float = 0.1,
        **kwargs: Any,
    ) -> None:
        """Initialize the NormalGuide.

        Args:
            init_scale: Initial scale for the guide parameters.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.init_scale = init_scale
        self._params = {}

    @beartype
    def create_guide(self, model: Callable, **kwargs: Any) -> Callable:
        """Create a guide function for the given model.

        Args:
            model: The model function for which to create a guide.
            **kwargs: Additional keyword arguments.

        Returns:
            The guide function.
        """
        # Store the model for later use
        self._model = model

        # Define the guide function
        def guide_fn(*args, **kwargs):
            # Register parameters for each latent variable
            # This is a simplified example; in practice, you would need to
            # register parameters for all latent variables in the model
            for name, shape in kwargs.get("latent_shapes", {}).items():
                if name not in self._params:
                    # Initialize location parameter
                    loc = pyro.param(
                        f"{name}_loc",
                        torch.zeros(shape),
                        constraint=dist.constraints.real,
                    )
                    # Initialize scale parameter
                    scale = pyro.param(
                        f"{name}_scale",
                        torch.ones(shape) * self.init_scale,
                        constraint=dist.constraints.positive,
                    )
                    self._params[name] = (loc, scale)
                else:
                    loc, scale = self._params[name]

                # Sample from the variational distribution
                pyro.sample(name, dist.Normal(loc, scale))

        return guide_fn

    @beartype
    def get_guide(self) -> Callable:
        """Get the created guide function.

        Returns:
            The guide function.

        Raises:
            RuntimeError: If the guide has not been created yet.
        """
        if not hasattr(self, "_model"):
            raise RuntimeError(
                "Guide has not been created yet. Call create_guide first."
            )
        return self.create_guide(self._model)

    @beartype
    def get_posterior(self) -> Dict[str, torch.Tensor]:
        """Get the posterior distribution parameters.

        Returns:
            Dictionary of posterior distribution parameters.

        Raises:
            RuntimeError: If the guide has not been created yet.
        """
        if not self._params:
            raise RuntimeError(
                "Guide has not been created yet or no parameters have been registered."
            )

        posterior = {}
        for name, (loc, scale) in self._params.items():
            posterior[f"{name}_loc"] = loc
            posterior[f"{name}_scale"] = scale

        return posterior

    @beartype
    def _setup_guide_impl(self, model: Callable, **kwargs) -> None:
        """
        Implementation of guide setup.

        This method creates a guide function for the given model.

        Args:
            model: Model function to guide
            **kwargs: Additional keyword arguments
        """
        self.create_guide(model, **kwargs)

    @beartype
    def _sample_posterior_impl(
        self, model: Callable, guide: Callable, **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Implementation of posterior sampling.

        This method samples from the posterior distribution using the guide.

        Args:
            model: Model function
            guide: Guide function
            **kwargs: Additional keyword arguments

        Returns:
            Dictionary of posterior samples
        """
        # Get the number of samples
        num_samples = kwargs.get("num_samples", 1000)

        # Create a predictive object
        predictive = pyro.infer.Predictive(
            model, guide=guide, num_samples=num_samples
        )

        # Get samples
        samples = predictive()

        # Filter out observed values
        posterior_samples = {
            k: v for k, v in samples.items() if not k.startswith("_")
        }

        return posterior_samples


@inference_guide_registry.register("delta")
class DeltaGuide(BaseInferenceGuide):
    """Delta guide for PyroVelocity models.

    This guide uses point estimates (delta distributions) to approximate
    the posterior distribution of the model parameters. This is equivalent
    to maximum likelihood estimation.

    Attributes:
        init_values: Initial values for the guide parameters.
    """

    @beartype
    def __init__(
        self,
        init_values: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the DeltaGuide.

        Args:
            init_values: Initial values for the guide parameters.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.init_values = init_values or {}
        self._params = {}

    @beartype
    def create_guide(self, model: Callable, **kwargs: Any) -> Callable:
        """Create a guide function for the given model.

        Args:
            model: The model function for which to create a guide.
            **kwargs: Additional keyword arguments.

        Returns:
            The guide function.
        """
        # Store the model for later use
        self._model = model

        # Define the guide function
        def guide_fn(*args, **kwargs):
            # Register parameters for each latent variable
            for name, shape in kwargs.get("latent_shapes", {}).items():
                if name not in self._params:
                    # Initialize parameter with provided value or zeros
                    init_value = self.init_values.get(name, torch.zeros(shape))
                    param = pyro.param(
                        f"{name}_param",
                        init_value,
                        constraint=dist.constraints.real,
                    )
                    self._params[name] = param
                else:
                    param = self._params[name]

                # Sample from delta distribution (point estimate)
                pyro.sample(name, dist.Delta(param))

        return guide_fn

    @beartype
    def get_guide(self) -> Callable:
        """Get the created guide function.

        Returns:
            The guide function.

        Raises:
            RuntimeError: If the guide has not been created yet.
        """
        if not hasattr(self, "_model"):
            raise RuntimeError(
                "Guide has not been created yet. Call create_guide first."
            )
        return self.create_guide(self._model)

    @beartype
    def get_posterior(self) -> Dict[str, torch.Tensor]:
        """Get the posterior distribution parameters.

        Returns:
            Dictionary of posterior distribution parameters.

        Raises:
            RuntimeError: If the guide has not been created yet.
        """
        if not self._params:
            raise RuntimeError(
                "Guide has not been created yet or no parameters have been registered."
            )

        posterior = {}
        for name, param in self._params.items():
            posterior[name] = param

        return posterior

    @beartype
    def _setup_guide_impl(self, model: Callable, **kwargs) -> None:
        """
        Implementation of guide setup.

        This method creates a guide function for the given model.

        Args:
            model: Model function to guide
            **kwargs: Additional keyword arguments
        """
        self.create_guide(model, **kwargs)

    @beartype
    def _sample_posterior_impl(
        self, model: Callable, guide: Callable, **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Implementation of posterior sampling.

        This method samples from the posterior distribution using the guide.
        For a Delta guide, this just returns the point estimates.

        Args:
            model: Model function
            guide: Guide function
            **kwargs: Additional keyword arguments

        Returns:
            Dictionary of posterior samples
        """
        # For a Delta guide, we just return the point estimates
        posterior_samples = {}
        for name, param in self._params.items():
            # Create a batch dimension for consistency with other guides
            posterior_samples[name] = param.unsqueeze(0)

        return posterior_samples
