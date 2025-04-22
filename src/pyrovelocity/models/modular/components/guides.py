"""Inference guides for PyroVelocity.

This module contains implementations of inference guides that define how
to perform approximate Bayesian inference in the PyroVelocity framework.
Guides are responsible for defining the variational distribution used to
approximate the posterior distribution of the model parameters.
"""

from typing import Any, Callable, Dict, Optional

import pyro
import pyro.distributions as dist
import torch
from beartype import beartype
from pyro.infer import Predictive, autoguide

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
        name: str = "inference_guide",
        **kwargs: Any,
    ) -> None:
        """Initialize the AutoGuideFactory.

        Args:
            guide_type: The type of AutoGuide to create.
            init_loc_fn: Function to initialize the location parameters of the guide.
            init_scale: Initial scale for the guide parameters.
            name: A unique name for this component instance.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(name=name)
        self.guide_type = guide_type
        self.init_loc_fn = init_loc_fn
        self.init_scale = init_scale
        self._guide = None
        self._model = None

    @beartype
    def create_guide(
        self, model: Callable, **kwargs: Any
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
    def _sample_posterior_impl(self, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Implementation of posterior sampling.

        This method samples from the posterior distribution using the guide.

        Args:
            **kwargs: Additional keyword arguments

        Returns:
            Dictionary of posterior samples
        """
        if self._guide is None or self._model is None:
            raise RuntimeError(
                "Guide has not been created yet. Call create_guide first."
            )

        # Get the number of samples
        num_samples = kwargs.get("num_samples", 1000)

        # Create a predictive object
        predictive = Predictive(
            self._model, guide=self._guide, num_samples=num_samples
        )

        # Get samples
        samples = predictive(**kwargs)

        # Filter out observed values
        posterior_samples = {
            k: v for k, v in samples.items() if not k.startswith("_")
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

        # If we still don't have a guide, raise an error
        if self._guide is None:
            raise RuntimeError(
                "Guide has not been created yet. Call create_guide first."
            )

        # Delegate to the created guide
        return self._guide(*args, **kwargs)


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
    def _sample_posterior_impl(self, **kwargs) -> Dict[str, torch.Tensor]:
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
        if (
            not hasattr(self, "_model")
            or not hasattr(self, "_params")
            or not self._params
        ):
            raise RuntimeError(
                "Guide has not been created yet or no parameters have been registered."
            )

        # Get the number of samples
        num_samples = kwargs.get("num_samples", 1000)

        # For a normal guide, we sample from the learned distributions
        posterior_samples = {}
        for name, (loc, scale) in self._params.items():
            # Sample from normal distribution
            samples = torch.distributions.Normal(loc, scale).sample(
                torch.Size([num_samples])
            )
            posterior_samples[name] = samples

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
        if len(args) > 0 and callable(args[0]) and not hasattr(self, "_model"):
            model = args[0]
            self.create_guide(model)
            self._model = model

        # If we still don't have a guide, raise an error
        if not hasattr(self, "_model"):
            raise RuntimeError(
                "Guide has not been created yet. Call create_guide first."
            )

        # Get the guide function and call it
        guide_fn = self.get_guide()
        return guide_fn(*args, **kwargs)


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
    def _sample_posterior_impl(self, **kwargs) -> Dict[str, torch.Tensor]:
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
        if (
            not hasattr(self, "_model")
            or not hasattr(self, "_params")
            or not self._params
        ):
            raise RuntimeError(
                "Guide has not been created yet or no parameters have been registered."
            )

        # For a Delta guide, we just return the point estimates
        posterior_samples = {}
        for name, param in self._params.items():
            # Create a batch dimension for consistency with other guides
            posterior_samples[name] = param.unsqueeze(0)

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
        if len(args) > 0 and callable(args[0]) and not hasattr(self, "_model"):
            model = args[0]
            self.create_guide(model)
            self._model = model

        # If we still don't have a guide, raise an error
        if not hasattr(self, "_model"):
            raise RuntimeError(
                "Guide has not been created yet. Call create_guide first."
            )

        # Get the guide function and call it
        guide_fn = self.get_guide()
        return guide_fn(*args, **kwargs)
