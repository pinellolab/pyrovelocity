"""
Protocol-First inference guide implementations for PyroVelocity's modular architecture.

This module contains inference guide implementations that directly implement the
InferenceGuide Protocol without inheriting from BaseInferenceGuide. These implementations
follow the Protocol-First approach, which embraces composition over inheritance
and allows for more flexible component composition.
"""

from typing import Any, Callable, Dict, Optional, Union

import pyro
import torch
from beartype import beartype
from pyro.infer import Predictive, autoguide
from pyro.infer.autoguide import init_to_median

from pyrovelocity.models.modular.interfaces import InferenceGuide
from pyrovelocity.models.modular.registry import inference_guide_registry

# Ensure the registry is initialized
inference_guide_registry._registry = inference_guide_registry._registry or {}


@inference_guide_registry.register("auto_direct")
class AutoGuideFactoryDirect:
    """Factory for creating AutoGuides for PyroVelocity models.

    This class provides a factory for creating different types of AutoGuides
    based on the specified guide type. It supports various AutoGuide types
    provided by Pyro, such as AutoNormal, AutoDiagonalNormal, etc.

    This implementation directly implements the InferenceGuide Protocol
    without inheriting from BaseInferenceGuide.

    Attributes:
        guide_type: The type of AutoGuide to create.
        init_loc_fn: Function to initialize the location parameters of the guide.
        init_scale: Initial scale for the guide parameters.
        name: A unique name for this component instance.
    """

    @beartype
    def __init__(
        self,
        guide_type: str = "AutoNormal",
        init_loc_fn: Optional[Callable] = None,
        init_scale: float = 0.1,
        name: str = "inference_guide_direct",
        **kwargs: Any,
    ) -> None:
        """Initialize the AutoGuideFactoryDirect.

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
        predictive = pyro.infer.Predictive(
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


@inference_guide_registry.register("normal_direct")
class NormalGuideDirect:
    """Normal guide for PyroVelocity models.

    This guide uses a multivariate normal distribution with diagonal covariance
    to approximate the posterior distribution of the model parameters.

    This implementation directly implements the InferenceGuide Protocol
    without inheriting from BaseInferenceGuide.

    Attributes:
        init_scale: Initial scale for the guide parameters.
        name: A unique name for this component instance.
    """

    @beartype
    def __init__(
        self,
        init_scale: float = 0.1,
        name: str = "normal_guide_direct",
        **kwargs: Any,
    ) -> None:
        """Initialize the NormalGuideDirect.

        Args:
            init_scale: Initial scale for the guide parameters.
            name: A unique name for this component instance.
            **kwargs: Additional keyword arguments.
        """
        self.name = name
        self.init_scale = init_scale
        self._params = {}
        self._model = None

    @beartype
    def forward(
        self,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Create a guide function for the model.

        This method takes a context dictionary containing the model function,
        creates a guide function that uses normal distributions for all latent variables,
        and updates the context with the guide function.

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

        # Store the model for later use
        self._model = model

        # Define the guide function
        def guide_fn(*args, **kwargs):
            # Register parameters for each latent variable
            for name, shape in kwargs.get("latent_shapes", {}).items():
                if name not in self._params:
                    # Initialize location parameter
                    loc = pyro.param(
                        f"{name}_loc",
                        torch.zeros(shape),
                        constraint=pyro.distributions.constraints.real,
                    )
                    # Initialize scale parameter
                    scale = pyro.param(
                        f"{name}_scale",
                        torch.ones(shape) * self.init_scale,
                        constraint=pyro.distributions.constraints.positive,
                    )
                    self._params[name] = (loc, scale)
                else:
                    loc, scale = self._params[name]

                # Sample from the variational distribution
                pyro.sample(name, pyro.distributions.Normal(loc, scale))

        # Update context with guide function
        context["guide"] = guide_fn

        return context

    @beartype
    def create_guide(
        self,
        model: Callable,
    ) -> Callable:
        """
        Create a guide function for the given model.

        Args:
            model: The model function

        Returns:
            A guide function compatible with the model
        """
        # Store the model for later use
        self._model = model

        # Define the guide function
        def guide_fn(*args, **kwargs):
            # Register parameters for each latent variable
            for name, shape in kwargs.get("latent_shapes", {}).items():
                if name not in self._params:
                    # Initialize location parameter
                    loc = pyro.param(
                        f"{name}_loc",
                        torch.zeros(shape),
                        constraint=pyro.distributions.constraints.real,
                    )
                    # Initialize scale parameter
                    scale = pyro.param(
                        f"{name}_scale",
                        torch.ones(shape) * self.init_scale,
                        constraint=pyro.distributions.constraints.positive,
                    )
                    self._params[name] = (loc, scale)
                else:
                    loc, scale = self._params[name]

                # Sample from the variational distribution
                pyro.sample(name, pyro.distributions.Normal(loc, scale))

        return guide_fn

    @beartype
    def __call__(
        self,
        model: Callable,
        *args: Any,
        **kwargs: Any,
    ) -> Callable:
        """
        Create a guide function for the given model.

        This method is called by Pyro's SVI when the guide is used directly
        in svi.step() as the guide parameter.

        Args:
            model: The model function
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            A guide function compatible with the model
        """
        return self.create_guide(model)

    @beartype
    def sample_posterior(
        self,
        num_samples: int = 1000,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """Sample from the posterior distribution.

        Args:
            num_samples: Number of samples to draw from the posterior.
            **kwargs: Additional keyword arguments.

        Returns:
            Dictionary of posterior samples.
        """
        if not self._params:
            raise RuntimeError(
                "No parameters have been registered. Call create_guide first."
            )

        # For a normal guide, we sample from the learned distributions
        posterior_samples = {}
        for name, (loc, scale) in self._params.items():
            # Sample from normal distribution
            samples = torch.distributions.Normal(loc, scale).sample(
                torch.Size([num_samples])
            )
            posterior_samples[name] = samples

        return posterior_samples


@inference_guide_registry.register("delta_direct")
class DeltaGuideDirect:
    """Delta guide for PyroVelocity models.

    This guide uses point estimates (delta distributions) to approximate
    the posterior distribution of the model parameters. This is equivalent
    to maximum likelihood estimation.

    This implementation directly implements the InferenceGuide Protocol
    without inheriting from BaseInferenceGuide.

    Attributes:
        init_values: Initial values for the guide parameters.
        name: A unique name for this component instance.
    """

    @beartype
    def __init__(
        self,
        init_values: Optional[Dict[str, torch.Tensor]] = None,
        name: str = "delta_guide_direct",
        **kwargs: Any,
    ) -> None:
        """Initialize the DeltaGuideDirect.

        Args:
            init_values: Initial values for the guide parameters.
            name: A unique name for this component instance.
            **kwargs: Additional keyword arguments.
        """
        self.name = name
        self.init_values = init_values or {}
        self._params = {}
        self._model = None

    @beartype
    def forward(
        self,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Create a guide function for the model.

        This method takes a context dictionary containing the model function,
        creates a guide function that uses delta distributions for all latent variables,
        and updates the context with the guide function.

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
                        constraint=pyro.distributions.constraints.real,
                    )
                    self._params[name] = param
                else:
                    param = self._params[name]

                # Sample from delta distribution (point estimate)
                pyro.sample(name, pyro.distributions.Delta(param))

        # Update context with guide function
        context["guide"] = guide_fn

        return context

    @beartype
    def create_guide(
        self,
        model: Callable,
    ) -> Callable:
        """
        Create a guide function for the given model.

        Args:
            model: The model function

        Returns:
            A guide function compatible with the model
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
                        constraint=pyro.distributions.constraints.real,
                    )
                    self._params[name] = param
                else:
                    param = self._params[name]

                # Sample from delta distribution (point estimate)
                pyro.sample(name, pyro.distributions.Delta(param))

        return guide_fn

    @beartype
    def __call__(
        self,
        model: Callable,
        *args: Any,
        **kwargs: Any,
    ) -> Callable:
        """
        Create a guide function for the given model.

        This method is called by Pyro's SVI when the guide is used directly
        in svi.step() as the guide parameter.

        Args:
            model: The model function
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            A guide function compatible with the model
        """
        return self.create_guide(model)

    @beartype
    def sample_posterior(
        self,
        num_samples: int = 1000,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """Sample from the posterior distribution.

        For a Delta guide, this just returns the point estimates repeated num_samples times.

        Args:
            num_samples: Number of samples to draw from the posterior.
            **kwargs: Additional keyword arguments.

        Returns:
            Dictionary of posterior samples.
        """
        if not self._params:
            raise RuntimeError(
                "No parameters have been registered. Call create_guide first."
            )

        # For a delta guide, we just repeat the point estimates
        posterior_samples = {}
        for name, param in self._params.items():
            # Repeat the point estimate num_samples times
            samples = param.unsqueeze(0).expand(num_samples, *param.shape)
            posterior_samples[name] = samples

        return posterior_samples