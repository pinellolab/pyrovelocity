"""
SVI utilities for PyroVelocity PyTorch/Pyro modular implementation.

This module contains SVI utilities, including:

- create_optimizer: Create an optimizer
- create_svi: Create an SVI object
- svi_step: Single SVI step
- run_svi_inference: Run SVI inference
- extract_posterior_samples: Extract posterior samples from SVI results
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pyro
import pyro.infer
import pyro.optim
import torch
from beartype import beartype
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoGuide

from pyrovelocity.models.modular.components.guides import InferenceGuide
from pyrovelocity.models.modular.inference.config import InferenceConfig


@dataclass
class TrainingState:
    """
    State of the training process.

    This class contains the state of the training process, including
    parameters, optimizer state, loss history, and best parameters.
    """

    step: int = 0
    params: Dict[str, torch.Tensor] = field(default_factory=dict)
    opt_state: Any = None
    loss_history: List[float] = field(default_factory=list)
    best_params: Dict[str, torch.Tensor] = field(default_factory=dict)
    best_loss: Optional[float] = None
    key: Optional[torch.Tensor] = None


@beartype
def create_optimizer(
    optimizer_name: str,
    learning_rate: float = 0.01,
    clip_norm: Optional[float] = None,
    max_epochs: Optional[int] = 1000,
    **kwargs: Any,
) -> Any:
    """
    Create a Pyro optimizer.

    Args:
        optimizer_name: Name of the optimizer ("adam", "sgd", "rmsprop", "velocity_clipped_adam")
        learning_rate: Learning rate
        clip_norm: Gradient clipping norm
        max_epochs: Maximum number of epochs (used for learning rate decay in VelocityClippedAdam)
        **kwargs: Additional optimizer parameters

    Returns:
        Pyro optimizer
    """
    # Import VelocityClippedAdam from the legacy implementation
    from pyrovelocity.models._trainer import VelocityClippedAdam

    # Set up optimizer parameters
    optim_args = {"lr": learning_rate, **kwargs}

    # Create optimizer based on name
    if optimizer_name.lower() == "velocity_clipped_adam":
        # Use VelocityClippedAdam with learning rate decay like in the legacy implementation
        return VelocityClippedAdam({"lr": learning_rate, "lrd": 0.1 ** (1 / max_epochs)})
    elif optimizer_name.lower() == "adam":
        if clip_norm is not None:
            return pyro.optim.ClippedAdam(optim_args)
        else:
            return pyro.optim.Adam(optim_args)
    elif optimizer_name.lower() == "sgd":
        return pyro.optim.SGD(optim_args)
    elif optimizer_name.lower() == "rmsprop":
        return pyro.optim.RMSprop(optim_args)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")


@beartype
def create_svi(
    model: Callable,
    guide: Union[AutoGuide, Callable, InferenceGuide],
    optimizer: Union[str, Any],
    loss: Optional[Any] = None,
    learning_rate: float = 0.01,
    clip_norm: Optional[float] = None,
    max_epochs: Optional[int] = 1000,
    **kwargs: Any,
) -> SVI:
    """
    Create an SVI object for variational inference.

    This function creates a Stochastic Variational Inference (SVI) object with the
    specified model, guide, optimizer, and loss function.

    Args:
        model: Pyro model function that defines the probabilistic model
        guide: Pyro guide function or AutoGuide object that defines the variational distribution
        optimizer: Pyro optimizer or optimizer name ("adam", "sgd", "velocity_clipped_adam", etc.)
        loss: ELBO loss function (defaults to Trace_ELBO if None)
        learning_rate: Learning rate for the optimizer (used if optimizer is a string)
        clip_norm: Gradient clipping norm (used if optimizer is a string)
        max_epochs: Maximum number of epochs (used for learning rate decay in VelocityClippedAdam)
        **kwargs: Additional optimizer parameters

    Returns:
        An SVI object for performing variational inference
    """
    # Use default loss if not provided
    if loss is None:
        # Use Trace_ELBO with strict_enumeration_warning=True like in the legacy implementation
        loss = Trace_ELBO(strict_enumeration_warning=True)

    # Handle string optimizer
    if isinstance(optimizer, str):
        # Use VelocityClippedAdam by default to match the legacy implementation
        if optimizer.lower() == "adam":
            optimizer = "velocity_clipped_adam"

        optimizer = create_optimizer(
            optimizer, learning_rate, clip_norm=clip_norm, max_epochs=max_epochs, **kwargs
        )

    # Create SVI object
    return SVI(model, guide, optimizer, loss)


@beartype
def svi_step(svi: SVI, *args: Any, **kwargs: Any) -> Any:
    """
    Perform a single SVI step.

    Args:
        svi: SVI object
        *args: Positional arguments to pass to the model and guide
        **kwargs: Keyword arguments to pass to the model and guide

    Returns:
        Loss value for this step
    """
    return svi.step(*args, **kwargs)


@beartype
def extract_posterior_samples(
    guide: Union[AutoGuide, Callable, InferenceGuide],
    params: Optional[
        Dict[str, torch.Tensor]
    ] = None,  # Unused but kept for API compatibility
    num_samples: int = 1000,
    seed: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """
    Extract posterior samples from SVI results.

    Args:
        guide: Guide function or AutoGuide object
        params: Parameters from SVI
        num_samples: Number of posterior samples to extract
        seed: Random seed

    Returns:
        Dictionary of posterior samples
    """
    # Set seed if provided
    if seed is not None:
        pyro.set_rng_seed(seed)

    # For all guide types, create a predictive object
    # First, get samples from the guide (latent variables)
    guide_predictive = pyro.infer.Predictive(guide, num_samples=num_samples)
    guide_samples = guide_predictive()

    # Now, we need to get deterministic sites from the model
    # We'll use the guide samples as input to the model
    # This is critical for getting ut and st which are deterministic sites

    # Get the model function from the guide if it's an AutoGuide
    if hasattr(guide, "model"):
        model_fn = guide.model
    else:
        # If guide is not an AutoGuide, we can't get the model
        # In this case, we'll just return the guide samples
        return guide_samples

    # Create a predictive object for the model, using the guide samples
    # Use True to return all sites, including deterministic ones
    model_predictive = pyro.infer.Predictive(
        model_fn,
        posterior_samples=guide_samples,
        return_sites=True,  # Return all sites, including deterministic
        num_samples=num_samples
    )

    # Run the model predictive to get all sites, including deterministic ones
    model_samples = model_predictive()

    # Combine guide and model samples
    # Guide samples take precedence if there's a conflict
    combined_samples = {**model_samples, **guide_samples}

    return combined_samples


@beartype
def run_svi_inference(
    model: Callable,
    guide: Union[AutoGuide, Callable, InferenceGuide],
    args: Tuple = (),
    kwargs: Optional[Dict[str, Any]] = None,
    config: Optional[InferenceConfig] = None,
    seed: Optional[int] = None,
) -> Tuple[TrainingState, Dict[str, torch.Tensor]]:
    """
    Run SVI inference with a model and guide.

    This function performs Stochastic Variational Inference (SVI) with the specified model
    and guide.

    Args:
        model: Pyro model function
        guide: Pyro guide function, AutoGuide object, or guide name
        args: Positional arguments to pass to the model and guide
        kwargs: Keyword arguments to pass to the model and guide
        config: Inference configuration
        seed: Random seed

    Returns:
        Tuple of (training state, posterior samples)
    """
    # Set default kwargs if None
    if kwargs is None:
        kwargs = {}

    # Set default config if None
    if config is None:
        config = InferenceConfig()

    # Set seed if provided
    if seed is not None:
        pyro.set_rng_seed(seed)
    elif config.seed is not None:
        pyro.set_rng_seed(config.seed)

    # Create SVI object
    svi = create_svi(
        model=model,
        guide=guide,
        optimizer=config.optimizer,
        learning_rate=config.learning_rate,
        clip_norm=config.clip_norm,
        max_epochs=config.num_epochs,
    )

    # Initialize training state
    state = TrainingState(step=0)

    # Run SVI for the specified number of epochs
    for _ in range(config.num_epochs):


        # Perform a single SVI step
        loss = svi.step(*args, **kwargs)
        loss_value = float(loss)  # Convert to float

        # Update training state
        state.step += 1
        state.loss_history.append(loss_value)

        # Update best parameters if this is the best loss so far
        if state.best_loss is None or loss_value < state.best_loss:
            state.best_loss = loss_value
            # In Pyro, we need to get the parameters from the param store
            state.best_params = {
                name: param.detach().clone()
                for name, param in pyro.get_param_store().items()
            }

        # Early stopping
        if (
            config.early_stopping
            and len(state.loss_history) > config.early_stopping_patience
            and all(
                state.loss_history[-i - 1] >= state.loss_history[-i - 2]
                for i in range(config.early_stopping_patience)
            )
        ):
            break

    # Use the best parameters
    # In Pyro, we need to set the parameters in the param store
    for name, param in state.best_params.items():
        pyro.param(name, param)
    state.params = state.best_params

    # Extract posterior samples
    # We need to pass both the guide and the model to extract_posterior_samples
    # This is critical for getting deterministic sites like ut and st
    posterior_samples = extract_posterior_samples(
        guide, state.params, config.num_samples, seed
    )

    # If we're using an AutoGuide, we can get the model from it
    if hasattr(guide, "model"):
        model_fn = guide.model

        # Create a predictive object for the model, using the guide samples
        # Return all sites, including deterministic ones
        model_predictive = pyro.infer.Predictive(
            model_fn,
            posterior_samples=posterior_samples,
            return_sites=None,  # Return all sites, including deterministic
            num_samples=config.num_samples
        )

        # Run the model predictive to get all sites, including deterministic ones
        model_samples = model_predictive()

        # Combine guide and model samples
        # Guide samples take precedence if there's a conflict
        posterior_samples = {**model_samples, **posterior_samples}

    return state, posterior_samples
