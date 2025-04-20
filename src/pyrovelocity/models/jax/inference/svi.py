"""
SVI utilities for PyroVelocity JAX/NumPyro implementation.

This module contains SVI utilities, including:

- create_svi: Create an SVI object
- svi_step: Single SVI step
- run_svi_inference: Run SVI inference
"""

from typing import Dict, Tuple, Optional, Any, List, Union, Callable
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO, TraceEnum_ELBO
from numpyro.infer.autoguide import AutoGuide
from jaxtyping import Array, Float, PyTree
from beartype import beartype

from pyrovelocity.models.jax.core.state import TrainingState, InferenceConfig

@beartype
def create_optimizer(
    optimizer_name: str = "adam",
    learning_rate: float = 0.01,
    **kwargs
) -> numpyro.optim._NumPyroOptim:
    """Create an optimizer.
    
    Args:
        optimizer_name: Name of the optimizer
        learning_rate: Learning rate
        **kwargs: Additional optimizer parameters
        
    Returns:
        NumPyro optimizer
    """
    # Create optimizer based on name
    if optimizer_name.lower() == "adam":
        return numpyro.optim.Adam(learning_rate, **kwargs)
    elif optimizer_name.lower() == "sgd":
        return numpyro.optim.SGD(learning_rate, **kwargs)
    elif optimizer_name.lower() == "rmsprop":
        return numpyro.optim.RMSProp(learning_rate, **kwargs)
    elif optimizer_name.lower() == "adagrad":
        return numpyro.optim.Adagrad(learning_rate, **kwargs)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

@beartype
def create_svi(
    model: Callable,
    guide: Union[AutoGuide, Callable],
    optimizer: numpyro.optim._NumPyroOptim,
    loss: Optional[numpyro.infer.ELBO] = None,
) -> SVI:
    """Create an SVI object.
    
    Args:
        model: NumPyro model function
        guide: NumPyro guide
        optimizer: NumPyro optimizer
        loss: ELBO loss function
        
    Returns:
        SVI object
    """
    # Use default loss if not provided
    if loss is None:
        loss = Trace_ELBO()
    
    # Create SVI object
    return SVI(model, guide, optimizer, loss)

@beartype
def svi_step(
    svi: SVI,
    state: TrainingState,
    *args,
    **kwargs
) -> TrainingState:
    """Single SVI step.
    
    Args:
        svi: SVI object
        state: Training state
        *args: Additional positional arguments for the model
        **kwargs: Additional keyword arguments for the model
        
    Returns:
        Updated training state
    """
    # Extract state components
    params = state.params
    opt_state = state.opt_state
    key = state.key
    step = state.step
    loss_history = state.loss_history
    best_params = state.best_params
    best_loss = state.best_loss
    
    # Generate a new key for this step
    key, subkey = jax.random.split(key)
    
    # Perform SVI step
    # Only use keyword arguments to avoid passing 'x' twice
    result = svi.update(params, opt_state, **kwargs)
    
    # Unpack the result based on the NumPyro SVI.update return value
    # In NumPyro, SVI.update returns (loss, optim_state) or ((loss, mutable_state), optim_state)
    if isinstance(result, tuple) and len(result) == 2:
        if isinstance(result[0], tuple) and len(result[0]) == 2:
            # Case: ((loss, mutable_state), optim_state)
            (loss, _), new_opt_state = result
        else:
            # Case: (loss, optim_state)
            loss, new_opt_state = result
        # Get the new parameters from the optimizer
        new_params = svi.optim.get_params(new_opt_state)
    else:
        raise ValueError(f"Unexpected return value from SVI.update: {result}")
    
    # Update loss history
    new_loss_history = loss_history + [float(loss)]
    
    # Update best parameters if this is the best loss so far
    if best_loss is None or loss < best_loss:
        new_best_params = new_params
        new_best_loss = float(loss)
    else:
        new_best_params = best_params
        new_best_loss = best_loss
    
    # Create updated state
    new_state = TrainingState(
        step=step + 1,
        params=new_params,
        opt_state=new_opt_state,
        loss_history=new_loss_history,
        best_params=new_best_params,
        best_loss=new_best_loss,
        key=key,
    )
    
    return new_state

@beartype
def run_svi_inference(
    model: Callable,
    guide: Union[AutoGuide, Callable],
    args: Tuple,
    kwargs: Dict[str, Any],
    config: InferenceConfig,
    key: jnp.ndarray,
) -> Tuple[TrainingState, Dict[str, jnp.ndarray]]:
    """Run SVI inference.
    
    Args:
        model: NumPyro model function
        guide: NumPyro guide
        args: Positional arguments for the model
        kwargs: Keyword arguments for the model
        config: Inference configuration
        key: JAX random key
        
    Returns:
        Tuple of (training_state, posterior_samples)
    """
    # Create optimizer
    optimizer = create_optimizer(
        optimizer_name=config.optimizer,
        learning_rate=config.learning_rate,
    )
    
    # Create SVI object
    svi = create_svi(model, guide, optimizer)
    
    # Initialize parameters
    key, subkey = jax.random.split(key)
    params = svi.init(subkey, *args, **kwargs)
    
    # Initialize optimizer state
    opt_state = optimizer.init(params)
    
    # Initialize training state
    state = TrainingState(
        step=0,
        params=params,
        opt_state=opt_state,
        loss_history=[],
        best_params=None,
        best_loss=None,
        key=key,
    )
    
    # JIT-compile the step function
    jitted_step = jax.jit(lambda state, *args, **kwargs: svi_step(svi, state, *args, **kwargs))
    
    # Run SVI for specified number of epochs
    patience_counter = 0
    for epoch in range(config.num_epochs):
        # Perform SVI step
        state = jitted_step(state, *args, **kwargs)
        
        # Check for early stopping
        if config.early_stopping and epoch > 0:
            # Get current and previous loss
            current_loss = state.loss_history[-1]
            previous_loss = state.loss_history[-2]
            
            # Check if loss improved
            if current_loss < previous_loss:
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Stop if patience exceeded
            if patience_counter >= config.early_stopping_patience:
                break
    
    # Use best parameters if available
    if state.best_params is not None:
        final_params = state.best_params
    else:
        final_params = state.params
    
    # Extract posterior samples
    key, subkey = jax.random.split(state.key)
    posterior_samples = extract_posterior_samples(guide, final_params, config.num_samples, subkey)
    
    # Update state with final key
    state = state.replace(key=key)
    
    return state, posterior_samples

@beartype
def extract_posterior_samples(
    guide: Union[AutoGuide, Callable],
    params: PyTree,
    num_samples: int,
    key: jnp.ndarray,
) -> Dict[str, jnp.ndarray]:
    """Extract posterior samples from guide.
    
    Args:
        guide: NumPyro guide
        params: Guide parameters
        num_samples: Number of posterior samples
        key: JAX random key
        
    Returns:
        Dictionary of posterior samples
    """
    # Handle different guide types
    if isinstance(guide, AutoGuide):
        # Use AutoGuide's sample_posterior method
        return guide.sample_posterior(
            rng_key=key,
            params=params,
            sample_shape=(num_samples,)
        )
    elif callable(guide):
        # For custom guides, we need to create a predictive object
        predictive = numpyro.infer.Predictive(
            guide,
            params=params,
            num_samples=num_samples,
        )
        # Sample from the guide
        samples = predictive(key)
        # Remove the "_auto_latent" prefix if present
        return {k.replace("_auto_latent", ""): v for k, v in samples.items()}
    else:
        raise TypeError(f"Unsupported guide type: {type(guide)}")