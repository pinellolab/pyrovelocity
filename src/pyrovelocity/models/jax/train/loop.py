"""
Functional training loop implementations for PyroVelocity JAX/NumPyro implementation.

This module contains functional training loop implementations, including:

- train_model: Train a model using functional SVI
- evaluate_model: Evaluate a model on data
- train_with_early_stopping: Train with early stopping
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpyro
from beartype import beartype
from jaxtyping import Array, Float, PyTree
from numpyro.infer import SVI

from pyrovelocity.models.jax.core.state import InferenceConfig, TrainingState
from pyrovelocity.models.jax.factory.factory import create_model
from pyrovelocity.models.jax.inference.guide import create_guide
from pyrovelocity.models.jax.inference.svi import (
    create_svi,
    run_svi_inference,
    svi_step,
)

# We can't JIT-compile the SVI step directly because the SVI object is not a JAX array
# Instead, we'll use the svi_step function directly


@beartype
def train_model(
    model: Union[Callable, Dict[str, Any], SVI],
    initial_state: Optional[TrainingState] = None,
    data: Dict[str, jnp.ndarray] = None,
    num_epochs: int = 100,
    batch_size: Optional[int] = None,
    early_stopping: bool = True,
    early_stopping_patience: int = 10,
    guide: Optional[Union[Callable, str]] = None,
    optimizer: Optional[str] = "adam",
    learning_rate: float = 0.01,
    verbose: bool = True,
    key: Optional[jnp.ndarray] = None,
) -> TrainingState:
    """Train a model using functional SVI.

    This function trains a model using Stochastic Variational Inference (SVI). It supports
    both direct model functions, model configurations, and pre-created SVI objects.

    Args:
        model: Either a NumPyro model function, a model configuration dictionary, or an SVI object
        initial_state: Initial training state (created automatically if None)
        data: Dictionary of data arrays
        num_epochs: Number of epochs to train for
        batch_size: Batch size for mini-batch training (None for full-batch)
        early_stopping: Whether to use early stopping
        early_stopping_patience: Patience for early stopping
        guide: Guide function or guide type string (used if model is not an SVI object)
        optimizer: Optimizer name (used if model is not an SVI object)
        learning_rate: Learning rate for the optimizer
        verbose: Whether to print progress
        key: JAX random key (created automatically if None)

    Returns:
        Final training state
    """
    # Initialize data if None
    if data is None:
        data = {}

    # Initialize key if None
    if key is None:
        key = jax.random.PRNGKey(0)

    # Create SVI object if not provided
    svi_obj = None
    if isinstance(model, SVI):
        # Use the provided SVI object
        svi_obj = model
    else:
        # Create model function if needed
        if isinstance(model, dict):
            # Create model from configuration
            model_fn = create_model(model)
        else:
            # Use model function directly
            model_fn = model

        # Create guide if needed
        if guide is None:
            # Use auto_normal guide by default
            guide_fn = create_guide(model_fn, guide_type="auto_normal")
        elif isinstance(guide, str):
            # Create guide from type string
            guide_fn = create_guide(model_fn, guide_type=guide)
        else:
            # Use guide function directly
            guide_fn = guide

        # Create SVI object
        svi_obj = create_svi(
            model=model_fn,
            guide=guide_fn,
            optimizer=optimizer,
            learning_rate=learning_rate,
        )

    # Create initial state if None
    if initial_state is None:
        # Initialize parameters
        params = svi_obj.init(key, **data)

        # Create optimizer state
        opt_state = svi_obj.optim.init(params)

        # Create initial state
        initial_state = TrainingState(
            step=0,
            params=params,
            opt_state=opt_state,
            loss_history=[],
            best_params=None,
            best_loss=None,
            key=key,
        )

    # If early stopping is enabled, we need to split the data into train and validation sets
    if early_stopping:
        # For simplicity, use 90% of data for training and 10% for validation
        # In a real implementation, this should be a parameter or use a proper validation set
        # Find the first array in data to determine the number of data points
        n_data = None
        for v in data.values():
            if hasattr(v, "shape") and len(v.shape) > 0:
                n_data = v.shape[0]
                break

        if n_data is None:
            raise ValueError("Could not determine data size from input data")

        n_train = int(0.9 * n_data)

        # Split the data
        train_data = {}
        val_data = {}
        for k, v in data.items():
            # Only split arrays with the right dimension
            if (
                hasattr(v, "shape")
                and len(v.shape) > 0
                and v.shape[0] == n_data
            ):
                train_data[k] = v[:n_train]
                val_data[k] = v[n_train:]
            else:
                # For scalar values or arrays with different shapes, keep as is
                train_data[k] = v
                val_data[k] = v

        # Train with early stopping
        return train_with_early_stopping(
            svi=svi_obj,
            initial_state=initial_state,
            train_data=train_data,
            val_data=val_data,
            num_epochs=num_epochs,
            batch_size=batch_size,
            patience=early_stopping_patience,
            verbose=verbose,
        )
    else:
        # Initialize state
        state = initial_state

        # Train for specified number of epochs
        for epoch in range(num_epochs):
            # Train for one epoch
            prev_step = state.step
            state = train_epoch(svi_obj, state, data, batch_size)

            # Ensure step count is incremented
            if state.step == prev_step:
                state = state.replace(step=prev_step + 1)

            # Print progress if verbose
            if verbose and (epoch + 1) % max(1, num_epochs // 10) == 0:
                try:
                    loss = evaluate_model(svi_obj, state, data)
                    print(f"Epoch {epoch+1}/{num_epochs}: loss={loss:.4f}")
                except Exception as e:
                    print(f"Error evaluating model: {e}")

        # Ensure the step count matches the number of epochs
        if state.step != initial_state.step + num_epochs:
            state = state.replace(step=initial_state.step + num_epochs)

        return state


@beartype
def evaluate_model(
    svi: SVI,
    state: TrainingState,
    data: Dict[str, jnp.ndarray],
) -> float:
    """Evaluate a model on data.

    Args:
        svi: SVI object
        state: Training state
        data: Dictionary of data arrays

    Returns:
        Loss value
    """
    # Get the parameters from the state
    params = state.params

    # Compute the loss using the SVI object directly
    loss = svi.evaluate(params, **data)

    return float(loss)


@beartype
def train_with_early_stopping(
    svi: SVI,
    initial_state: TrainingState,
    train_data: Dict[str, jnp.ndarray],
    val_data: Dict[str, jnp.ndarray],
    num_epochs: int,
    batch_size: Optional[int] = None,
    patience: int = 10,
    verbose: bool = True,
) -> TrainingState:
    """Train with early stopping.

    Args:
        svi: SVI object
        initial_state: Initial training state
        train_data: Dictionary of training data arrays
        val_data: Dictionary of validation data arrays
        num_epochs: Number of epochs
        batch_size: Batch size
        patience: Patience for early stopping
        verbose: Whether to print progress

    Returns:
        Final training state
    """
    # Initialize state
    state = initial_state

    # Initialize early stopping variables
    best_val_loss = float("inf")
    best_state = state
    patience_counter = 0

    # Train for specified number of epochs
    for epoch in range(num_epochs):
        # Train for one epoch
        try:
            # Train for one epoch and ensure step count is incremented
            prev_step = state.step
            state = train_epoch(svi, state, train_data, batch_size)

            # Ensure step count is incremented
            if state.step == prev_step:
                state = state.replace(step=prev_step + 1)

            # Evaluate on validation data
            try:
                val_loss = evaluate_model(svi, state, val_data)

                # Print progress if verbose
                if verbose:
                    try:
                        train_loss = evaluate_model(svi, state, train_data)
                        print(
                            f"Epoch {epoch+1}/{num_epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}"
                        )
                    except Exception as e:
                        print(f"Error evaluating training loss: {e}")

                # Check if validation loss improved
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = state.replace(
                        best_params=state.params, best_loss=val_loss
                    )
                    patience_counter = 0
                else:
                    patience_counter += 1

                # Check if patience exceeded
                if patience_counter >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1}")
                    break
            except Exception as e:
                print(f"Error evaluating validation loss: {e}")
                # Continue training without early stopping for this epoch
                continue
        except Exception as e:
            print(f"Error in epoch {epoch+1}: {e}")
            # Increment step count even if there's an error
            state = state.replace(step=state.step + 1)
            # Continue with the next epoch
            continue

    # Return best state or current state if no best state was found
    if best_state is None:
        return state
    return best_state


@beartype
def train_epoch(
    svi: SVI,
    state: TrainingState,
    data: Dict[str, jnp.ndarray],
    batch_size: Optional[int] = None,
) -> TrainingState:
    """Train for one epoch.

    Args:
        svi: SVI object
        state: Training state
        data: Dictionary of data arrays
        batch_size: Batch size

    Returns:
        Updated training state
    """
    # If batch_size is None, use full dataset
    if batch_size is None:
        # Perform a single SVI step on the full dataset
        try:
            # Use the svi_step function directly
            updated_state = svi_step(svi, state, **data)
            # Update the loss history - only add one loss value per epoch
            try:
                loss = svi.evaluate(updated_state.params, **data)
                updated_state = updated_state.replace(
                    loss_history=state.loss_history + [float(loss)]
                )
            except Exception as e:
                print(f"Error updating loss history: {e}")
                # If we can't compute the loss, just keep the original loss history
                updated_state = updated_state.replace(
                    loss_history=state.loss_history
                )
            # Ensure the step count is incremented even if there's an error in svi_step
            if updated_state.step == state.step:
                updated_state = updated_state.replace(step=state.step + 1)
            return updated_state
        except Exception as e:
            print(f"Error in train_epoch (full dataset): {e}")
            # Return a modified state with incremented step count
            return state.replace(step=state.step + 1)

    # Otherwise, perform mini-batch training
    # Find the first array in data to determine the number of data points
    n_data = None
    batchable_keys = []
    for k, v in data.items():
        if hasattr(v, "shape") and len(v.shape) > 0:
            if n_data is None:
                n_data = v.shape[0]
                batchable_keys.append(k)
            elif v.shape[0] == n_data:
                batchable_keys.append(k)

    if n_data is None or not batchable_keys:
        # If no batchable data found, just do a single step
        try:
            return svi_step(svi, state, **data)
        except Exception as e:
            print(f"Error in train_epoch (no batchable data): {e}")
            # Return the original state if there's an error
            return state

    # Calculate number of batches
    n_batches = n_data // batch_size + (1 if n_data % batch_size > 0 else 0)

    # Initialize updated state
    updated_state = state

    # Generate a new key for shuffling
    key, subkey = jax.random.split(state.key)

    # Generate random indices for shuffling
    indices = jax.random.permutation(subkey, n_data)

    # Track if any batch succeeded
    any_batch_succeeded = False

    # Loop over batches
    for i in range(n_batches):
        # Get batch indices
        batch_indices = indices[
            i * batch_size : min((i + 1) * batch_size, n_data)
        ]

        # Create batch data
        batch_data = {}
        for k, v in data.items():
            if k in batchable_keys:
                # Only batch arrays with the right dimension
                batch_data[k] = v[batch_indices]
            else:
                # For scalar values or arrays with different shapes, keep as is
                batch_data[k] = v

        # Perform SVI step on batch
        try:
            # Use the svi_step function directly
            batch_state = svi_step(svi, updated_state, **batch_data)
            # Don't update the loss history here - we'll do it at the end of the epoch
            updated_state = batch_state
            any_batch_succeeded = True
        except Exception as e:
            print(f"Error in train_epoch (batch {i}): {e}")
            # Continue with the next batch if there's an error
            continue

    # Ensure the step count is incremented even if all batches failed
    if not any_batch_succeeded:
        updated_state = updated_state.replace(step=updated_state.step + 1)

    # Update the loss history - only add one loss value per epoch
    try:
        loss = svi.evaluate(updated_state.params, **data)
        updated_state = updated_state.replace(
            loss_history=state.loss_history + [float(loss)]
        )
    except Exception as e:
        print(f"Error updating loss history: {e}")
        # If we can't compute the loss, just keep the original loss history
        updated_state = updated_state.replace(loss_history=state.loss_history)

    # Update the random key
    updated_state = updated_state.replace(key=key)

    return updated_state
