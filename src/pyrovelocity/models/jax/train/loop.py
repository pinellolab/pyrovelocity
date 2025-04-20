"""
Functional training loop implementations for PyroVelocity JAX/NumPyro implementation.

This module contains functional training loop implementations, including:

- train_model: Train a model using functional SVI
- evaluate_model: Evaluate a model on data
- train_with_early_stopping: Train with early stopping
"""

from typing import Dict, Tuple, Optional, Any, List, Union, Callable
import jax
import jax.numpy as jnp
import numpyro
from numpyro.infer import SVI
from jaxtyping import Array, Float, PyTree
from beartype import beartype

from pyrovelocity.models.jax.core.state import TrainingState, InferenceConfig
from pyrovelocity.models.jax.inference.svi import svi_step

@beartype
def train_model(
    svi: SVI,
    initial_state: TrainingState,
    data: Dict[str, jnp.ndarray],
    num_epochs: int,
    batch_size: Optional[int] = None,
    early_stopping: bool = True,
    early_stopping_patience: int = 10,
    verbose: bool = True,
) -> TrainingState:
    """Train a model using functional SVI.
    
    Args:
        svi: SVI object
        initial_state: Initial training state
        data: Dictionary of data arrays
        num_epochs: Number of epochs
        batch_size: Batch size
        early_stopping: Whether to use early stopping
        early_stopping_patience: Patience for early stopping
        verbose: Whether to print progress
        
    Returns:
        Final training state
    """
    # If early stopping is enabled, we need to split the data into train and validation sets
    if early_stopping:
        # For simplicity, use 90% of data for training and 10% for validation
        # In a real implementation, this should be a parameter or use a proper validation set
        n_data = next(iter(data.values())).shape[0]
        n_train = int(0.9 * n_data)
        
        # Split the data
        train_data = {}
        val_data = {}
        for k, v in data.items():
            train_data[k] = v[:n_train]
            val_data[k] = v[n_train:]
        
        # Train with early stopping
        return train_with_early_stopping(
            svi=svi,
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
            state = train_epoch(svi, state, data, batch_size)
            
            # Print progress if verbose
            if verbose and (epoch + 1) % max(1, num_epochs // 10) == 0:
                loss = evaluate_model(svi, state, data)
                print(f"Epoch {epoch+1}/{num_epochs}: loss={loss:.4f}")
        
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
    # Get the loss function from the SVI object
    loss_fn = svi.loss
    
    # Get the model and guide from the SVI object
    model = svi.model
    guide = svi.guide
    
    # Get the parameters from the state
    params = state.params
    
    # Compute the loss
    loss = loss_fn(model, guide, params, **data)
    
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
    best_val_loss = float('inf')
    best_state = state
    patience_counter = 0
    
    # Train for specified number of epochs
    for epoch in range(num_epochs):
        # Train for one epoch
        state = train_epoch(svi, state, train_data, batch_size)
        
        # Evaluate on validation data
        val_loss = evaluate_model(svi, state, val_data)
        
        # Print progress if verbose
        if verbose:
            train_loss = evaluate_model(svi, state, train_data)
            print(f"Epoch {epoch+1}/{num_epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
        
        # Check if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = state.replace(best_params=state.params, best_loss=val_loss)
            patience_counter = 0
        else:
            patience_counter += 1
            
        # Check if patience exceeded
        if patience_counter >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Return best state
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
        return svi_step(svi, state, **data)
    
    # Otherwise, perform mini-batch training
    # Get the number of data points
    n_data = next(iter(data.values())).shape[0]
    
    # Calculate number of batches
    n_batches = n_data // batch_size + (1 if n_data % batch_size > 0 else 0)
    
    # Initialize updated state
    updated_state = state
    
    # Generate a new key for shuffling
    key, subkey = jax.random.split(state.key)
    
    # Generate random indices for shuffling
    indices = jax.random.permutation(subkey, n_data)
    
    # Loop over batches
    for i in range(n_batches):
        # Get batch indices
        batch_indices = indices[i * batch_size:min((i + 1) * batch_size, n_data)]
        
        # Create batch data
        batch_data = {}
        for k, v in data.items():
            batch_data[k] = v[batch_indices]
        
        # Perform SVI step on batch
        updated_state = svi_step(svi, updated_state, **batch_data)
    
    # Update the random key
    updated_state = updated_state.replace(key=key)
    
    return updated_state