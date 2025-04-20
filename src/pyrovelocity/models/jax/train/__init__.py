"""
Training utilities for PyroVelocity JAX/NumPyro implementation.

This module contains utilities for training, including functional training loops,
optimizers, and metrics.
"""

from pyrovelocity.models.jax.train.loop import (
    train_model,
    evaluate_model,
    train_with_early_stopping,
    train_epoch,
)

from pyrovelocity.models.jax.train.optimizer import (
    create_optimizer,
    learning_rate_schedule,
    clip_gradients,
    create_optimizer_with_schedule,
)

from pyrovelocity.models.jax.train.metrics import (
    compute_loss,
    compute_elbo,
    compute_predictive_log_likelihood,
    compute_metrics,
    compute_validation_metrics,
)

__all__ = [
    # Loop
    "train_model",
    "evaluate_model",
    "train_with_early_stopping",
    "train_epoch",
    
    # Optimizer
    "create_optimizer",
    "learning_rate_schedule",
    "clip_gradients",
    "create_optimizer_with_schedule",
    
    # Metrics
    "compute_loss",
    "compute_elbo",
    "compute_predictive_log_likelihood",
    "compute_metrics",
    "compute_validation_metrics",
]