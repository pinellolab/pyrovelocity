"""
Bayesian model comparison and criticism for PyroVelocity's modular architecture.

This module provides tools for comparing and evaluating Bayesian models in the
PyroVelocity framework, implementing various information criteria and cross-validation
methods for model selection and criticism.

The module includes:
1. BayesianModelComparison - Core class for model comparison metrics (WAIC, LOO, Bayes factors)
2. Utility functions for model selection and comparison visualization
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import arviz as az
import numpy as np
import pandas as pd
import pyro
import torch
from anndata import AnnData
from beartype import beartype
from jaxtyping import Float, jaxtyped

from pyrovelocity.models.modular.model import PyroVelocityModel

logger = logging.getLogger(__name__)


@dataclass
class ComparisonResult:
    """
    Container for model comparison results.

    This dataclass stores the results of model comparison operations,
    including metric values and metadata about the comparison.

    Attributes:
        metric_name: Name of the comparison metric (e.g., "WAIC", "LOO")
        values: Dictionary mapping model names to metric values
        differences: Optional dictionary of pairwise differences between models
        standard_errors: Optional dictionary of standard errors for the metrics
        metadata: Optional dictionary for additional metadata
    """

    metric_name: str
    values: Dict[str, float]
    differences: Optional[Dict[str, Dict[str, float]]] = None
    standard_errors: Optional[Dict[str, float]] = None
    metadata: Optional[Dict[str, Any]] = None

    def best_model(self) -> str:
        """
        Return the name of the best model according to this metric.

        For information criteria (lower is better):
        - WAIC, LOO: Returns model with lowest value

        For Bayes factors (higher is better):
        - Bayes factor: Returns model with highest value

        Returns:
            Name of the best model
        """
        if self.metric_name.lower() in ["waic", "loo", "dic"]:
            # For information criteria, lower is better
            return min(self.values.items(), key=lambda x: x[1])[0]
        else:
            # For Bayes factors, higher is better
            return max(self.values.items(), key=lambda x: x[1])[0]

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert comparison results to a pandas DataFrame.

        Returns:
            DataFrame with models as rows and metrics as columns
        """
        data = {
            "model": list(self.values.keys()),
            self.metric_name: list(self.values.values()),
        }

        if self.standard_errors:
            data[f"{self.metric_name}_se"] = [
                self.standard_errors.get(model, np.nan)
                for model in self.values.keys()
            ]

        return pd.DataFrame(data)


class BayesianModelComparison:
    """
    Tools for Bayesian model comparison and criticism.

    This class provides methods for comparing Bayesian models using various
    information criteria and cross-validation techniques, including WAIC
    (Widely Applicable Information Criterion), LOO (Leave-One-Out cross-validation),
    and Bayes factors.

    The class is designed to work with PyroVelocity's modular architecture and
    leverages the arviz library for many of the computations.
    """

    def __init__(self, name: str = "model_comparison"):
        """
        Initialize the model comparison tool.

        Args:
            name: A unique name for this component instance
        """
        self.name = name

    @staticmethod
    def _extract_log_likelihood(
        model: PyroVelocityModel,
        posterior_samples: Dict[str, torch.Tensor],
        data: Dict[str, torch.Tensor],
        num_samples: int = 100,
    ) -> torch.Tensor:
        """
        Extract log likelihood values from a model for given posterior samples.

        Args:
            model: PyroVelocityModel instance
            posterior_samples: Dictionary of posterior samples
            data: Dictionary of observed data
            num_samples: Number of posterior samples to use

        Returns:
            Tensor of log likelihood values with shape [num_samples, num_observations]
        """
        # Extract observations and predictions
        observations = data.get("observations", None)
        if observations is None:
            raise ValueError("Data dictionary must contain 'observations' key")

        # Get likelihood model from the PyroVelocityModel
        likelihood_model = model.likelihood_model

        # Determine the actual number of samples from the posterior_samples
        if posterior_samples and len(posterior_samples) > 0:
            first_key = next(iter(posterior_samples))
            actual_num_samples = posterior_samples[first_key].shape[0]
            # Use the minimum of the provided num_samples and actual available samples
            num_samples = min(num_samples, actual_num_samples)
        else:
            raise ValueError("posterior_samples dictionary is empty")

        # Initialize log likelihood tensor
        num_observations = observations.shape[0]
        log_likes = torch.zeros((num_samples, num_observations))

        # Compute log likelihood for each posterior sample
        for i in range(num_samples):
            # Extract parameters for this sample
            sample_params = {k: v[i] for k, v in posterior_samples.items()}

            # Generate predictions using the dynamics model
            predictions = model.dynamics_model.forward(
                context={"parameters": sample_params, **data}
            ).get("predictions", None)

            if predictions is None:
                raise ValueError(
                    "Dynamics model did not return 'predictions' key"
                )

            # Use PyTorch tensors for likelihood computation
            scale_factors = data.get("scale_factors", None)

            # Compute log likelihood
            log_prob = likelihood_model.log_prob(
                observations=observations,
                predictions=predictions,
                scale_factors=scale_factors,
            )

            # Ensure result is a PyTorch tensor
            if not isinstance(log_prob, torch.Tensor):
                log_prob = torch.tensor(np.array(log_prob))

            log_likes[i] = log_prob

        return log_likes

    @jaxtyped
    @beartype
    def compute_waic(
        self,
        model: PyroVelocityModel,
        posterior_samples: Dict[str, torch.Tensor],
        data: Dict[str, torch.Tensor],
        pointwise: bool = False,
    ) -> Union[float, Tuple[float, torch.Tensor]]:
        """
        Compute the Widely Applicable Information Criterion (WAIC).

        WAIC is a fully Bayesian approach for estimating the out-of-sample
        expectation, using the computed log-likelihood evaluated at the
        posterior simulations of the parameter values.

        Args:
            model: PyroVelocityModel instance
            posterior_samples: Dictionary of posterior samples
            data: Dictionary of observed data
            pointwise: Whether to return pointwise WAIC values

        Returns:
            WAIC value (lower is better) and optionally pointwise values
        """
        # Extract log likelihood values
        log_likes = self._extract_log_likelihood(model, posterior_samples, data)

        # Convert to numpy for arviz
        log_likes_np = log_likes.numpy()

        # Compute WAIC using arviz
        waic_data = az.data.convert_to_dataset(
            {"log_likelihood": log_likes_np[None, ...]}
        )
        waic_result = az.waic(waic_data, pointwise=True)

        if pointwise:
            return waic_result.waic, torch.tensor(waic_result.pointwise)
        else:
            return waic_result.waic

    @jaxtyped
    @beartype
    def compute_loo(
        self,
        model: PyroVelocityModel,
        posterior_samples: Dict[str, torch.Tensor],
        data: Dict[str, torch.Tensor],
        pointwise: bool = False,
    ) -> Union[float, Tuple[float, torch.Tensor]]:
        """
        Compute Leave-One-Out (LOO) cross-validation.

        LOO cross-validation approximates the expected log predictive density
        using Pareto smoothed importance sampling (PSIS).

        Args:
            model: PyroVelocityModel instance
            posterior_samples: Dictionary of posterior samples
            data: Dictionary of observed data
            pointwise: Whether to return pointwise LOO values

        Returns:
            LOO value (lower is better) and optionally pointwise values
        """
        # Extract log likelihood values
        log_likes = self._extract_log_likelihood(model, posterior_samples, data)

        # Convert to numpy for arviz
        log_likes_np = log_likes.numpy()

        # Compute LOO using arviz
        loo_data = az.data.convert_to_dataset(
            {"log_likelihood": log_likes_np[None, ...]}
        )
        loo_result = az.loo(loo_data, pointwise=True)

        if pointwise:
            return loo_result.loo, torch.tensor(loo_result.pointwise)
        else:
            return loo_result.loo

    @beartype
    def compute_bayes_factor(
        self,
        model1: PyroVelocityModel,
        model2: PyroVelocityModel,
        data: Dict[str, torch.Tensor],
        num_samples: int = 1000,
    ) -> float:
        """
        Compute the Bayes factor between two models.

        The Bayes factor is the ratio of the marginal likelihoods of two models,
        and provides a measure of the evidence in favor of one model over another.

        Args:
            model1: First PyroVelocityModel instance
            model2: Second PyroVelocityModel instance
            data: Dictionary of observed data
            num_samples: Number of samples for marginal likelihood estimation

        Returns:
            Bayes factor (BF > 1 favors model1, BF < 1 favors model2)
        """
        # Compute marginal likelihoods using importance sampling
        log_ml1 = self._compute_log_marginal_likelihood(
            model1, data, num_samples
        )
        log_ml2 = self._compute_log_marginal_likelihood(
            model2, data, num_samples
        )

        # Compute Bayes factor
        log_bf = log_ml1 - log_ml2
        return np.exp(log_bf)

    def _compute_log_marginal_likelihood(
        self,
        model: PyroVelocityModel,
        data: Dict[str, torch.Tensor],
        num_samples: int = 1000,
    ) -> float:
        """
        Compute the log marginal likelihood of a model.

        This method uses importance sampling to estimate the marginal likelihood.

        Args:
            model: PyroVelocityModel instance
            data: Dictionary of observed data
            num_samples: Number of samples for estimation

        Returns:
            Log marginal likelihood
        """

        # Define model and guide functions for Pyro
        def model_fn(data=data):
            # Forward pass through the model
            return model.forward(
                x=data.get("x", None),
                time_points=data.get("time_points", None),
                cell_state=data.get("cell_state", None),
            )

        def guide_fn(data=data):
            # Forward pass through the guide
            return model.guide(
                x=data.get("x", None),
                time_points=data.get("time_points", None),
                cell_state=data.get("cell_state", None),
            )

        # Compute log marginal likelihood using Pyro's importance sampling
        # Note: Using Importance class directly as importance_sampling is not available
        importance = pyro.infer.Importance(
            model_fn, guide=guide_fn, num_samples=num_samples
        )
        importance_results = importance.run()
        log_ml = importance_results.log_mean()

        return log_ml.item()

    @beartype
    def compare_models(
        self,
        models: Dict[str, PyroVelocityModel],
        posterior_samples: Dict[str, Dict[str, torch.Tensor]],
        data: Dict[str, torch.Tensor],
        metric: str = "waic",
    ) -> ComparisonResult:
        """
        Compare multiple models using the specified metric.

        Args:
            models: Dictionary mapping model names to PyroVelocityModel instances
            posterior_samples: Dictionary mapping model names to posterior samples
            data: Dictionary of observed data
            metric: Comparison metric to use ("waic", "loo")

        Returns:
            ComparisonResult containing the comparison results
        """
        metric = metric.lower()
        values = {}
        standard_errors = {}

        # Compute metric for each model
        for model_name, model in models.items():
            model_samples = posterior_samples.get(model_name, {})
            
            if not model_samples:
                logger.warning(f"No posterior samples found for model: {model_name}")
                continue

            if metric == "waic":
                # Compute WAIC
                waic_data = az.data.convert_to_dataset(
                    {
                        "log_likelihood": self._extract_log_likelihood(
                            model, model_samples, data
                        ).numpy()[None, ...]
                    }
                )
                waic_result = az.waic(waic_data)
                values[model_name] = waic_result.waic
                standard_errors[model_name] = waic_result.waic_se
            
            elif metric == "loo":
                # Compute LOO
                loo_data = az.data.convert_to_dataset(
                    {
                        "log_likelihood": self._extract_log_likelihood(
                            model, model_samples, data
                        ).numpy()[None, ...]
                    }
                )
                loo_result = az.loo(loo_data)
                values[model_name] = loo_result.loo
                standard_errors[model_name] = loo_result.loo_se
            
            else:
                raise ValueError(f"Unknown metric: {metric}")

        # Compute pairwise differences
        differences = {}
        for model1_name in values:
            differences[model1_name] = {}
            for model2_name in values:
                if model1_name != model2_name:
                    diff = values[model1_name] - values[model2_name]
                    differences[model1_name][model2_name] = diff

        # Create and return ComparisonResult
        return ComparisonResult(
            metric_name=metric.upper(),
            values=values,
            differences=differences,
            standard_errors=standard_errors,
        )

    @beartype
    def compare_models_bayes_factors(
        self,
        models: Dict[str, PyroVelocityModel],
        data: Dict[str, torch.Tensor],
        reference_model: Optional[str] = None,
        num_samples: int = 1000,
    ) -> ComparisonResult:
        """
        Compare multiple models using Bayes factors.

        Args:
            models: Dictionary mapping model names to PyroVelocityModel instances
            data: Dictionary of observed data
            reference_model: Name of the reference model to compare against
            num_samples: Number of samples for marginal likelihood estimation

        Returns:
            ComparisonResult containing the Bayes factors
        """
        model_names = list(models.keys())
        if reference_model is None:
            reference_model = model_names[0]
        elif reference_model not in model_names:
            raise ValueError(f"Reference model '{reference_model}' not found in models")

        # Compute log marginal likelihood for each model
        log_mls = {}
        for model_name, model in models.items():
            log_ml = self._compute_log_marginal_likelihood(
                model, data, num_samples
            )
            log_mls[model_name] = log_ml

        # Compute Bayes factors using the reference model
        bayes_factors = {}
        for model_name in model_names:
            # Compute Bayes factor
            log_bf = log_mls[model_name] - log_mls[reference_model]
            bf = np.exp(log_bf)
            bayes_factors[model_name] = bf

        # Create and return ComparisonResult
        return ComparisonResult(
            metric_name="Bayes Factor",
            values=bayes_factors,
            metadata={"reference_model": reference_model},
        )


def select_best_model(
    comparison_result: ComparisonResult,
    threshold: float = 2.0,
) -> Tuple[str, bool]:
    """
    Select the best model from a comparison result.

    Args:
        comparison_result: ComparisonResult from a model comparison
        threshold: Threshold for determining significance

    Returns:
        Tuple of (best_model_name, is_significant)
    """
    # Get the best model according to the metric
    best_model = comparison_result.best_model()
    
    # Check if the difference is significant
    is_significant = False
    
    if comparison_result.differences:
        # For each other model, check if the difference is significant
        best_model_diffs = comparison_result.differences.get(best_model, {})
        for other_model, diff in best_model_diffs.items():
            # For information criteria (WAIC, LOO), lower is better
            if comparison_result.metric_name.upper() in ["WAIC", "LOO"]:
                # Best model should have smaller value, so diff should be negative
                # is_significant = all differences are below negative threshold
                if diff < -threshold:
                    is_significant = True
                else:
                    # If any difference is not significant, overall result is not significant
                    is_significant = False
                    break
            # For Bayes factors, higher is better
            else:
                # Best model should have larger value
                # Convert difference to ratio for Bayes factors
                ratio = np.exp(diff) if diff != 0 else 0
                if ratio > threshold:
                    is_significant = True
                else:
                    is_significant = False
                    break
    
    return best_model, is_significant


def create_comparison_table(
    comparison_results: List[ComparisonResult],
) -> pd.DataFrame:
    """
    Create a comparison table from multiple comparison results.

    Args:
        comparison_results: List of ComparisonResult objects

    Returns:
        DataFrame with models as rows and metrics as columns
    """
    # Get unique model names from all comparison results
    model_names = set()
    for result in comparison_results:
        model_names.update(result.values.keys())
    model_names = sorted(list(model_names))
    
    # Create DataFrame with model names as index
    df = pd.DataFrame(index=model_names)
    
    # Add each metric as a column
    for result in comparison_results:
        # Add values column
        metric_name = result.metric_name
        df[metric_name] = [
            result.values.get(model, np.nan) for model in model_names
        ]
        
        # Add standard errors column if available
        if result.standard_errors:
            df[f"{metric_name}_se"] = [
                result.standard_errors.get(model, np.nan) for model in model_names
            ]
    
    # Reset index to make model names a column
    df = df.reset_index().rename(columns={"index": "model"})
    
    return df


def compute_waic(
    model: PyroVelocityModel,
    posterior_samples: Dict[str, torch.Tensor],
    data: Dict[str, torch.Tensor],
    pointwise: bool = False,
) -> Union[float, Tuple[float, torch.Tensor]]:
    """
    Convenience function to compute WAIC without instantiating a comparison object.

    Args:
        model: PyroVelocityModel instance
        posterior_samples: Dictionary of posterior samples
        data: Dictionary of observed data
        pointwise: Whether to return pointwise WAIC values

    Returns:
        WAIC value (lower is better) and optionally pointwise values
    """
    comparison = BayesianModelComparison()
    return comparison.compute_waic(model, posterior_samples, data, pointwise)


def compute_loo(
    model: PyroVelocityModel,
    posterior_samples: Dict[str, torch.Tensor],
    data: Dict[str, torch.Tensor],
    pointwise: bool = False,
) -> Union[float, Tuple[float, torch.Tensor]]:
    """
    Convenience function to compute LOO without instantiating a comparison object.

    Args:
        model: PyroVelocityModel instance
        posterior_samples: Dictionary of posterior samples
        data: Dictionary of observed data
        pointwise: Whether to return pointwise LOO values

    Returns:
        LOO value (lower is better) and optionally pointwise values
    """
    comparison = BayesianModelComparison()
    return comparison.compute_loo(model, posterior_samples, data, pointwise)
