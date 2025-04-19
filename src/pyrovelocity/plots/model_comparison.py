"""
Visualization tools for model comparison in PyroVelocity.

This module provides visualization functions for comparing Bayesian models
in the PyroVelocity framework, including comparison plots for information criteria,
diagnostic plots, and posterior predictive check visualizations.

The module is designed to work with the BayesianModelComparison class and
ComparisonResult objects from the pyrovelocity.models.comparison module.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from anndata import AnnData
from beartype import beartype
from jaxtyping import Array, Float, jaxtyped
from matplotlib.figure import Figure

from pyrovelocity.models.comparison import BayesianModelComparison, ComparisonResult

logger = logging.getLogger(__name__)


@jaxtyped
@beartype
def plot_model_comparison(
    comparison_result: ComparisonResult,
    title: Optional[str] = None,
    figsize: Tuple[Union[float, int], Union[float, int]] = (10, 6),
    save_path: Optional[Union[str, Path]] = None,
    show_differences: bool = True,
    highlight_best: bool = True,
    threshold: float = 2.0,
) -> Figure:
    """
    Create a bar plot comparing models based on a ComparisonResult.
    
    Args:
        comparison_result: ComparisonResult from model comparison
        title: Optional title for the plot
        figsize: Figure size (width, height) in inches
        save_path: Optional path to save the figure
        show_differences: Whether to show pairwise differences
        highlight_best: Whether to highlight the best model
        threshold: Threshold for considering a difference significant
            
    Returns:
        Matplotlib Figure object
    """
    # Convert comparison result to DataFrame
    df = comparison_result.to_dataframe()
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Determine the best model
    best_model = comparison_result.best_model()
    
    # Determine if differences are significant
    is_significant = False
    if comparison_result.differences:
        metric_name = comparison_result.metric_name.lower()
        
        if metric_name in ["waic", "loo", "dic"]:
            # For information criteria, check absolute differences
            for model, diffs in comparison_result.differences.items():
                for other_model, diff in diffs.items():
                    if abs(diff) > threshold:
                        is_significant = True
                        break
        else:
            # For Bayes factors, check ratios
            best_value = comparison_result.values[best_model]
            for model, value in comparison_result.values.items():
                if model != best_model:
                    if best_value / value > threshold:
                        is_significant = True
                        break
    
    # Create bar plot
    colors = ['#1f77b4'] * len(df)  # Default color for all bars
    
    # Highlight the best model if requested
    if highlight_best:
        best_idx = df[df['model'] == best_model].index[0]
        colors[best_idx] = '#2ca02c' if is_significant else '#ff7f0e'
    
    # Plot bars
    bars = ax.bar(df['model'], df[comparison_result.metric_name], color=colors)
    
    # Add error bars if standard errors are available
    se_col = f"{comparison_result.metric_name}_se"
    if se_col in df.columns:
        ax.errorbar(
            df['model'], 
            df[comparison_result.metric_name], 
            yerr=df[se_col], 
            fmt='none', 
            color='black', 
            capsize=5
        )
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            height + 0.1,
            f'{height:.2f}',
            ha='center', 
            va='bottom',
            fontsize=9
        )
    
    # Add pairwise differences if requested
    if show_differences and comparison_result.differences:
        y_max = max(df[comparison_result.metric_name]) * 1.2
        
        for i, (model1, diffs) in enumerate(comparison_result.differences.items()):
            for j, (model2, diff) in enumerate(diffs.items()):
                # Only show significant differences
                if abs(diff) > threshold:
                    idx1 = df[df['model'] == model1].index[0]
                    idx2 = df[df['model'] == model2].index[0]
                    
                    # Draw an arrow or line between the bars
                    x1 = idx1
                    x2 = idx2
                    y = y_max - (i * 0.1 * y_max)
                    
                    ax.annotate(
                        f'diff = {diff:.2f}',
                        xy=(x2, y),
                        xytext=(x1, y),
                        arrowprops=dict(arrowstyle='<->', color='red' if abs(diff) > threshold else 'gray'),
                        ha='center',
                        va='bottom',
                        fontsize=8
                    )
    
    # Set title and labels
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"Model Comparison - {comparison_result.metric_name}")
    
    ax.set_ylabel(comparison_result.metric_name)
    ax.set_xlabel("Model")
    
    # Add a legend explaining the colors
    if highlight_best:
        if is_significant:
            ax.bar(0, 0, color='#2ca02c', label=f'Best model (significant: diff > {threshold})')
        else:
            ax.bar(0, 0, color='#ff7f0e', label=f'Best model (not significant: diff <= {threshold})')
        ax.bar(0, 0, color='#1f77b4', label='Other models')
        ax.legend(loc='best')
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig


@jaxtyped
@beartype
def plot_model_comparison_grid(
    comparison_results: List[ComparisonResult],
    titles: Optional[List[str]] = None,
    figsize: Tuple[Union[float, int], Union[float, int]] = (15, 10),
    save_path: Optional[Union[str, Path]] = None,
    highlight_best: bool = True,
    threshold: float = 2.0,
) -> Figure:
    """
    Create a grid of model comparison plots for multiple metrics.
    
    Args:
        comparison_results: List of ComparisonResult objects
        titles: Optional list of titles for each subplot
        figsize: Figure size (width, height) in inches
        save_path: Optional path to save the figure
        highlight_best: Whether to highlight the best model
        threshold: Threshold for considering a difference significant
            
    Returns:
        Matplotlib Figure object
    """
    n_plots = len(comparison_results)
    
    # Determine grid layout
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_plots == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Create each subplot
    for i, result in enumerate(comparison_results):
        if i < len(axes):
            ax = axes[i]
            
            # Convert comparison result to DataFrame
            df = result.to_dataframe()
            
            # Determine the best model
            best_model = result.best_model()
            
            # Determine if differences are significant
            is_significant = False
            if result.differences:
                metric_name = result.metric_name.lower()
                
                if metric_name in ["waic", "loo", "dic"]:
                    # For information criteria, check absolute differences
                    for model, diffs in result.differences.items():
                        for other_model, diff in diffs.items():
                            if abs(diff) > threshold:
                                is_significant = True
                                break
                else:
                    # For Bayes factors, check ratios
                    best_value = result.values[best_model]
                    for model, value in result.values.items():
                        if model != best_model:
                            if best_value / value > threshold:
                                is_significant = True
                                break
            
            # Create bar plot
            colors = ['#1f77b4'] * len(df)  # Default color for all bars
            
            # Highlight the best model if requested
            if highlight_best:
                best_idx = df[df['model'] == best_model].index[0]
                colors[best_idx] = '#2ca02c' if is_significant else '#ff7f0e'
            
            # Plot bars
            bars = ax.bar(df['model'], df[result.metric_name], color=colors)
            
            # Add error bars if standard errors are available
            se_col = f"{result.metric_name}_se"
            if se_col in df.columns:
                ax.errorbar(
                    df['model'], 
                    df[result.metric_name], 
                    yerr=df[se_col], 
                    fmt='none', 
                    color='black', 
                    capsize=5
                )
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.,
                    height + 0.1,
                    f'{height:.2f}',
                    ha='center', 
                    va='bottom',
                    fontsize=8
                )
            
            # Set title and labels
            if titles and i < len(titles):
                ax.set_title(titles[i])
            else:
                ax.set_title(f"{result.metric_name}")
            
            ax.set_ylabel(result.metric_name)
            ax.set_xlabel("Model")
            
            # Rotate x-axis labels if there are many models
            if len(df) > 3:
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Hide unused subplots
    for i in range(n_plots, len(axes)):
        axes[i].set_visible(False)
    
    # Add a legend explaining the colors
    if highlight_best:
        fig.subplots_adjust(bottom=0.15)
        legend_ax = fig.add_axes([0.15, 0.05, 0.7, 0.03])
        legend_ax.axis('off')
        legend_ax.bar(0, 0, color='#2ca02c', label=f'Best model (significant: diff > {threshold})')
        legend_ax.bar(1, 0, color='#ff7f0e', label=f'Best model (not significant: diff <= {threshold})')
        legend_ax.bar(2, 0, color='#1f77b4', label='Other models')
        legend_ax.legend(loc='center', ncol=3)
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig


@jaxtyped
@beartype
def plot_pointwise_comparison(
    model1_name: str,
    model2_name: str,
    model1_pointwise: np.ndarray,
    model2_pointwise: np.ndarray,
    metric_name: str = "WAIC",
    figsize: Tuple[Union[float, int], Union[float, int]] = (10, 6),
    save_path: Optional[Union[str, Path]] = None,
) -> Figure:
    """
    Create a scatter plot comparing pointwise metric values between two models.
    
    Args:
        model1_name: Name of the first model
        model2_name: Name of the second model
        model1_pointwise: Pointwise metric values for the first model
        model2_pointwise: Pointwise metric values for the second model
        metric_name: Name of the metric being compared
        figsize: Figure size (width, height) in inches
        save_path: Optional path to save the figure
            
    Returns:
        Matplotlib Figure object
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create scatter plot
    ax.scatter(model1_pointwise, model2_pointwise, alpha=0.6)
    
    # Add diagonal line
    min_val = min(np.min(model1_pointwise), np.min(model2_pointwise))
    max_val = max(np.max(model1_pointwise), np.max(model2_pointwise))
    margin = (max_val - min_val) * 0.1
    ax.plot([min_val - margin, max_val + margin], [min_val - margin, max_val + margin], 'k--', alpha=0.5)
    
    # Set axis limits
    ax.set_xlim(min_val - margin, max_val + margin)
    ax.set_ylim(min_val - margin, max_val + margin)
    
    # Set title and labels
    ax.set_title(f"Pointwise {metric_name} Comparison")
    ax.set_xlabel(f"{model1_name} Pointwise {metric_name}")
    ax.set_ylabel(f"{model2_name} Pointwise {metric_name}")
    
    # Add text showing the number of points above/below the diagonal
    n_above = np.sum(model2_pointwise > model1_pointwise)
    n_below = np.sum(model1_pointwise > model2_pointwise)
    n_equal = np.sum(np.isclose(model1_pointwise, model2_pointwise))
    total = len(model1_pointwise)
    
    text = (
        f"Points where {model2_name} > {model1_name}: {n_above} ({n_above/total:.1%})\n"
        f"Points where {model1_name} > {model2_name}: {n_below} ({n_below/total:.1%})\n"
        f"Points where {model1_name} ~= {model2_name}: {n_equal} ({n_equal/total:.1%})"
    )
    
    ax.text(
        0.05, 0.95, text,
        transform=ax.transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig


@jaxtyped
@beartype
def plot_posterior_predictive_check(
    observed_data: np.ndarray,
    predicted_data: np.ndarray,
    model_name: str = "Model",
    figsize: Tuple[Union[float, int], Union[float, int]] = (10, 6),
    save_path: Optional[Union[str, Path]] = None,
) -> Figure:
    """
    Create a posterior predictive check plot comparing observed and predicted data.
    
    Args:
        observed_data: Observed data values
        predicted_data: Predicted data values from posterior samples
        model_name: Name of the model
        figsize: Figure size (width, height) in inches
        save_path: Optional path to save the figure
            
    Returns:
        Matplotlib Figure object
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate summary statistics for predicted data
    pred_mean = np.mean(predicted_data, axis=0)
    pred_lower = np.percentile(predicted_data, 2.5, axis=0)
    pred_upper = np.percentile(predicted_data, 97.5, axis=0)
    
    # Sort observed data and corresponding predictions
    sort_idx = np.argsort(observed_data)
    observed_sorted = observed_data[sort_idx]
    pred_mean_sorted = pred_mean[sort_idx]
    pred_lower_sorted = pred_lower[sort_idx]
    pred_upper_sorted = pred_upper[sort_idx]
    
    # Plot observed data
    ax.scatter(np.arange(len(observed_sorted)), observed_sorted, 
               alpha=0.7, label='Observed', color='#1f77b4', s=20)
    
    # Plot predicted mean
    ax.plot(np.arange(len(pred_mean_sorted)), pred_mean_sorted, 
            'o-', alpha=0.7, label='Predicted (mean)', color='#ff7f0e', markersize=3)
    
    # Plot prediction intervals
    ax.fill_between(np.arange(len(pred_mean_sorted)), 
                    pred_lower_sorted, pred_upper_sorted, 
                    alpha=0.2, color='#ff7f0e', label='95% Prediction Interval')
    
    # Set title and labels
    ax.set_title(f"Posterior Predictive Check - {model_name}")
    ax.set_xlabel("Sorted Observation Index")
    ax.set_ylabel("Value")
    ax.legend()
    
    # Calculate and display metrics
    mse = np.mean((observed_data - pred_mean) ** 2)
    mae = np.mean(np.abs(observed_data - pred_mean))
    coverage = np.mean((observed_data >= pred_lower) & (observed_data <= pred_upper))
    
    metrics_text = (
        f"MSE: {mse:.4f}\n"
        f"MAE: {mae:.4f}\n"
        f"95% Coverage: {coverage:.1%}"
    )
    
    ax.text(
        0.05, 0.95, metrics_text,
        transform=ax.transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig


@jaxtyped
@beartype
def plot_diagnostic_metrics(
    models: Dict[str, Dict[str, float]],
    figsize: Tuple[Union[float, int], Union[float, int]] = (12, 8),
    save_path: Optional[Union[str, Path]] = None,
) -> Figure:
    """
    Create a radar chart comparing diagnostic metrics across models.
    
    Args:
        models: Dictionary mapping model names to dictionaries of metrics
        figsize: Figure size (width, height) in inches
        save_path: Optional path to save the figure
            
    Returns:
        Matplotlib Figure object
    """
    # Extract metrics and model names
    metrics = set()
    for model_metrics in models.values():
        metrics.update(model_metrics.keys())
    metrics = sorted(list(metrics))
    
    model_names = list(models.keys())
    
    # Number of variables
    N = len(metrics)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    
    # Compute angle for each metric
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Initialize the plot
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    # Draw one axis per variable and add labels
    plt.xticks(angles[:-1], metrics)
    
    # Draw ylabels
    ax.set_rlabel_position(0)
    
    # Plot data
    for i, model in enumerate(model_names):
        values = [models[model].get(metric, 0) for metric in metrics]
        values += values[:1]  # Close the loop
        
        # Normalize values to [0, 1] for better visualization
        min_vals = [min(models[m].get(metric, float('inf')) for m in model_names) for metric in metrics]
        max_vals = [max(models[m].get(metric, 0) for m in model_names) for metric in metrics]
        
        # Avoid division by zero
        ranges = [max(0.001, max_val - min_val) for min_val, max_val in zip(min_vals, max_vals)]
        norm_values = [(val - min_val) / range_val for val, min_val, range_val in zip(values[:-1], min_vals, ranges)]
        norm_values += norm_values[:1]  # Close the loop
        
        ax.plot(angles, norm_values, linewidth=2, linestyle='solid', label=model)
        ax.fill(angles, norm_values, alpha=0.1)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.title("Model Diagnostic Metrics Comparison", size=15, y=1.1)
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig


@jaxtyped
@beartype
def plot_waic_loo_comparison(
    comparison_instance: BayesianModelComparison,
    models: Dict[str, object],
    posterior_samples: Dict[str, Dict[str, object]],
    data: Dict[str, object],
    figsize: Tuple[Union[float, int], Union[float, int]] = (12, 10),
    save_path: Optional[Union[str, Path]] = None,
) -> Figure:
    """
    Create a comparison plot showing both WAIC and LOO metrics.
    
    Args:
        comparison_instance: BayesianModelComparison instance
        models: Dictionary mapping model names to model objects
        posterior_samples: Dictionary mapping model names to posterior samples
        data: Dictionary of observed data
        figsize: Figure size (width, height) in inches
        save_path: Optional path to save the figure
            
    Returns:
        Matplotlib Figure object
    """
    # Compute WAIC and LOO for all models
    waic_result = comparison_instance.compare_models(
        models=models,
        posterior_samples=posterior_samples,
        data=data,
        metric="waic",
    )
    
    loo_result = comparison_instance.compare_models(
        models=models,
        posterior_samples=posterior_samples,
        data=data,
        metric="loo",
    )
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot WAIC
    waic_df = waic_result.to_dataframe()
    best_waic_model = waic_result.best_model()
    colors = ['#1f77b4'] * len(waic_df)
    best_idx = waic_df[waic_df['model'] == best_waic_model].index[0]
    colors[best_idx] = '#2ca02c'
    
    ax1.bar(waic_df['model'], waic_df['WAIC'], color=colors)
    if 'WAIC_se' in waic_df.columns:
        ax1.errorbar(
            waic_df['model'], 
            waic_df['WAIC'], 
            yerr=waic_df['WAIC_se'], 
            fmt='none', 
            color='black', 
            capsize=5
        )
    
    ax1.set_title("WAIC Comparison")
    ax1.set_ylabel("WAIC")
    ax1.set_xlabel("Model")
    
    # Plot LOO
    loo_df = loo_result.to_dataframe()
    best_loo_model = loo_result.best_model()
    colors = ['#1f77b4'] * len(loo_df)
    best_idx = loo_df[loo_df['model'] == best_loo_model].index[0]
    colors[best_idx] = '#2ca02c'
    
    ax2.bar(loo_df['model'], loo_df['LOO'], color=colors)
    if 'LOO_se' in loo_df.columns:
        ax2.errorbar(
            loo_df['model'], 
            loo_df['LOO'], 
            yerr=loo_df['LOO_se'], 
            fmt='none', 
            color='black', 
            capsize=5
        )
    
    ax2.set_title("LOO Comparison")
    ax2.set_ylabel("LOO")
    ax2.set_xlabel("Model")
    
    # Add a note if the best model is different between WAIC and LOO
    if best_waic_model != best_loo_model:
        fig.text(
            0.5, 0.01,
            f"Note: Best model differs between metrics (WAIC: {best_waic_model}, LOO: {best_loo_model})",
            ha='center',
            fontsize=12,
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.2)
        )
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig