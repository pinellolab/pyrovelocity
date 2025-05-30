"""
Validation utilities for PyroVelocity parameter recovery validation.

This module provides helper functions for analyzing and visualizing parameter
recovery validation results from the PyroVelocity modular implementation.
"""

from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from beartype import beartype


@beartype
def create_validation_summary(validation_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create summary statistics from validation results.

    Args:
        validation_results: Dictionary of validation results from validate_parameter_recovery()

    Returns:
        Dictionary containing summary statistics
    """
    # Extract results excluding summary
    results = {k: v for k, v in validation_results.items() if k != '_summary'}
    
    if not results:
        return {
            'total_count': 0,
            'successful_count': 0,
            'overall_success_rate': 0.0,
            'mean_correlation': 0.0,
            'mean_error': float('inf'),
            'by_pattern': {}
        }
    
    # Collect metrics
    correlations = []
    errors = []
    successful_count = 0
    pattern_metrics = {}
    
    for key, result in results.items():
        if 'error' not in result and 'recovery_metrics' in result:
            metrics = result['recovery_metrics']
            correlation = metrics.get('overall_correlation', 0.0)
            error = metrics.get('mean_absolute_error', float('inf'))
            success = result.get('success', False)
            pattern = result.get('pattern', 'unknown')
            
            correlations.append(correlation)
            errors.append(error)
            if success:
                successful_count += 1
            
            # Track by pattern
            if pattern not in pattern_metrics:
                pattern_metrics[pattern] = {
                    'correlations': [],
                    'errors': [],
                    'successes': 0,
                    'total': 0
                }
            
            pattern_metrics[pattern]['correlations'].append(correlation)
            pattern_metrics[pattern]['errors'].append(error)
            pattern_metrics[pattern]['total'] += 1
            if success:
                pattern_metrics[pattern]['successes'] += 1
    
    # Compute overall statistics
    mean_correlation = np.mean(correlations) if correlations else 0.0
    mean_error = np.mean(errors) if errors else float('inf')
    overall_success_rate = successful_count / len(results) if results else 0.0
    
    # Compute pattern-wise statistics
    by_pattern = {}
    for pattern, metrics in pattern_metrics.items():
        by_pattern[pattern] = {
            'correlation': np.mean(metrics['correlations']),
            'error': np.mean(metrics['errors']),
            'success_rate': metrics['successes'] / metrics['total'] if metrics['total'] > 0 else 0.0,
            'count': metrics['total']
        }
    
    return {
        'total_count': len(results),
        'successful_count': successful_count,
        'overall_success_rate': overall_success_rate,
        'mean_correlation': mean_correlation,
        'mean_error': mean_error,
        'by_pattern': by_pattern
    }


@beartype
def assess_validation_status(validation_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Assess overall validation status and readiness.

    Args:
        validation_results: Dictionary of validation results from validate_parameter_recovery()

    Returns:
        Dictionary containing validation status assessment
    """
    summary = create_validation_summary(validation_results)
    
    success_rate = summary['overall_success_rate']
    mean_correlation = summary['mean_correlation']
    
    # Determine status
    if success_rate >= 0.9 and mean_correlation >= 0.9:
        status = "EXCELLENT"
        status_emoji = "🏆"
        description = "Parameter recovery is excellent. Model is ready for production use."
        ready_for_production = True
    elif success_rate >= 0.8 and mean_correlation >= 0.8:
        status = "GOOD"
        status_emoji = "✅"
        description = "Parameter recovery is good. Model is suitable for most applications."
        ready_for_production = True
    elif success_rate >= 0.6 and mean_correlation >= 0.6:
        status = "ACCEPTABLE"
        status_emoji = "⚠️"
        description = "Parameter recovery is acceptable but may need improvement for critical applications."
        ready_for_production = False
    else:
        status = "POOR"
        status_emoji = "❌"
        description = "Parameter recovery is poor. Model needs significant improvement before use."
        ready_for_production = False
    
    return {
        'status': status,
        'status_emoji': status_emoji,
        'description': description,
        'ready_for_production': ready_for_production,
        'success_rate': success_rate,
        'mean_correlation': mean_correlation
    }


@beartype
def display_validation_table(validation_results: Dict[str, Any]) -> None:
    """
    Display validation results in a formatted table.

    Args:
        validation_results: Dictionary of validation results from validate_parameter_recovery()
    """
    # Extract results excluding summary
    results = {k: v for k, v in validation_results.items() if k != '_summary'}
    
    if not results:
        print("No validation results to display.")
        return
    
    print("\n📋 PARAMETER RECOVERY VALIDATION RESULTS")
    print("=" * 80)
    
    # Create table data
    table_data = []
    for key, result in results.items():
        if 'error' in result:
            table_data.append([
                key,
                result.get('pattern', 'unknown'),
                "ERROR",
                "N/A",
                "N/A",
                result['error'][:50] + "..." if len(result['error']) > 50 else result['error']
            ])
        else:
            metrics = result.get('recovery_metrics', {})
            training = result.get('training_result', {})
            
            correlation = metrics.get('overall_correlation', 0.0)
            success_rate = metrics.get('success_rate', 0.0)
            epochs = training.get('num_epochs', 0)
            success = "✅" if result.get('success', False) else "❌"
            
            table_data.append([
                key,
                result.get('pattern', 'unknown'),
                success,
                f"{correlation:.3f}",
                f"{success_rate:.1%}",
                f"{epochs} epochs"
            ])
    
    # Print table
    headers = ["Dataset", "Pattern", "Success", "Correlation", "Param Success", "Training"]
    col_widths = [max(len(str(row[i])) for row in [headers] + table_data) + 2 for i in range(len(headers))]
    
    # Print header
    header_row = "".join(f"{headers[i]:<{col_widths[i]}}" for i in range(len(headers)))
    print(header_row)
    print("-" * len(header_row))
    
    # Print data rows
    for row in table_data:
        data_row = "".join(f"{str(row[i]):<{col_widths[i]}}" for i in range(len(row)))
        print(data_row)
    
    # Print summary
    summary = create_validation_summary(validation_results)
    print("\n📊 SUMMARY:")
    print(f"  Overall Success Rate: {summary['overall_success_rate']:.1%}")
    print(f"  Mean Correlation: {summary['mean_correlation']:.3f}")
    print(f"  Successful Datasets: {summary['successful_count']}/{summary['total_count']}")
    
    # Print pattern-wise summary
    if summary['by_pattern']:
        print(f"\n📈 By Pattern:")
        for pattern, metrics in summary['by_pattern'].items():
            print(f"  {pattern.upper()}: {metrics['correlation']:.3f} correlation, {metrics['success_rate']:.1%} success")


@beartype
def plot_validation_summary(
    validation_results: Dict[str, Any],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 12),
    show_success_threshold: bool = True
) -> plt.Figure:
    """
    Generate comprehensive validation summary plots.

    Args:
        validation_results: Dictionary of validation results from validate_parameter_recovery()
        save_path: Optional path to save the figure
        figsize: Figure size (width, height)
        show_success_threshold: Whether to show success threshold lines

    Returns:
        matplotlib Figure object
    """
    # Extract results excluding summary
    results = {k: v for k, v in validation_results.items() if k != '_summary'}
    
    if not results:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.text(0.5, 0.5, 'No validation results to plot', 
                ha='center', va='center', transform=ax.transAxes, fontsize=16)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        return fig
    
    # Create 2x2 subplot layout
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
    
    # Collect data
    patterns = []
    correlations = []
    success_rates = []
    epochs = []
    dataset_names = []
    
    for key, result in results.items():
        if 'error' not in result and 'recovery_metrics' in result:
            patterns.append(result.get('pattern', 'unknown'))
            correlations.append(result['recovery_metrics'].get('overall_correlation', 0.0))
            success_rates.append(result['recovery_metrics'].get('success_rate', 0.0))
            epochs.append(result.get('training_result', {}).get('num_epochs', 0))
            dataset_names.append(key)
    
    if not correlations:
        for ax in [ax1, ax2, ax3, ax4]:
            ax.text(0.5, 0.5, 'No valid results to plot', 
                    ha='center', va='center', transform=ax.transAxes)
        return fig
    
    # Plot 1: Correlation by pattern
    pattern_corrs = {}
    for pattern, corr in zip(patterns, correlations):
        if pattern not in pattern_corrs:
            pattern_corrs[pattern] = []
        pattern_corrs[pattern].append(corr)
    
    pattern_names = list(pattern_corrs.keys())
    pattern_means = [np.mean(pattern_corrs[p]) for p in pattern_names]
    pattern_stds = [np.std(pattern_corrs[p]) for p in pattern_names]
    
    bars1 = ax1.bar(pattern_names, pattern_means, yerr=pattern_stds, capsize=5, alpha=0.7)
    ax1.set_title('Parameter Recovery Correlation by Pattern', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Correlation')
    ax1.set_ylim(0, 1)
    if show_success_threshold:
        ax1.axhline(y=0.9, color='red', linestyle='--', alpha=0.7, label='Success Threshold')
        ax1.legend()
    
    # Plot 2: Success rates by pattern
    pattern_success = {}
    for pattern, success in zip(patterns, success_rates):
        if pattern not in pattern_success:
            pattern_success[pattern] = []
        pattern_success[pattern].append(success)
    
    success_means = [np.mean(pattern_success[p]) for p in pattern_names]
    success_stds = [np.std(pattern_success[p]) for p in pattern_names]
    
    bars2 = ax2.bar(pattern_names, success_means, yerr=success_stds, capsize=5, alpha=0.7, color='orange')
    ax2.set_title('Parameter Success Rate by Pattern', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Success Rate')
    ax2.set_ylim(0, 1)
    if show_success_threshold:
        ax2.axhline(y=0.9, color='red', linestyle='--', alpha=0.7, label='Success Threshold')
        ax2.legend()
    
    # Plot 3: Parameter recovery scatter
    colors = plt.cm.Set1(np.linspace(0, 1, len(set(patterns))))
    pattern_colors = {p: colors[i] for i, p in enumerate(set(patterns))}
    
    for i, (pattern, corr, success) in enumerate(zip(patterns, correlations, success_rates)):
        ax3.scatter(corr, success, c=[pattern_colors[pattern]], 
                   label=pattern if pattern not in [p for p in patterns[:i]] else "", 
                   alpha=0.7, s=60)
    
    ax3.set_xlabel('Overall Correlation')
    ax3.set_ylabel('Parameter Success Rate')
    ax3.set_title('Parameter Recovery: Correlation vs Success Rate', fontsize=14, fontweight='bold')
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    if show_success_threshold:
        ax3.axhline(y=0.9, color='red', linestyle='--', alpha=0.7)
        ax3.axvline(x=0.9, color='red', linestyle='--', alpha=0.7)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Training convergence summary
    pattern_epochs = {}
    for pattern, epoch in zip(patterns, epochs):
        if pattern not in pattern_epochs:
            pattern_epochs[pattern] = []
        pattern_epochs[pattern].append(epoch)
    
    epoch_means = [np.mean(pattern_epochs[p]) for p in pattern_names]
    epoch_stds = [np.std(pattern_epochs[p]) for p in pattern_names]
    
    bars4 = ax4.bar(pattern_names, epoch_means, yerr=epoch_stds, capsize=5, alpha=0.7, color='green')
    ax4.set_title('Training Epochs by Pattern', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Epochs')
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
    
    return fig
