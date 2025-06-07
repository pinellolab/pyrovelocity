#!/usr/bin/env python3
"""
Prior Hyperparameter Calibration for PyroVelocity Dimensionless Parameterization

This script systematically analyzes and calibrates prior hyperparameters for the 
dimensionless PyroVelocity parameterization to restore balanced coverage of gene 
expression patterns. It implements the 5 gene expression patterns defined in the 
documentation and provides optimization recommendations.

The script is designed to precede execution of prior-predictive-check.py and help 
diagnose why the current dimensionless parameterization produces less uniform 
coverage of gene expression patterns.
"""

import torch
import pyro
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, Tuple, Any, List
from pathlib import Path
from beartype import beartype

# Import PyroVelocity components
from pyrovelocity.models.modular.factory import create_piecewise_activation_model
from pyrovelocity.plots.predictive_checks import combine_pdfs
from pyrovelocity.plots.parameter_metadata import get_parameter_label
from pyrovelocity.styles import configure_matplotlib_style

configure_matplotlib_style()

def _latex_safe_text(text: str) -> str:
    """
    Make text safe for LaTeX rendering by escaping problematic characters.

    Args:
        text: Input text that may contain LaTeX-problematic characters

    Returns:
        LaTeX-safe text
    """
    # Replace underscores with escaped underscores
    text = text.replace('_', r'\_')

    # Replace other problematic characters
    replacements = {
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '^': r'\^{}',
        '~': r'\~{}',
        '{': r'\{',
        '}': r'\}',
    }

    for char, replacement in replacements.items():
        text = text.replace(char, replacement)

    return text

def _format_pattern_name(pattern_name: str) -> str:
    """
    Format pattern names for display in legends and titles.

    Args:
        pattern_name: Raw pattern name (e.g., 'pre_activation', 'decay_only')

    Returns:
        Formatted pattern name suitable for LaTeX rendering
    """
    # Pattern name mappings for better display
    pattern_mappings = {
        'activation': 'Activation',
        'pre_activation': 'Pre-activation',
        'decay_only': 'Decay Only',
        'transient': 'Transient',
        'sustained': 'Sustained'
    }

    formatted = pattern_mappings.get(pattern_name, pattern_name.replace('_', ' ').title())
    return _latex_safe_text(formatted)
# Set style for plots
# plt.style.use('seaborn-v0_8')
# sns.set_palette("husl")

# Disable LaTeX rendering to avoid Unicode issues
# plt.rcParams['text.usetex'] = False

# Set random seeds for reproducibility
torch.manual_seed(42)
pyro.set_rng_seed(42)
np.random.seed(42)


class PriorHyperparameterCalibrator:
    """
    Calibrator for PyroVelocity prior hyperparameters in dimensionless parameterization.
    
    This class analyzes the current prior settings, evaluates pattern coverage,
    and provides optimization recommendations for balanced gene expression patterns.
    """
    
    def __init__(self, save_path: str = "reports/docs/prior_calibration"):
        """
        Initialize the calibrator.
        
        Args:
            save_path: Directory to save analysis outputs
        """
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        
        # Current dimensionless prior hyperparameters (from priors.py) - UPDATED AFTER CALIBRATION
        self.current_priors = {
            # Piecewise activation parameters (for pattern analysis)
            'R_on': {'loc': 0.693, 'scale': 0.35},       # log(2.0), fold-change (LogNormal) - UPDATED
            't_on_star': {'loc': 0.5, 'scale': 0.8},     # Normal(0.5, 0.8²), allows negatives - CALIBRATED
            'delta_star': {'loc': -0.8, 'scale': 0.45},  # log(0.45) (LogNormal) - CALIBRATED
            'gamma_star': {'loc': -0.223, 'scale': 0.3}, # log(0.8) (LogNormal) - UPDATED for lower velocity magnitudes

            # Hierarchical time structure parameters
            'T_M_star': {'alpha': 4.0, 'beta': 0.08},    # Gamma(4.0, 0.08), mean = 50
            't_loc': {'alpha': 1.0, 'beta': 2.0},        # Gamma(1.0, 2.0), mean = 0.5
            't_scale': {'alpha': 1.0, 'beta': 4.0},      # Gamma(1.0, 4.0), mean = 0.25

            # Technical parameters
            'U_0i': {'loc': 2.3, 'scale': 0.4},          # log(10) (LogNormal) - REDUCED from log(100) for realistic single-cell count scales
            'lambda_j': {'loc': 0.0, 'scale': 0.2},      # log(1.0) (LogNormal)
        }
        
        # Gene expression pattern definitions (from documentation)
        self.pattern_constraints = {
            'activation': {
                'R_on': ('>', 3.0),
                't_on_star': ('>', 0.0),
                't_on_star_upper': ('<', 0.4),
                'delta_star': ('>', 0.3),
            },
            'pre_activation': {
                'R_on': ('>', 2.0),
                't_on_star': ('<', 0.0),
                'delta_star': ('>', 0.3),
            },
            'decay_only': {
                'R_on': ('>', 1.5),
                't_on_star_beyond': ('>', 1.0),  # Beyond observation window
            },
            'transient': {
                'R_on': ('>', 2.0),
                't_on_star': ('>', 0.0),
                't_on_star_upper': ('<', 0.5),
                'delta_star': ('<', 0.4),
            },
            'sustained': {
                'R_on': ('>', 2.0),
                't_on_star': ('>', 0.0),
                't_on_star_upper': ('<', 0.3),
                'delta_star': ('>', 0.5),
            }
        }
        
        # Create model for dynamics calculations
        self.model = create_piecewise_activation_model()
        
    @beartype
    def calculate_lognormal_cdf_probability(
        self,
        loc: float,
        scale: float,
        threshold: float,
        upper: bool = True
    ) -> float:
        """
        Calculate P(X > threshold) or P(X < threshold) for LogNormal(loc, scale).

        Args:
            loc: Location parameter (log-space mean)
            scale: Scale parameter (log-space std)
            threshold: Threshold value
            upper: If True, calculate P(X > threshold), else P(X < threshold)

        Returns:
            Probability value between 0 and 1
        """
        if upper:
            return 1 - stats.lognorm.cdf(threshold, s=scale, scale=np.exp(loc))
        else:
            return stats.lognorm.cdf(threshold, s=scale, scale=np.exp(loc))

    @beartype
    def calculate_gamma_cdf_probability(
        self,
        alpha: float,
        beta: float,
        threshold: float,
        upper: bool = True
    ) -> float:
        """
        Calculate P(X > threshold) or P(X < threshold) for Gamma(alpha, beta).

        Args:
            alpha: Shape parameter
            beta: Rate parameter (not scale!)
            threshold: Threshold value
            upper: If True, calculate P(X > threshold), else P(X < threshold)

        Returns:
            Probability value between 0 and 1
        """
        if upper:
            return 1 - stats.gamma.cdf(threshold, a=alpha, scale=1/beta)
        else:
            return stats.gamma.cdf(threshold, a=alpha, scale=1/beta)
    
    @beartype
    def calculate_normal_cdf_probability(
        self,
        loc: float,
        scale: float,
        threshold: float,
        upper: bool = True
    ) -> float:
        """
        Calculate P(X > threshold) or P(X < threshold) for Normal(loc, scale).

        Args:
            loc: Location parameter (mean)
            scale: Scale parameter (std)
            threshold: Threshold value
            upper: If True, calculate P(X > threshold), else P(X < threshold)

        Returns:
            Probability value between 0 and 1
        """
        if upper:
            return 1 - stats.norm.cdf(threshold, loc=loc, scale=scale)
        else:
            return stats.norm.cdf(threshold, loc=loc, scale=scale)

    @beartype
    def analyze_constraint_feasibility(self) -> Dict[str, Dict[str, float]]:
        """
        Analyze feasibility of pattern constraints with current priors.

        Returns:
            Dictionary with constraint feasibility probabilities for each pattern
        """
        results = {}

        print("\n" + "="*50)
        print("CONSTRAINT FEASIBILITY ANALYSIS")
        print("="*50)

        for pattern_name, constraints in self.pattern_constraints.items():
            pattern_results = {}
            joint_probability = 1.0

            print(f"\n{_format_pattern_name(pattern_name).upper()} PATTERN:")

            for constraint_name, (operator, threshold) in constraints.items():
                # Handle special constraint names
                param_name = constraint_name
                if constraint_name.endswith('_upper'):
                    param_name = constraint_name.replace('_upper', '')
                elif constraint_name.endswith('_beyond'):
                    param_name = constraint_name.replace('_beyond', '')

                if param_name not in self.current_priors:
                    continue

                prior = self.current_priors[param_name]
                upper = (operator == '>')

                # Calculate probability based on distribution type
                if param_name == 't_on_star':  # Normal distribution
                    prob = self.calculate_normal_cdf_probability(
                        prior['loc'], prior['scale'], threshold, upper=upper
                    )
                else:  # LogNormal distribution
                    prob = self.calculate_lognormal_cdf_probability(
                        prior['loc'], prior['scale'], threshold, upper=upper
                    )

                pattern_results[constraint_name] = prob
                joint_probability *= prob

                # Status indicator
                status = "✅" if prob > 0.2 else "⚠️" if prob > 0.1 else "❌"
                print(f"  P({param_name} {operator} {threshold}) = {prob:.3f} {status}")

            pattern_results['joint_probability'] = joint_probability
            results[pattern_name] = pattern_results

            # Overall status
            status = "✅" if joint_probability > 0.2 else "⚠️" if joint_probability > 0.05 else "❌"
            print(f"  JOINT PROBABILITY: {joint_probability:.4f} {status}")

            if joint_probability < 0.05:
                print(f"  ⚠️  CRITICAL: Joint probability too low for reliable sampling!")

        return results

    @beartype
    def calculate_hpdi_ranges(self, confidence: float = 0.95) -> Dict[str, Dict[str, float]]:
        """
        Calculate 95% HPDI ranges for all prior distributions.

        Args:
            confidence: Confidence level for HPDI calculation (default: 0.95)

        Returns:
            Dictionary with HPDI ranges for each parameter
        """
        alpha = 1 - confidence
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        hpdi_ranges = {}

        print(f"\n" + "="*60)
        print(f"95% HIGHEST POSTERIOR DENSITY INTERVAL (HPDI) ANALYSIS")
        print("="*60)
        print(f"Confidence level: {confidence*100:.1f}%")
        print(f"Percentiles: [{lower_percentile:.1f}%, {upper_percentile:.1f}%]")

        for param_name, prior_config in self.current_priors.items():
            if 'loc' in prior_config and 'scale' in prior_config:
                # LogNormal or Normal distribution
                if param_name == 't_on_star':  # Normal distribution
                    lower = stats.norm.ppf(alpha/2, loc=prior_config['loc'], scale=prior_config['scale'])
                    upper = stats.norm.ppf(1-alpha/2, loc=prior_config['loc'], scale=prior_config['scale'])
                    dist_type = "Normal"
                    mean_val = prior_config['loc']
                else:  # LogNormal distribution
                    lower = stats.lognorm.ppf(alpha/2, s=prior_config['scale'], scale=np.exp(prior_config['loc']))
                    upper = stats.lognorm.ppf(1-alpha/2, s=prior_config['scale'], scale=np.exp(prior_config['loc']))
                    dist_type = "LogNormal"
                    mean_val = np.exp(prior_config['loc'] + 0.5 * prior_config['scale']**2)

            elif 'alpha' in prior_config and 'beta' in prior_config:
                # Gamma distribution
                lower = stats.gamma.ppf(alpha/2, a=prior_config['alpha'], scale=1/prior_config['beta'])
                upper = stats.gamma.ppf(1-alpha/2, a=prior_config['alpha'], scale=1/prior_config['beta'])
                dist_type = "Gamma"
                mean_val = prior_config['alpha'] / prior_config['beta']
            else:
                continue

            hpdi_ranges[param_name] = {
                'lower': lower,
                'upper': upper,
                'mean': mean_val,
                'distribution': dist_type,
                'width': upper - lower
            }

            # Display with biological interpretation
            print(f"\n{param_name} ~ {dist_type}:")
            print(f"  95% HPDI: [{lower:.3f}, {upper:.3f}]")
            print(f"  Mean: {mean_val:.3f}")
            print(f"  Width: {upper - lower:.3f}")

            # Add biological interpretation
            self._add_biological_interpretation(param_name, lower, upper, mean_val)

        return hpdi_ranges

    @beartype
    def calculate_hpdi_for_single_param(self, param_name: str, confidence: float = 0.95) -> Dict[str, float]:
        """
        Calculate HPDI for a single parameter.

        Args:
            param_name: Name of the parameter
            confidence: Confidence level (default: 0.95)

        Returns:
            Dictionary with HPDI information
        """
        if param_name not in self.current_priors:
            return {}

        alpha = 1 - confidence
        prior_config = self.current_priors[param_name]

        if 'loc' in prior_config and 'scale' in prior_config:
            # LogNormal or Normal distribution
            if param_name == 't_on_star':  # Normal distribution
                lower = stats.norm.ppf(alpha/2, loc=prior_config['loc'], scale=prior_config['scale'])
                upper = stats.norm.ppf(1-alpha/2, loc=prior_config['loc'], scale=prior_config['scale'])
                mean_val = prior_config['loc']
                mode_val = prior_config['loc']
            else:  # LogNormal distribution
                lower = stats.lognorm.ppf(alpha/2, s=prior_config['scale'], scale=np.exp(prior_config['loc']))
                upper = stats.lognorm.ppf(1-alpha/2, s=prior_config['scale'], scale=np.exp(prior_config['loc']))
                mean_val = np.exp(prior_config['loc'] + 0.5 * prior_config['scale']**2)
                mode_val = np.exp(prior_config['loc'] - prior_config['scale']**2)

        elif 'alpha' in prior_config and 'beta' in prior_config:
            # Gamma distribution
            lower = stats.gamma.ppf(alpha/2, a=prior_config['alpha'], scale=1/prior_config['beta'])
            upper = stats.gamma.ppf(1-alpha/2, a=prior_config['alpha'], scale=1/prior_config['beta'])
            mean_val = prior_config['alpha'] / prior_config['beta']
            mode_val = max(0, (prior_config['alpha'] - 1) / prior_config['beta']) if prior_config['alpha'] > 1 else 0
        else:
            return {}

        return {
            'lower': lower,
            'upper': upper,
            'mean': mean_val,
            'mode': mode_val,
            'width': upper - lower
        }

    @beartype
    def _add_biological_interpretation(self, param_name: str, lower: float, upper: float, mean: float) -> None:
        """Add biological interpretation for parameter ranges."""
        interpretations = {
            'R_on': f"  Interpretation: Activation fold-change from {lower:.1f}× to {upper:.1f}× (mean: {mean:.1f}×)",
            't_on_star': f"  Interpretation: Onset time from {lower:.2f} to {upper:.2f} (negative = pre-activation)",
            'delta_star': f"  Interpretation: Activation duration from {lower:.2f} to {upper:.2f} (fraction of timeline)",
            'gamma_star': f"  Interpretation: Relative degradation rate from {lower:.2f} to {upper:.2f} (1.0 = balanced)",
            'T_M_star': f"  Interpretation: Maximum timeline from {lower:.1f} to {upper:.1f} time units",
            't_loc': f"  Interpretation: Population time center from {lower:.2f} to {upper:.2f}",
            't_scale': f"  Interpretation: Population time spread from {lower:.2f} to {upper:.2f}",
            'U_0i': f"  Interpretation: Concentration scale from {lower:.0f} to {upper:.0f} counts",
            'lambda_j': f"  Interpretation: Capture efficiency from {lower:.2f} to {upper:.2f} (1.0 = perfect)"
        }

        if param_name in interpretations:
            print(interpretations[param_name])

            # Add warnings for potentially problematic ranges
            if param_name == 'R_on' and (lower < 1.2 or upper > 10):
                print(f"  ⚠️  Warning: Fold-change range may include weak ({lower:.1f}×) or extreme ({upper:.1f}×) values")
            elif param_name == 'lambda_j' and upper > 2.0:
                print(f"  ⚠️  Warning: Capture efficiency > 2.0 may be unrealistic for single-cell data")
            elif param_name == 'U_0i' and (lower < 10 or upper > 1000):
                print(f"  ⚠️  Warning: Concentration scale may be outside typical single-cell range")





    @beartype
    def generate_pattern_examples(self, n_examples: int = 3) -> Dict[str, List[Dict[str, torch.Tensor]]]:
        """
        Generate representative parameter sets for each expression pattern.

        Args:
            n_examples: Number of examples to generate per pattern

        Returns:
            Dictionary with parameter sets for each pattern
        """
        pattern_examples = {}

        print("\n" + "="*50)
        print("GENERATING PATTERN EXAMPLES")
        print("="*50)

        for pattern_name, constraints in self.pattern_constraints.items():
            formatted_pattern = _format_pattern_name(pattern_name)
            print(f"\nGenerating {n_examples} examples for {formatted_pattern.upper()} pattern...")

            examples = []
            attempts = 0
            max_attempts = 10000

            while len(examples) < n_examples and attempts < max_attempts:
                attempts += 1

                # Sample parameters from priors
                R_on = np.exp(np.random.normal(
                    self.current_priors['R_on']['loc'],
                    self.current_priors['R_on']['scale']
                ))
                t_on_star = np.random.normal(
                    self.current_priors['t_on_star']['loc'],
                    self.current_priors['t_on_star']['scale']
                )
                delta_star = np.exp(np.random.normal(
                    self.current_priors['delta_star']['loc'],
                    self.current_priors['delta_star']['scale']
                ))
                gamma_star = np.exp(np.random.normal(
                    self.current_priors['gamma_star']['loc'],
                    self.current_priors['gamma_star']['scale']
                ))

                # Check if parameters satisfy pattern constraints
                satisfies_constraints = True

                for constraint_name, (operator, threshold) in constraints.items():
                    # Handle special constraint names
                    param_name = constraint_name
                    if constraint_name.endswith('_upper'):
                        param_name = constraint_name.replace('_upper', '')
                    elif constraint_name.endswith('_beyond'):
                        param_name = constraint_name.replace('_beyond', '')

                    # Get parameter value
                    if param_name == 'R_on':
                        value = R_on
                    elif param_name == 't_on_star':
                        value = t_on_star
                    elif param_name == 'delta_star':
                        value = delta_star
                    elif param_name == 'gamma_star':
                        value = gamma_star
                    else:
                        continue

                    # Check constraint
                    if operator == '>' and value <= threshold:
                        satisfies_constraints = False
                        break
                    elif operator == '<' and value >= threshold:
                        satisfies_constraints = False
                        break

                if satisfies_constraints:
                    examples.append({
                        'R_on': torch.tensor(R_on),
                        't_on_star': torch.tensor(t_on_star),
                        'delta_star': torch.tensor(delta_star),
                        'gamma_star': torch.tensor(gamma_star),
                        'alpha_off': torch.tensor(1.0),  # Fixed in dimensionless parameterization
                        'alpha_on': torch.tensor(R_on),  # Since alpha_off = 1.0
                    })

            if len(examples) < n_examples:
                print(f"  ⚠️  Warning: Only generated {len(examples)}/{n_examples} examples after {max_attempts} attempts")
            else:
                print(f"  ✅ Successfully generated {len(examples)} examples")

            pattern_examples[pattern_name] = examples

        return pattern_examples

    @beartype
    def plot_pattern_time_courses(
        self,
        pattern_examples: Dict[str, List[Dict[str, torch.Tensor]]]
    ) -> plt.Figure:
        """
        Plot time courses for each pattern example.

        Args:
            pattern_examples: Dictionary with parameter sets for each pattern

        Returns:
            matplotlib Figure object
        """
        n_patterns = len(pattern_examples)
        fig, axes = plt.subplots(n_patterns, 3, figsize=(15, 4*n_patterns))

        if n_patterns == 1:
            axes = axes.reshape(1, -1)

        # Time points for evaluation
        t_star = torch.linspace(0, 1.0, 100)

        pattern_colors = ['red', 'blue', 'green', 'orange', 'purple']

        for pattern_idx, (pattern_name, examples) in enumerate(pattern_examples.items()):
            if not examples:
                continue

            color = pattern_colors[pattern_idx % len(pattern_colors)]
            formatted_pattern = _format_pattern_name(pattern_name)

            # Plot multiple examples for this pattern
            for example_idx, params in enumerate(examples):
                # Compute time courses using piecewise dynamics
                u_star, s_star = self._compute_time_course(t_star, params)

                # Plot unspliced
                axes[pattern_idx, 0].plot(
                    t_star.numpy(), u_star.numpy(),
                    color=color, alpha=0.7, linewidth=2,
                    label=f'Example {example_idx+1}' if example_idx < 3 else None
                )

                # Plot spliced
                axes[pattern_idx, 1].plot(
                    t_star.numpy(), s_star.numpy(),
                    color=color, alpha=0.7, linewidth=2,
                    label=f'Example {example_idx+1}' if example_idx < 3 else None
                )

                # Plot phase portrait
                axes[pattern_idx, 2].plot(
                    u_star.numpy(), s_star.numpy(),
                    color=color, alpha=0.7, linewidth=2,
                    label=f'Example {example_idx+1}' if example_idx < 3 else None
                )

            # Format axes with LaTeX-safe labels
            axes[pattern_idx, 0].set_xlabel(_latex_safe_text('Dimensionless Time (t*)'))
            axes[pattern_idx, 0].set_ylabel(_latex_safe_text('Unspliced (u*)'))
            axes[pattern_idx, 0].set_title(f'{formatted_pattern}: Unspliced')
            axes[pattern_idx, 0].grid(True, alpha=0.3)
            axes[pattern_idx, 0].legend()

            axes[pattern_idx, 1].set_xlabel(_latex_safe_text('Dimensionless Time (t*)'))
            axes[pattern_idx, 1].set_ylabel(_latex_safe_text('Spliced (s*)'))
            axes[pattern_idx, 1].set_title(f'{formatted_pattern}: Spliced')
            axes[pattern_idx, 1].grid(True, alpha=0.3)

            axes[pattern_idx, 2].set_xlabel(_latex_safe_text('Unspliced (u*)'))
            axes[pattern_idx, 2].set_ylabel(_latex_safe_text('Spliced (s*)'))
            axes[pattern_idx, 2].set_title(f'{formatted_pattern}: Phase Portrait')
            axes[pattern_idx, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    @beartype
    def _compute_time_course(
        self,
        t_star: torch.Tensor,
        params: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute time course using piecewise activation dynamics.

        Args:
            t_star: Time points to evaluate
            params: Parameter dictionary

        Returns:
            Tuple of (u_star, s_star) time courses
        """
        # Extract parameters
        alpha_off = params['alpha_off']
        alpha_on = params['alpha_on']
        gamma_star = params['gamma_star']
        t_on_star = params['t_on_star']
        delta_star = params['delta_star']

        # Initialize output tensors
        u_star = torch.zeros_like(t_star)
        s_star = torch.zeros_like(t_star)

        # Phase 1: Off state (t* < t*_on)
        phase1_mask = t_star < t_on_star
        u_star[phase1_mask] = 1.0  # Fixed reference state
        s_star[phase1_mask] = 1.0 / gamma_star

        # Phase 2: On state (t*_on ≤ t* < t*_on + δ*)
        phase2_mask = (t_star >= t_on_star) & (t_star < t_on_star + delta_star)
        if phase2_mask.any():
            tau_on = t_star[phase2_mask] - t_on_star
            u_on, s_on = self._compute_on_phase_solution(tau_on, alpha_on, gamma_star)
            u_star[phase2_mask] = u_on
            s_star[phase2_mask] = s_on

        # Phase 3: Return to off state (t* ≥ t*_on + δ*)
        phase3_mask = t_star >= t_on_star + delta_star
        if phase3_mask.any():
            tau_off = t_star[phase3_mask] - (t_on_star + delta_star)
            u_off, s_off = self._compute_off_phase_solution(
                tau_off, alpha_off, alpha_on, gamma_star, delta_star
            )
            u_star[phase3_mask] = u_off
            s_star[phase3_mask] = s_off

        return u_star, s_star

    @beartype
    def _compute_on_phase_solution(
        self,
        tau_on: torch.Tensor,
        alpha_on: torch.Tensor,
        gamma_star: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute analytical solution for ON phase."""
        # Initial conditions: u*_0 = 1.0, s*_0 = 1.0/γ*
        u_0 = 1.0
        s_0 = 1.0 / gamma_star

        # Analytical solution for ON phase
        exp_tau = torch.exp(-tau_on)
        exp_gamma_tau = torch.exp(-gamma_star * tau_on)

        u_on = alpha_on + (u_0 - alpha_on) * exp_tau

        s_on = (alpha_on / gamma_star +
                (s_0 - alpha_on / gamma_star) * exp_gamma_tau +
                (alpha_on - u_0) * (exp_tau - exp_gamma_tau) / (gamma_star - 1))

        return u_on, s_on

    @beartype
    def _compute_off_phase_solution(
        self,
        tau_off: torch.Tensor,
        alpha_off: torch.Tensor,
        alpha_on: torch.Tensor,
        gamma_star: torch.Tensor,
        delta_star: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute analytical solution for return to OFF phase."""
        # Initial conditions: endpoint values from ON phase
        u_end, s_end = self._compute_on_phase_solution(
            delta_star, alpha_on, gamma_star
        )

        # Analytical solution for OFF phase
        exp_tau = torch.exp(-tau_off)
        exp_gamma_tau = torch.exp(-gamma_star * tau_off)

        u_off = alpha_off + (u_end - alpha_off) * exp_tau

        s_off = (alpha_off / gamma_star +
                (s_end - alpha_off / gamma_star) * exp_gamma_tau +
                (alpha_off - u_end) * (exp_tau - exp_gamma_tau) / (gamma_star - 1))

        return u_off, s_off

    @beartype
    def generate_optimization_recommendations(
        self,
        feasibility_results: Dict[str, Dict[str, float]]
    ) -> Dict[str, Any]:
        """
        Generate specific optimization recommendations based on feasibility analysis.

        Args:
            feasibility_results: Results from constraint feasibility analysis

        Returns:
            Dictionary with optimization recommendations
        """
        recommendations = {
            'summary': {},
            'critical_issues': [],
            'optimization_options': {},
            'recommended_action': {}
        }

        # Analyze overall pattern coverage
        joint_probs = {pattern: results['joint_probability']
                      for pattern, results in feasibility_results.items()}

        critical_patterns = [pattern for pattern, prob in joint_probs.items() if prob < 0.05]
        marginal_patterns = [pattern for pattern, prob in joint_probs.items() if 0.05 <= prob < 0.2]
        acceptable_patterns = [pattern for pattern, prob in joint_probs.items() if prob >= 0.2]

        recommendations['summary'] = {
            'critical_patterns': critical_patterns,
            'marginal_patterns': marginal_patterns,
            'acceptable_patterns': acceptable_patterns,
            'overall_coverage': len(acceptable_patterns) / len(joint_probs)
        }

        # Identify critical issues
        for pattern in critical_patterns:
            pattern_results = feasibility_results[pattern]
            problematic_constraints = []

            for constraint, prob in pattern_results.items():
                if constraint != 'joint_probability' and prob < 0.1:
                    problematic_constraints.append((constraint, prob))

            recommendations['critical_issues'].append({
                'pattern': pattern,
                'joint_probability': pattern_results['joint_probability'],
                'problematic_constraints': problematic_constraints
            })

        # Generate optimization options
        recommendations['optimization_options'] = {
            'option_1_relax_constraints': {
                'description': 'Relax pattern constraints to improve feasibility',
                'changes': self._suggest_constraint_relaxations(feasibility_results),
                'pros': ['Quick implementation', 'Maintains current priors'],
                'cons': ['Reduces pattern distinctness', 'May affect biological interpretability']
            },
            'option_2_adjust_priors': {
                'description': 'Adjust prior hyperparameters to support current constraints',
                'changes': self._suggest_prior_adjustments(feasibility_results),
                'pros': ['Preserves constraint interpretability', 'Maintains pattern distinctness'],
                'cons': ['Changes prior assumptions', 'Requires validation']
            },
            'option_3_hybrid': {
                'description': 'Balanced approach with minimal changes to both',
                'changes': {
                    'constraints': self._suggest_minimal_constraint_relaxations(feasibility_results),
                    'priors': self._suggest_minimal_prior_adjustments(feasibility_results)
                },
                'pros': ['Balanced trade-offs', 'Preserves most interpretability'],
                'cons': ['More complex implementation']
            }
        }

        # Recommend best action
        if len(critical_patterns) == 0:
            recommendations['recommended_action'] = {
                'action': 'no_change',
                'rationale': 'All patterns have acceptable feasibility (>20%)'
            }
        elif len(critical_patterns) <= 2:
            recommendations['recommended_action'] = {
                'action': 'option_3_hybrid',
                'rationale': f'Hybrid approach best for {len(critical_patterns)} critical patterns'
            }
        else:
            recommendations['recommended_action'] = {
                'action': 'option_2_adjust_priors',
                'rationale': f'Prior adjustment needed for {len(critical_patterns)} critical patterns'
            }

        return recommendations

    @beartype
    def _suggest_constraint_relaxations(self, feasibility_results: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Suggest constraint relaxations to improve feasibility."""
        relaxations = {}

        for pattern, results in feasibility_results.items():
            if results['joint_probability'] < 0.2:
                pattern_relaxations = {}

                # Find most problematic constraints
                for constraint, prob in results.items():
                    if constraint != 'joint_probability' and prob < 0.15:
                        if 'delta_star' in constraint:
                            if '<' in constraint:
                                pattern_relaxations[constraint] = ('increase_upper_bound', 0.1)
                            else:
                                pattern_relaxations[constraint] = ('decrease_lower_bound', 0.1)
                        elif 't_on_star' in constraint:
                            if '<' in constraint:
                                pattern_relaxations[constraint] = ('increase_upper_bound', 0.1)
                            else:
                                pattern_relaxations[constraint] = ('decrease_lower_bound', 0.1)

                if pattern_relaxations:
                    relaxations[pattern] = pattern_relaxations

        return relaxations

    @beartype
    def _suggest_prior_adjustments(self, feasibility_results: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Suggest prior hyperparameter adjustments."""
        adjustments = {}

        # Analyze which parameters are most problematic across patterns
        param_issues = {'R_on': 0, 't_on_star': 0, 'delta_star': 0}

        for pattern, results in feasibility_results.items():
            if results['joint_probability'] < 0.2:
                for constraint, prob in results.items():
                    if constraint != 'joint_probability' and prob < 0.15:
                        for param in param_issues.keys():
                            if param in constraint:
                                param_issues[param] += 1

        # Suggest adjustments for most problematic parameters
        if param_issues['delta_star'] > 1:
            adjustments['delta_star'] = {
                'loc': -0.9,  # Increase from -1.0 to -0.9 (log(0.41) vs log(0.37))
                'scale': 0.4,  # Increase from 0.35 to 0.4 for more spread
                'rationale': 'Increase mean and spread to better support pattern constraints'
            }

        if param_issues['t_on_star'] > 1:
            adjustments['t_on_star'] = {
                'loc': 0.15,  # Decrease from 0.2 to 0.15
                'scale': 0.7,  # Increase from 0.6 to 0.7 for more spread
                'rationale': 'Adjust to better balance positive and negative onset times'
            }

        return adjustments

    @beartype
    def _suggest_minimal_constraint_relaxations(self, feasibility_results: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Suggest minimal constraint relaxations."""
        relaxations = {}

        # Only relax the most critical constraints
        for pattern, results in feasibility_results.items():
            if results['joint_probability'] < 0.05:  # Only most critical
                for constraint, prob in results.items():
                    if constraint != 'joint_probability' and prob < 0.05:
                        if pattern == 'transient' and 'delta_star' in constraint and '<' in constraint:
                            relaxations[f'{pattern}_{constraint}'] = ('relax_to', 0.45)
                        elif pattern == 'sustained' and 'delta_star' in constraint and '>' in constraint:
                            relaxations[f'{pattern}_{constraint}'] = ('relax_to', 0.45)

        return relaxations

    @beartype
    def _suggest_minimal_prior_adjustments(self, feasibility_results: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Suggest minimal prior adjustments."""
        adjustments = {}

        # Count critical issues
        critical_delta_issues = 0
        for pattern, results in feasibility_results.items():
            if results['joint_probability'] < 0.05:
                for constraint, prob in results.items():
                    if 'delta_star' in constraint and prob < 0.05:
                        critical_delta_issues += 1

        if critical_delta_issues > 0:
            adjustments['delta_star'] = {
                'loc': -0.95,  # Modest increase from -1.0
                'scale': 0.37,  # Modest increase from 0.35
                'rationale': 'Minimal adjustment to improve delta_star constraint feasibility'
            }

        return adjustments

    @beartype
    def run_comprehensive_parameter_space_analysis(self) -> None:
        """
        Run comprehensive parameter space analysis including:
        1. Multi-dimensional parameter range analysis
        2. Joint distribution sampling and pattern classification
        3. Phase diagram mapping
        4. Conditional dependency analysis
        5. Hierarchical parameter impact assessment
        """
        print("Starting comprehensive parameter space analysis...")

        # Step 1: Multi-dimensional parameter sampling
        print("\n1. Multi-dimensional parameter sampling...")
        param_samples = self._sample_full_parameter_space(n_samples=50000)

        # Step 2: Pattern classification using soft scoring
        print("\n2. Pattern classification analysis...")
        pattern_scores = self._classify_patterns_soft_scoring(param_samples)

        # Step 3: Phase diagram mapping
        print("\n3. Phase diagram mapping...")
        phase_diagrams = self._create_phase_diagrams(param_samples, pattern_scores)

        # Step 4: Conditional dependency analysis
        print("\n4. Conditional dependency analysis...")
        dependency_analysis = self._analyze_parameter_dependencies(param_samples, pattern_scores)

        # Step 5: Hierarchical parameter impact
        print("\n5. Hierarchical parameter impact assessment...")
        hierarchical_impact = self._assess_hierarchical_impact(param_samples)

        # Step 6: Generate comprehensive visualizations
        print("\n6. Generating comprehensive visualizations...")
        self._create_comprehensive_plots(param_samples, pattern_scores, phase_diagrams, dependency_analysis, hierarchical_impact)

        # Step 7: Optimization recommendations
        print("\n7. Generating optimization recommendations...")
        recommendations = self._generate_comprehensive_recommendations(
            param_samples, pattern_scores, dependency_analysis
        )

        # Step 8: Save comprehensive report
        self._save_comprehensive_analysis_report(recommendations, pattern_scores)

        # Step 9: Combine PDFs into single comprehensive report
        print("\nCombining PDF outputs into comprehensive report...")
        try:
            combine_pdfs(
                pdf_directory=str(self.save_path),
                output_filename="comprehensive_prior_calibration_report.pdf",
                exclude_patterns=["combined_*.pdf", "comprehensive_*.pdf", "01_prior_distributions.pdf",
                                "02_hpdi_summary.pdf", "03_pattern_time_courses.pdf", "04_calibration_summary.pdf",
                                "07_calibration_summary.pdf"]
            )
        except Exception as e:
            print(f"Warning: Could not combine PDFs: {e}")

        print(f"\n✅ Comprehensive analysis complete! Results saved to: {self.save_path}")
        print("\nGenerated comprehensive report:")
        print("- comprehensive_prior_calibration_report.pdf")
        print("\nReport contents:")
        print("1. Clean Prior Distributions (with HPDI ranges and mode annotations)")
        print("2. Pattern Time Courses")
        print("3. Phase Diagrams")
        print("4. Parameter Correlation Analysis")
        print("5. Hierarchical Impact Analysis")
        print("6. Enhanced Prior Distributions (with pattern overlays)")
        print("7. Pattern Coverage Analysis")
        print("8. Calibration Summary (with pie charts)")

    @beartype
    def _sample_full_parameter_space(self, n_samples: int = 50000) -> Dict[str, torch.Tensor]:
        """Sample from the complete parameter space including hierarchical parameters."""
        print(f"  Sampling {n_samples} parameter sets from full prior space...")

        # Sample all parameters including hierarchical time structure
        samples = {}

        # Hierarchical time parameters
        samples['T_M_star'] = torch.distributions.Gamma(2.0, 0.5).sample((n_samples,))
        samples['t_loc'] = torch.distributions.Gamma(2.0, 5.0).sample((n_samples,))
        samples['t_scale'] = torch.distributions.Gamma(2.0, 10.0).sample((n_samples,))

        # Piecewise activation parameters
        samples['R_on'] = torch.distributions.LogNormal(
            self.current_priors['R_on']['loc'],
            self.current_priors['R_on']['scale']
        ).sample((n_samples,))

        samples['t_on_star'] = torch.distributions.Normal(
            self.current_priors['t_on_star']['loc'],
            self.current_priors['t_on_star']['scale']
        ).sample((n_samples,))

        samples['delta_star'] = torch.distributions.LogNormal(
            self.current_priors['delta_star']['loc'],
            self.current_priors['delta_star']['scale']
        ).sample((n_samples,))

        samples['gamma_star'] = torch.distributions.LogNormal(
            self.current_priors['gamma_star']['loc'],
            self.current_priors['gamma_star']['scale']
        ).sample((n_samples,))

        # Additional parameters
        samples['U_0i'] = torch.distributions.LogNormal(4.6, 0.5).sample((n_samples,))
        samples['lambda_j'] = torch.distributions.LogNormal(0.0, 0.2).sample((n_samples,))

        # Fixed parameters
        samples['alpha_off'] = torch.ones(n_samples)
        samples['alpha_on'] = samples['R_on'] * samples['alpha_off']

        return samples

    @beartype
    def _classify_patterns_soft_scoring(self, param_samples: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Classify patterns using soft scoring instead of hard constraints."""
        print("  Computing soft pattern scores for all samples...")

        n_samples = len(param_samples['R_on'])
        pattern_scores = {}

        for pattern_name in self.pattern_constraints.keys():
            scores = torch.zeros(n_samples)

            for i in range(n_samples):
                score = self._compute_soft_pattern_score(param_samples, i, pattern_name)
                scores[i] = score

            pattern_scores[pattern_name] = scores

        # Compute pattern assignments (highest scoring pattern for each sample)
        all_scores = torch.stack(list(pattern_scores.values()), dim=1)
        pattern_assignments = torch.argmax(all_scores, dim=1)
        pattern_scores['assignments'] = pattern_assignments

        # Compute pattern coverage
        pattern_names = list(self.pattern_constraints.keys())
        coverage = {}
        for i, pattern_name in enumerate(pattern_names):
            coverage[pattern_name] = (pattern_assignments == i).float().mean().item()

        pattern_scores['coverage'] = coverage
        print(f"  Pattern coverage: {coverage}")

        return pattern_scores

    @beartype
    def _compute_soft_pattern_score(self, param_samples: Dict[str, torch.Tensor], idx: int, pattern: str) -> float:
        """Compute soft membership score for a pattern using sigmoid functions."""
        R_on = param_samples['R_on'][idx].item()
        t_on_star = param_samples['t_on_star'][idx].item()
        delta_star = param_samples['delta_star'][idx].item()

        def sigmoid_score(value: float, threshold: float, direction: str, steepness: float = 5.0) -> float:
            """Sigmoid function for soft constraint scoring."""
            if direction == '>':
                return torch.sigmoid(torch.tensor(steepness * (value - threshold))).item()
            else:  # direction == '<'
                return torch.sigmoid(torch.tensor(steepness * (threshold - value))).item()

        # Pattern-specific scoring
        if pattern == 'activation':
            scores = [
                sigmoid_score(R_on, 3.0, '>'),
                sigmoid_score(t_on_star, 0.0, '>'),
                sigmoid_score(t_on_star, 0.4, '<'),
                sigmoid_score(delta_star, 0.3, '>')
            ]
        elif pattern == 'pre_activation':
            scores = [
                sigmoid_score(R_on, 2.0, '>'),
                sigmoid_score(t_on_star, 0.0, '<'),
                sigmoid_score(delta_star, 0.3, '>')
            ]
        elif pattern == 'decay_only':
            scores = [
                sigmoid_score(R_on, 1.5, '>'),
                sigmoid_score(t_on_star, 1.0, '>')  # Beyond observation window
            ]
        elif pattern == 'transient':
            scores = [
                sigmoid_score(R_on, 2.0, '>'),
                sigmoid_score(t_on_star, 0.0, '>'),
                sigmoid_score(t_on_star, 0.5, '<'),
                sigmoid_score(delta_star, 0.4, '<')
            ]
        elif pattern == 'sustained':
            scores = [
                sigmoid_score(R_on, 2.0, '>'),
                sigmoid_score(t_on_star, 0.0, '>'),
                sigmoid_score(t_on_star, 0.3, '<'),
                sigmoid_score(delta_star, 0.5, '>')
            ]
        else:
            scores = [0.0]

        return np.mean(scores)

    @beartype
    def _create_phase_diagrams(
        self,
        param_samples: Dict[str, torch.Tensor],
        pattern_scores: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        """Create 2D phase diagrams showing pattern boundaries."""
        print("  Creating 2D phase diagrams...")

        phase_diagrams = {}

        # Key parameter pairs for phase diagrams
        param_pairs = [
            ('R_on', 'delta_star'),
            ('t_on_star', 'delta_star'),
            ('R_on', 't_on_star'),
            ('gamma_star', 'delta_star')
        ]

        for param_x, param_y in param_pairs:
            x_vals = param_samples[param_x].numpy()
            y_vals = param_samples[param_y].numpy()
            assignments = pattern_scores['assignments'].numpy()

            phase_diagrams[f'{param_x}_vs_{param_y}'] = {
                'x_vals': x_vals,
                'y_vals': y_vals,
                'assignments': assignments,
                'x_label': param_x,
                'y_label': param_y
            }

        return phase_diagrams

    @beartype
    def _analyze_parameter_dependencies(
        self,
        param_samples: Dict[str, torch.Tensor],
        pattern_scores: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        """Analyze correlations and dependencies between parameters."""
        print("  Analyzing parameter dependencies...")

        # Convert to numpy for correlation analysis
        param_matrix = torch.stack([
            param_samples['R_on'],
            param_samples['t_on_star'],
            param_samples['delta_star'],
            param_samples['gamma_star'],
            param_samples['T_M_star'],
            param_samples['U_0i']
        ], dim=1).numpy()

        param_names = ['R_on', 't_on_star', 'delta_star', 'gamma_star', 'T_M_star', 'U_0i']

        # Compute correlation matrix
        correlation_matrix = np.corrcoef(param_matrix.T)

        # Analyze pattern-specific correlations
        pattern_correlations = {}
        assignments = pattern_scores['assignments'].numpy()

        for i, pattern_name in enumerate(self.pattern_constraints.keys()):
            mask = assignments == i
            if mask.sum() > 10:  # Need sufficient samples
                pattern_matrix = param_matrix[mask]
                pattern_corr = np.corrcoef(pattern_matrix.T)
                pattern_correlations[pattern_name] = pattern_corr

        return {
            'overall_correlation': correlation_matrix,
            'pattern_correlations': pattern_correlations,
            'param_names': param_names
        }

    @beartype
    def _assess_hierarchical_impact(self, param_samples: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Assess impact of hierarchical time parameters on pattern formation."""
        print("  Assessing hierarchical parameter impact...")

        # Analyze how T_M_star affects pattern boundaries
        T_M_values = param_samples['T_M_star'].numpy()
        t_on_values = param_samples['t_on_star'].numpy()

        # Compute effective onset times: t_on_effective = t_on_star * T_M_star
        t_on_effective = t_on_values * T_M_values

        # Analyze impact on pattern feasibility
        impact_analysis = {
            'T_M_range': (T_M_values.min(), T_M_values.max()),
            'T_M_mean_std': (T_M_values.mean(), T_M_values.std()),
            't_on_effective_range': (t_on_effective.min(), t_on_effective.max()),
            'scaling_factor_impact': np.corrcoef(T_M_values, np.abs(t_on_values))[0, 1]
        }

        return impact_analysis

    @beartype
    def _create_comprehensive_plots(
        self,
        param_samples: Dict[str, torch.Tensor],
        pattern_scores: Dict[str, torch.Tensor],
        phase_diagrams: Dict[str, Any],
        dependency_analysis: Dict[str, Any],
        hierarchical_impact: Dict[str, Any]
    ) -> None:
        """Create comprehensive visualization plots."""
        print("  Creating comprehensive plots...")

        # Plot 1: Clean prior distributions with HPDI ranges and mode annotations
        fig1 = self._plot_comprehensive_prior_distributions()
        fig1.savefig(self.save_path / "01_comprehensive_prior_distributions.png", dpi=300, bbox_inches='tight')
        fig1.savefig(self.save_path / "01_comprehensive_prior_distributions.pdf", bbox_inches='tight')
        plt.close(fig1)

        # Plot 2: Pattern time courses
        pattern_examples = self.generate_pattern_examples(n_examples=3)
        fig2 = self.plot_pattern_time_courses(pattern_examples)
        fig2.savefig(self.save_path / "02_pattern_time_courses.png", dpi=300, bbox_inches='tight')
        fig2.savefig(self.save_path / "02_pattern_time_courses.pdf", bbox_inches='tight')
        plt.close(fig2)

        # Plot 3: Phase diagrams
        fig3 = self._plot_phase_diagrams(phase_diagrams)
        fig3.savefig(self.save_path / "03_phase_diagrams.png", dpi=300, bbox_inches='tight')
        fig3.savefig(self.save_path / "03_phase_diagrams.pdf", bbox_inches='tight')
        plt.close(fig3)

        # Plot 4: Parameter correlation analysis
        fig4 = self._plot_correlation_analysis(dependency_analysis)
        fig4.savefig(self.save_path / "04_correlation_analysis.png", dpi=300, bbox_inches='tight')
        fig4.savefig(self.save_path / "04_correlation_analysis.pdf", bbox_inches='tight')
        plt.close(fig4)

        # Plot 5: Hierarchical impact analysis
        fig5 = self._plot_hierarchical_impact(param_samples, hierarchical_impact)
        fig5.savefig(self.save_path / "05_hierarchical_impact.png", dpi=300, bbox_inches='tight')
        fig5.savefig(self.save_path / "05_hierarchical_impact.pdf", bbox_inches='tight')
        plt.close(fig5)

        # Plot 6: Enhanced prior distributions with pattern overlays
        fig6 = self._plot_enhanced_prior_distributions(param_samples, pattern_scores)
        fig6.savefig(self.save_path / "06_enhanced_prior_distributions.png", dpi=300, bbox_inches='tight')
        fig6.savefig(self.save_path / "06_enhanced_prior_distributions.pdf", bbox_inches='tight')
        plt.close(fig6)

        # Plot 7: Pattern coverage analysis
        fig7 = self._plot_pattern_coverage_analysis(pattern_scores)
        fig7.savefig(self.save_path / "07_pattern_coverage.png", dpi=300, bbox_inches='tight')
        fig7.savefig(self.save_path / "07_pattern_coverage.pdf", bbox_inches='tight')
        plt.close(fig7)

        # Step 8: Save comprehensive analysis report
        print("  Computing constraint feasibility for summary...")
        feasibility_results = self.analyze_constraint_feasibility()
        recommendations = self.generate_optimization_recommendations(feasibility_results)

    @beartype
    def _plot_comprehensive_prior_distributions(self) -> plt.Figure:
        """Plot comprehensive prior distributions with HPDI ranges and mode annotations."""
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        axes = axes.flatten()

        # Parameters to plot (9 parameters for 3x3 grid)
        params_to_plot = ['R_on', 't_on_star', 'delta_star', 'gamma_star', 'T_M_star', 't_loc', 't_scale', 'U_0i', 'lambda_j']

        for idx, param_name in enumerate(params_to_plot):
            ax = axes[idx]

            # Get distribution type configuration
            distribution_types = {
                'R_on': 'lognormal',
                't_on_star': 'normal',
                'delta_star': 'lognormal',
                'gamma_star': 'lognormal',
                'T_M_star': 'gamma',
                't_loc': 'gamma',
                't_scale': 'gamma',
                'U_0i': 'lognormal',
                'lambda_j': 'lognormal'
            }

            if param_name not in distribution_types:
                continue

            distribution_type = distribution_types[param_name]

            # Calculate HPDI to determine appropriate x-range
            hpdi_info = self.calculate_hpdi_for_single_param(param_name)
            if not hpdi_info:
                continue

            # Determine x-range based on HPDI with appropriate buffer
            hpdi_lower = hpdi_info['lower']
            hpdi_upper = hpdi_info['upper']
            hpdi_width = hpdi_upper - hpdi_lower

            # Add buffer margins (50% of HPDI width on each side, minimum 20%)
            buffer_margin = max(hpdi_width * 0.5, hpdi_width * 0.2)

            # For lognormal distributions, ensure we don't go below zero
            if distribution_type in ['lognormal', 'gamma']:
                x_min = max(0.001, hpdi_lower - buffer_margin)
            else:  # normal distribution
                x_min = hpdi_lower - buffer_margin

            x_max = hpdi_upper + buffer_margin

            # Generate x values for PDF calculation with extended range
            x = np.linspace(x_min, x_max, 2000)

            # Get prior configuration
            if param_name in self.current_priors:
                prior = self.current_priors[param_name]

                # Calculate PDF and mode based on distribution type
                if distribution_type == 'lognormal':
                    pdf = stats.lognorm.pdf(x, s=prior['scale'], scale=np.exp(prior['loc']))
                    mode = np.exp(prior['loc'] - prior['scale']**2)
                elif distribution_type == 'normal':
                    pdf = stats.norm.pdf(x, loc=prior['loc'], scale=prior['scale'])
                    mode = prior['loc']
                elif distribution_type == 'gamma':
                    pdf = stats.gamma.pdf(x, a=prior['alpha'], scale=1/prior['beta'])
                    mode = max(0, (prior['alpha'] - 1) / prior['beta']) if prior['alpha'] > 1 else 0
                else:
                    continue

                # Plot main prior distribution with gray styling
                ax.fill_between(x, pdf, alpha=0.7, color='lightgray', edgecolor='none', label='Prior PDF')

                # Add HPDI bar and annotations
                hpdi_info = self.calculate_hpdi_for_single_param(param_name)
                if hpdi_info:
                    # Set y-axis limits to accommodate HPDI bars
                    max_density = max(pdf)
                    y_bar_height = max_density * 0.08
                    y_margin = max_density * 0.15
                    ax.set_ylim(-y_margin, max_density * 1.2)

                    # Position HPDI bar at the bottom
                    y_bar_pos = -y_margin * 0.4

                    # Draw HPDI bar
                    ax.barh(y_bar_pos, hpdi_info['width'], left=hpdi_info['lower'],
                           height=y_bar_height, color='darkblue', alpha=0.8,
                           label=f"95% HPDI")

                    # Add HPDI text annotations
                    ax.text(hpdi_info['lower'], y_bar_pos - y_bar_height * 0.7,
                           f"{hpdi_info['lower']:.2f}", ha='center', va='top', fontsize=8)
                    ax.text(hpdi_info['upper'], y_bar_pos - y_bar_height * 0.7,
                           f"{hpdi_info['upper']:.2f}", ha='center', va='top', fontsize=8)

                    # Add mode annotation
                    if x_min <= mode <= x_max:
                        mode_density = np.interp(mode, x, pdf)
                        ax.axvline(mode, color='red', linestyle='--', alpha=0.7, linewidth=1)
                        ax.text(mode, mode_density * 1.1, f'mode={mode:.2f}',
                               ha='center', va='bottom', fontsize=9,
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

            # Get proper parameter labels using metadata system
            x_label = get_parameter_label(
                param_name=param_name,
                label_type="display",
                model=self.model,
                fallback_to_legacy=True
            )
            title_label = get_parameter_label(
                param_name=param_name,
                label_type="short",
                model=self.model,
                fallback_to_legacy=True
            )

            ax.set_xlabel(x_label)
            ax.set_ylabel('Density')
            ax.set_title(title_label)
            ax.legend(fontsize=8, loc='upper right')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    @beartype
    def _plot_enhanced_prior_distributions(
        self,
        param_samples: Dict[str, torch.Tensor],
        pattern_scores: Dict[str, torch.Tensor]
    ) -> plt.Figure:
        """Plot enhanced prior distributions with pattern-specific overlays."""
        fig, axes = plt.subplots(3, 2, figsize=(15, 18))
        axes = axes.flatten()

        # Parameters to plot
        params_to_plot = ['R_on', 't_on_star', 'delta_star', 'gamma_star', 'T_M_star', 'U_0i']
        pattern_names = list(self.pattern_constraints.keys())
        colors = ['red', 'blue', 'green', 'orange', 'purple']

        for idx, param_name in enumerate(params_to_plot):
            ax = axes[idx]

            # Plot overall distribution
            values = param_samples[param_name].numpy()
            ax.hist(values, bins=50, alpha=0.3, color='gray', density=True, label='Overall')

            # Plot pattern-specific distributions
            assignments = pattern_scores['assignments'].numpy()
            for i, (pattern_name, color) in enumerate(zip(pattern_names, colors)):
                mask = assignments == i
                if mask.sum() > 10:
                    pattern_values = values[mask]
                    formatted_pattern = _format_pattern_name(pattern_name)
                    ax.hist(pattern_values, bins=30, alpha=0.6, color=color,
                           density=True, label=f'{formatted_pattern} ({mask.sum()})')

            # Get proper parameter labels using metadata system
            x_label = get_parameter_label(
                param_name=param_name,
                label_type="display",
                model=self.model,
                fallback_to_legacy=True
            )
            title_label = get_parameter_label(
                param_name=param_name,
                label_type="short",
                model=self.model,
                fallback_to_legacy=True
            )

            ax.set_xlabel(x_label)
            ax.set_ylabel('Density')
            ax.set_title(f'{title_label} Distribution by Pattern')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    @beartype
    def _plot_phase_diagrams(self, phase_diagrams: Dict[str, Any]) -> plt.Figure:
        """Plot 2D phase diagrams showing pattern boundaries."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()

        pattern_names = list(self.pattern_constraints.keys())
        colors = ['red', 'blue', 'green', 'orange', 'purple']

        for idx, (_, diagram_data) in enumerate(phase_diagrams.items()):
            if idx >= 4:
                break

            ax = axes[idx]

            x_vals = diagram_data['x_vals']
            y_vals = diagram_data['y_vals']
            assignments = diagram_data['assignments']

            # Create scatter plot colored by pattern
            for i, (pattern_name, color) in enumerate(zip(pattern_names, colors)):
                mask = assignments == i
                if mask.sum() > 0:
                    formatted_pattern = _format_pattern_name(pattern_name)
                    ax.scatter(x_vals[mask], y_vals[mask], c=color, alpha=0.6, s=1, label=formatted_pattern)

            # Get proper parameter labels using metadata system
            x_label = get_parameter_label(
                param_name=diagram_data['x_label'],
                label_type="display",
                model=self.model,
                fallback_to_legacy=True
            )
            y_label = get_parameter_label(
                param_name=diagram_data['y_label'],
                label_type="display",
                model=self.model,
                fallback_to_legacy=True
            )
            x_short = get_parameter_label(
                param_name=diagram_data['x_label'],
                label_type="short",
                model=self.model,
                fallback_to_legacy=True
            )
            y_short = get_parameter_label(
                param_name=diagram_data['y_label'],
                label_type="short",
                model=self.model,
                fallback_to_legacy=True
            )

            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_title(f'Phase Diagram: {x_short} vs {y_short}')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    @beartype
    def _plot_correlation_analysis(self, dependency_analysis: Dict[str, Any]) -> plt.Figure:
        """Plot parameter correlation analysis."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Overall correlation heatmap
        corr_matrix = dependency_analysis['overall_correlation']
        param_names = dependency_analysis['param_names']

        # Get proper parameter labels using metadata system
        param_labels = []
        for param_name in param_names:
            label = get_parameter_label(
                param_name=param_name,
                label_type="display",
                model=self.model,
                fallback_to_legacy=True
            )
            param_labels.append(label)

        im1 = axes[0].imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        axes[0].set_xticks(range(len(param_names)))
        axes[0].set_yticks(range(len(param_names)))
        axes[0].set_xticklabels(param_labels, rotation=45)
        axes[0].set_yticklabels(param_labels)
        axes[0].set_title('Overall Parameter Correlations')

        # Add correlation values to heatmap
        for i in range(len(param_names)):
            for j in range(len(param_names)):
                axes[0].text(j, i, f'{corr_matrix[i, j]:.2f}',
                           ha='center', va='center', fontsize=8)

        plt.colorbar(im1, ax=axes[0])

        # Pattern-specific correlation differences
        pattern_correlations = dependency_analysis['pattern_correlations']
        if pattern_correlations:
            # Show difference from overall correlation for first available pattern
            first_pattern = list(pattern_correlations.keys())[0]
            pattern_corr = pattern_correlations[first_pattern]
            diff_matrix = pattern_corr - corr_matrix

            im2 = axes[1].imshow(diff_matrix, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
            axes[1].set_xticks(range(len(param_names)))
            axes[1].set_yticks(range(len(param_names)))
            axes[1].set_xticklabels(param_labels, rotation=45)
            axes[1].set_yticklabels(param_labels)
            axes[1].set_title(f'{first_pattern} Pattern - Overall Correlation Difference')

            plt.colorbar(im2, ax=axes[1])
        else:
            axes[1].text(0.5, 0.5, 'Insufficient pattern-specific data',
                        ha='center', va='center', transform=axes[1].transAxes)
            axes[1].set_title('Pattern-Specific Analysis')

        plt.tight_layout()
        return fig

    @beartype
    def _plot_hierarchical_impact(
        self,
        param_samples: Dict[str, torch.Tensor],
        hierarchical_impact: Dict[str, Any]
    ) -> plt.Figure:
        """Plot hierarchical parameter impact analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        T_M_values = param_samples['T_M_star'].numpy()
        t_on_values = param_samples['t_on_star'].numpy()

        # Get proper parameter labels using metadata system
        T_M_label = get_parameter_label(
            param_name='T_M_star',
            label_type="display",
            model=self.model,
            fallback_to_legacy=True
        )
        t_on_label = get_parameter_label(
            param_name='t_on_star',
            label_type="display",
            model=self.model,
            fallback_to_legacy=True
        )
        T_M_short = get_parameter_label(
            param_name='T_M_star',
            label_type="short",
            model=self.model,
            fallback_to_legacy=True
        )
        t_on_short = get_parameter_label(
            param_name='t_on_star',
            label_type="short",
            model=self.model,
            fallback_to_legacy=True
        )

        # Plot 1: T_M_star distribution
        axes[0, 0].hist(T_M_values, bins=50, alpha=0.7, color='blue')
        axes[0, 0].set_xlabel(T_M_label)
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title(f'{T_M_short} Distribution')
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: t_on_star vs T_M_star
        axes[0, 1].scatter(T_M_values, t_on_values, alpha=0.5, s=1)
        axes[0, 1].set_xlabel(T_M_label)
        axes[0, 1].set_ylabel(t_on_label)
        axes[0, 1].set_title(f'{t_on_short} vs {T_M_short}')
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Effective onset times
        t_on_effective = t_on_values * T_M_values
        axes[1, 0].hist(t_on_effective, bins=50, alpha=0.7, color='green')
        axes[1, 0].set_xlabel(f'Effective Onset Time ({t_on_short} × {T_M_short})')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Effective Onset Time Distribution')
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Impact summary text
        axes[1, 1].axis('off')
        impact_text = f"""
Hierarchical Impact Analysis

T_M_star Range: {hierarchical_impact['T_M_range'][0]:.2f} - {hierarchical_impact['T_M_range'][1]:.2f}
T_M_star Mean ± Std: {hierarchical_impact['T_M_mean_std'][0]:.2f} ± {hierarchical_impact['T_M_mean_std'][1]:.2f}

Effective Onset Range: {hierarchical_impact['t_on_effective_range'][0]:.2f} - {hierarchical_impact['t_on_effective_range'][1]:.2f}

Scaling Factor Correlation: {hierarchical_impact['scaling_factor_impact']:.3f}
        """
        axes[1, 1].text(0.05, 0.95, impact_text.strip(), transform=axes[1, 1].transAxes,
                        fontsize=10, verticalalignment='top', fontfamily='monospace')

        plt.tight_layout()
        return fig

    @beartype
    def _plot_pattern_coverage_analysis(self, pattern_scores: Dict[str, torch.Tensor]) -> plt.Figure:
        """Plot pattern coverage analysis."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        coverage = pattern_scores['coverage']
        patterns = list(coverage.keys())
        coverages = list(coverage.values())

        # Format pattern names for LaTeX compatibility
        formatted_patterns = [_format_pattern_name(p) for p in patterns]

        # Bar plot of coverage
        colors = ['red', 'blue', 'green', 'orange', 'purple'][:len(patterns)]
        bars = axes[0].bar(formatted_patterns, coverages, color=colors, alpha=0.7)
        axes[0].set_ylabel('Coverage Fraction')
        axes[0].set_title('Pattern Coverage Distribution')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(True, alpha=0.3)

        # Add coverage percentages on bars
        for bar, coverage_val in zip(bars, coverages):
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{coverage_val:.1%}', ha='center', va='bottom')

        # Pie chart of coverage
        axes[1].pie(coverages, labels=formatted_patterns, colors=colors, autopct='%1.1f%%', startangle=90)
        axes[1].set_title('Pattern Coverage Distribution')

        plt.tight_layout()
        return fig

    @beartype
    def _generate_comprehensive_recommendations(
        self,
        param_samples: Dict[str, torch.Tensor],
        pattern_scores: Dict[str, torch.Tensor],
        dependency_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate comprehensive optimization recommendations."""
        coverage = pattern_scores['coverage']

        # Analyze coverage uniformity
        coverage_values = list(coverage.values())
        coverage_std = np.std(coverage_values)
        coverage_min = min(coverage_values)
        coverage_max = max(coverage_values)

        # Note: correlation analysis available in dependency_analysis if needed

        recommendations = {
            'coverage_analysis': {
                'mean_coverage': np.mean(coverage_values),
                'coverage_std': coverage_std,
                'coverage_range': (coverage_min, coverage_max),
                'uniformity_score': 1.0 - coverage_std  # Higher is more uniform
            },
            'critical_parameters': self._identify_critical_parameters(param_samples, pattern_scores),
            'hyperparameter_adjustments': self._suggest_hyperparameter_adjustments(coverage, param_samples),
            'methodology_improvements': self._suggest_methodology_improvements(coverage, dependency_analysis)
        }

        return recommendations

    @beartype
    def _identify_critical_parameters(
        self,
        param_samples: Dict[str, torch.Tensor],
        pattern_scores: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        """Identify which parameters most strongly influence pattern classification."""
        # Compute parameter importance for pattern separation
        assignments = pattern_scores['assignments'].numpy()

        importance_scores = {}
        for param_name in ['R_on', 't_on_star', 'delta_star', 'gamma_star']:
            param_values = param_samples[param_name].numpy()

            # Compute between-pattern variance vs within-pattern variance
            between_var = 0.0
            within_var = 0.0
            total_mean = param_values.mean()

            for pattern_idx in range(len(self.pattern_constraints)):
                mask = assignments == pattern_idx
                if mask.sum() > 1:
                    pattern_values = param_values[mask]
                    pattern_mean = pattern_values.mean()
                    pattern_var = pattern_values.var()

                    between_var += mask.sum() * (pattern_mean - total_mean) ** 2
                    within_var += (mask.sum() - 1) * pattern_var

            # F-statistic-like measure
            if within_var > 0:
                importance_scores[param_name] = between_var / within_var
            else:
                importance_scores[param_name] = 0.0

        return importance_scores

    @beartype
    def _suggest_hyperparameter_adjustments(
        self,
        coverage: Dict[str, float],
        param_samples: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        """Suggest specific hyperparameter adjustments based on coverage analysis."""
        adjustments = {}

        # Find under-represented patterns
        mean_coverage = np.mean(list(coverage.values()))
        under_represented = {k: v for k, v in coverage.items() if v < mean_coverage * 0.8}

        if 'sustained' in under_represented:
            # Sustained pattern needs longer durations
            current_delta_mean = param_samples['delta_star'].mean().item()
            adjustments['delta_star'] = {
                'current_loc': self.current_priors['delta_star']['loc'],
                'suggested_loc': self.current_priors['delta_star']['loc'] + 0.2,  # Increase mean
                'current_scale': self.current_priors['delta_star']['scale'],
                'suggested_scale': self.current_priors['delta_star']['scale'] + 0.1,  # Increase spread
                'rationale': f'Increase duration to support sustained patterns (current mean: {current_delta_mean:.3f})'
            }

        if 'decay_only' in under_represented:
            # Decay-only pattern needs later onset times
            adjustments['t_on_star'] = {
                'current_loc': self.current_priors['t_on_star']['loc'],
                'suggested_loc': self.current_priors['t_on_star']['loc'] + 0.3,  # Shift toward later times
                'current_scale': self.current_priors['t_on_star']['scale'],
                'suggested_scale': self.current_priors['t_on_star']['scale'] + 0.2,  # Increase spread
                'rationale': 'Shift onset times later to support decay-only patterns'
            }

        return adjustments

    @beartype
    def _suggest_methodology_improvements(
        self,
        coverage: Dict[str, float],
        dependency_analysis: Dict[str, Any]
    ) -> Dict[str, str]:
        """Suggest improvements to the analysis methodology."""
        improvements = {}

        coverage_uniformity = 1.0 - np.std(list(coverage.values()))

        if coverage_uniformity < 0.7:
            improvements['pattern_classification'] = (
                "Consider using soft pattern scoring with adjustable steepness parameters "
                "instead of hard constraints to improve pattern coverage uniformity"
            )

        # Check for strong parameter correlations
        corr_matrix = dependency_analysis['overall_correlation']
        max_corr = np.max(np.abs(corr_matrix - np.eye(len(corr_matrix))))

        if max_corr > 0.7:
            improvements['parameter_independence'] = (
                f"Strong parameter correlations detected (max: {max_corr:.3f}). "
                "Consider reparameterization or conditional priors to reduce dependencies"
            )

        return improvements

    @beartype
    def _save_comprehensive_analysis_report(
        self,
        recommendations: Dict[str, Any],
        pattern_scores: Dict[str, torch.Tensor]
    ) -> None:
        """Save comprehensive analysis report."""
        report_path = self.save_path / "comprehensive_calibration_report.txt"

        with open(report_path, 'w') as f:
            f.write("PyroVelocity Comprehensive Prior Hyperparameter Analysis\n")
            f.write("=" * 70 + "\n\n")

            # Coverage analysis
            f.write("PATTERN COVERAGE ANALYSIS:\n")
            f.write("-" * 30 + "\n")
            coverage = pattern_scores['coverage']
            for pattern, cov in coverage.items():
                f.write(f"{pattern}: {cov:.3f} ({cov:.1%})\n")

            coverage_analysis = recommendations['coverage_analysis']
            f.write(f"\nCoverage Statistics:\n")
            f.write(f"  Mean: {coverage_analysis['mean_coverage']:.3f}\n")
            f.write(f"  Std Dev: {coverage_analysis['coverage_std']:.3f}\n")
            f.write(f"  Range: {coverage_analysis['coverage_range'][0]:.3f} - {coverage_analysis['coverage_range'][1]:.3f}\n")
            f.write(f"  Uniformity Score: {coverage_analysis['uniformity_score']:.3f}\n\n")

            # Critical parameters
            f.write("CRITICAL PARAMETER ANALYSIS:\n")
            f.write("-" * 32 + "\n")
            critical_params = recommendations['critical_parameters']
            for param, importance in critical_params.items():
                f.write(f"{param}: {importance:.3f}\n")
            f.write("\n")

            # Hyperparameter adjustments
            f.write("SUGGESTED HYPERPARAMETER ADJUSTMENTS:\n")
            f.write("-" * 40 + "\n")
            adjustments = recommendations['hyperparameter_adjustments']
            for param, adjustment in adjustments.items():
                f.write(f"\n{param}:\n")
                f.write(f"  Current: loc={adjustment['current_loc']:.3f}, scale={adjustment['current_scale']:.3f}\n")
                f.write(f"  Suggested: loc={adjustment['suggested_loc']:.3f}, scale={adjustment['suggested_scale']:.3f}\n")
                f.write(f"  Rationale: {adjustment['rationale']}\n")

            # Methodology improvements
            f.write("\nMETHODOLOGY IMPROVEMENTS:\n")
            f.write("-" * 25 + "\n")
            improvements = recommendations['methodology_improvements']
            for improvement_type, suggestion in improvements.items():
                f.write(f"{improvement_type}:\n  {suggestion}\n\n")

        print(f"Comprehensive report saved to: {report_path}")










def main():
    """Main calibration workflow."""
    print("PyroVelocity Prior Hyperparameter Calibration")
    print("=" * 60)
    print("Analyzing dimensionless parameterization for balanced gene expression pattern coverage")
    print()

    # Initialize calibrator
    calibrator = PriorHyperparameterCalibrator()

    print(f"Analysis outputs will be saved to: {calibrator.save_path}")
    print()

    print("Running comprehensive parameter space analysis...")
    calibrator.run_comprehensive_parameter_space_analysis()


if __name__ == "__main__":
    main()
