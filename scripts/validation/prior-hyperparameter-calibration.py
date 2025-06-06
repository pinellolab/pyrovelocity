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

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Disable LaTeX rendering to avoid Unicode issues
plt.rcParams['text.usetex'] = False

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
        
        # Current dimensionless prior hyperparameters (from priors.py)
        self.current_priors = {
            'R_on': {'loc': 0.916, 'scale': 0.4},        # log(2.5), fold-change (LogNormal)
            't_on_star': {'loc': 0.2, 'scale': 0.6},     # Normal(0.2, 0.6²), allows negatives
            'delta_star': {'loc': -1.0, 'scale': 0.35},  # log(0.37) (LogNormal)
            'gamma_star': {'loc': 0.0, 'scale': 0.5},    # log(1.0) (LogNormal)
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

            print(f"\n{pattern_name.upper()} PATTERN:")

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
    def plot_prior_distributions(self) -> plt.Figure:
        """
        Plot current prior distributions with pattern constraint overlays.

        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        # Define parameter plotting details
        param_details = {
            'R_on': {
                'title': 'Fold-change (R_on)',
                'x_range': (0.5, 10),
                'distribution': 'lognormal',
                'latex_label': r'$R_{\mathrm{on},i}$'
            },
            't_on_star': {
                'title': 'Onset Time (t*_on)',
                'x_range': (-1.5, 1.5),
                'distribution': 'normal',
                'latex_label': r'$t^*_{0,\mathrm{on}i}$'
            },
            'delta_star': {
                'title': 'Duration (delta*)',
                'x_range': (0.1, 2.0),
                'distribution': 'lognormal',
                'latex_label': r'$\delta^*_i$'
            },
            'gamma_star': {
                'title': 'Relative Degradation (gamma*)',
                'x_range': (0.2, 5.0),
                'distribution': 'lognormal',
                'latex_label': r'$\gamma^*_i$'
            }
        }

        for idx, (param_name, details) in enumerate(param_details.items()):
            ax = axes[idx]
            prior = self.current_priors[param_name]
            x_range = details['x_range']

            # Generate x values
            x = np.linspace(x_range[0], x_range[1], 1000)

            # Calculate PDF
            if details['distribution'] == 'lognormal':
                pdf = stats.lognorm.pdf(x, s=prior['scale'], scale=np.exp(prior['loc']))
            else:  # normal
                pdf = stats.norm.pdf(x, loc=prior['loc'], scale=prior['scale'])

            # Plot distribution
            ax.plot(x, pdf, 'b-', linewidth=2, label='Prior PDF')
            ax.fill_between(x, pdf, alpha=0.3)

            # Add constraint lines for relevant patterns
            constraint_colors = ['red', 'orange', 'green', 'purple', 'brown']
            color_idx = 0

            for pattern_name, constraints in self.pattern_constraints.items():
                for constraint_name, (operator, threshold) in constraints.items():
                    # Handle special constraint names
                    base_param = constraint_name
                    if constraint_name.endswith('_upper'):
                        base_param = constraint_name.replace('_upper', '')
                    elif constraint_name.endswith('_beyond'):
                        base_param = constraint_name.replace('_beyond', '')

                    if base_param == param_name and x_range[0] <= threshold <= x_range[1]:
                        color = constraint_colors[color_idx % len(constraint_colors)]
                        linestyle = '--' if operator == '<' else '-'
                        ax.axvline(threshold, color=color, linestyle=linestyle, alpha=0.7,
                                 label=f'{pattern_name}: {operator}{threshold}')
                        color_idx += 1

            ax.set_xlabel(details['latex_label'])
            ax.set_ylabel('Density')
            ax.set_title(details['title'])
            ax.grid(True, alpha=0.3)

            # Add legend if there are constraint lines
            if color_idx > 0:
                ax.legend(fontsize=8)

        plt.tight_layout()
        return fig

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
            print(f"\nGenerating {n_examples} examples for {pattern_name.upper()} pattern...")

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

            # Format axes
            axes[pattern_idx, 0].set_xlabel('Dimensionless Time (t*)')
            axes[pattern_idx, 0].set_ylabel('Unspliced (u*)')
            axes[pattern_idx, 0].set_title(f'{pattern_name.title()}: Unspliced')
            axes[pattern_idx, 0].grid(True, alpha=0.3)
            axes[pattern_idx, 0].legend()

            axes[pattern_idx, 1].set_xlabel('Dimensionless Time (t*)')
            axes[pattern_idx, 1].set_ylabel('Spliced (s*)')
            axes[pattern_idx, 1].set_title(f'{pattern_name.title()}: Spliced')
            axes[pattern_idx, 1].grid(True, alpha=0.3)

            axes[pattern_idx, 2].set_xlabel('Unspliced (u*)')
            axes[pattern_idx, 2].set_ylabel('Spliced (s*)')
            axes[pattern_idx, 2].set_title(f'{pattern_name.title()}: Phase Portrait')
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
    def run_full_analysis(self) -> None:
        """Run the complete calibration analysis and save all outputs."""
        print("Starting comprehensive prior hyperparameter calibration analysis...")

        # Step 1: Analyze constraint feasibility
        feasibility_results = self.analyze_constraint_feasibility()

        # Step 2: Plot prior distributions
        print("\nGenerating prior distribution plots...")
        prior_fig = self.plot_prior_distributions()
        prior_fig.savefig(self.save_path / "01_prior_distributions.png", dpi=300, bbox_inches='tight')
        prior_fig.savefig(self.save_path / "01_prior_distributions.pdf", bbox_inches='tight')
        plt.close(prior_fig)

        # Step 3: Generate pattern examples
        pattern_examples = self.generate_pattern_examples(n_examples=3)

        # Step 4: Plot pattern time courses
        print("\nGenerating pattern time course plots...")
        time_course_fig = self.plot_pattern_time_courses(pattern_examples)
        time_course_fig.savefig(self.save_path / "02_pattern_time_courses.png", dpi=300, bbox_inches='tight')
        time_course_fig.savefig(self.save_path / "02_pattern_time_courses.pdf", bbox_inches='tight')
        plt.close(time_course_fig)

        # Step 5: Generate optimization recommendations
        print("\nGenerating optimization recommendations...")
        recommendations = self.generate_optimization_recommendations(feasibility_results)

        # Step 6: Create summary plot
        summary_fig = self._create_summary_plot(feasibility_results, recommendations)
        summary_fig.savefig(self.save_path / "03_calibration_summary.png", dpi=300, bbox_inches='tight')
        summary_fig.savefig(self.save_path / "03_calibration_summary.pdf", bbox_inches='tight')
        plt.close(summary_fig)

        # Step 7: Save recommendations to text file
        self._save_recommendations_report(recommendations, feasibility_results)

        # Step 8: Combine PDFs
        print("\nCombining PDF outputs...")
        try:
            combine_pdfs(
                pdf_directory=str(self.save_path),
                output_filename="combined_prior_calibration_analysis.pdf",
                exclude_patterns=["combined_*.pdf"]
            )
        except Exception as e:
            print(f"Warning: Could not combine PDFs: {e}")

        print(f"\n✅ Analysis complete! Results saved to: {self.save_path}")
        print("\nNext steps:")
        print("1. Review the combined PDF report")
        print("2. Consider implementing recommended hyperparameter adjustments")
        print("3. Re-run prior-predictive-check.py to validate improvements")

    @beartype
    def _create_summary_plot(
        self,
        feasibility_results: Dict[str, Dict[str, float]],
        recommendations: Dict[str, Any]
    ) -> plt.Figure:
        """Create a summary plot of the calibration analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Plot 1: Pattern feasibility bar chart
        patterns = list(feasibility_results.keys())
        joint_probs = [feasibility_results[p]['joint_probability'] for p in patterns]

        colors = ['red' if p < 0.05 else 'orange' if p < 0.2 else 'green' for p in joint_probs]

        axes[0, 0].bar(patterns, joint_probs, color=colors, alpha=0.7)
        axes[0, 0].axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='Critical threshold')
        axes[0, 0].axhline(y=0.2, color='orange', linestyle='--', alpha=0.7, label='Acceptable threshold')
        axes[0, 0].set_ylabel('Joint Probability')
        axes[0, 0].set_title('Pattern Constraint Feasibility')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Coverage summary pie chart
        summary = recommendations['summary']
        sizes = [len(summary['critical_patterns']), len(summary['marginal_patterns']), len(summary['acceptable_patterns'])]
        labels = ['Critical (<5%)', 'Marginal (5-20%)', 'Acceptable (>20%)']
        colors_pie = ['red', 'orange', 'green']

        axes[0, 1].pie(sizes, labels=labels, colors=colors_pie, autopct='%1.0f%%', startangle=90)
        axes[0, 1].set_title('Pattern Coverage Distribution')

        # Plot 3: Parameter constraint violations
        param_violations = {}
        for pattern, results in feasibility_results.items():
            for constraint, prob in results.items():
                if constraint != 'joint_probability' and prob < 0.2:
                    param_base = constraint.replace('_upper', '').replace('_beyond', '')
                    if param_base not in param_violations:
                        param_violations[param_base] = 0
                    param_violations[param_base] += 1

        if param_violations:
            params = list(param_violations.keys())
            violations = list(param_violations.values())
            axes[1, 0].bar(params, violations, color='red', alpha=0.7)
            axes[1, 0].set_ylabel('Number of Violations')
            axes[1, 0].set_title('Parameter Constraint Violations')
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'No violations detected', ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Parameter Constraint Violations')

        # Plot 4: Recommendations text summary
        axes[1, 1].axis('off')
        rec_action = recommendations['recommended_action']

        summary_text = f"""
CALIBRATION SUMMARY

Overall Coverage: {summary['overall_coverage']:.1%}

Critical Patterns: {len(summary['critical_patterns'])}
{', '.join(summary['critical_patterns']) if summary['critical_patterns'] else 'None'}

Marginal Patterns: {len(summary['marginal_patterns'])}
{', '.join(summary['marginal_patterns']) if summary['marginal_patterns'] else 'None'}

RECOMMENDED ACTION:
{rec_action['action'].replace('_', ' ').title()}

RATIONALE:
{rec_action['rationale']}
        """

        axes[1, 1].text(0.05, 0.95, summary_text.strip(), transform=axes[1, 1].transAxes,
                        fontsize=10, verticalalignment='top', fontfamily='monospace')

        plt.tight_layout()
        return fig

    @beartype
    def _save_recommendations_report(
        self,
        recommendations: Dict[str, Any],
        feasibility_results: Dict[str, Dict[str, float]]
    ) -> None:
        """Save detailed recommendations to a text file."""
        report_path = self.save_path / "calibration_recommendations.txt"

        with open(report_path, 'w') as f:
            f.write("PyroVelocity Prior Hyperparameter Calibration Report\n")
            f.write("=" * 60 + "\n\n")

            # Current prior settings
            f.write("CURRENT PRIOR SETTINGS:\n")
            f.write("-" * 25 + "\n")
            for param, settings in self.current_priors.items():
                f.write(f"{param}: loc={settings['loc']:.3f}, scale={settings['scale']:.3f}\n")
            f.write("\n")

            # Feasibility results
            f.write("CONSTRAINT FEASIBILITY ANALYSIS:\n")
            f.write("-" * 35 + "\n")
            for pattern, results in feasibility_results.items():
                f.write(f"\n{pattern.upper()} PATTERN:\n")
                for constraint, prob in results.items():
                    if constraint != 'joint_probability':
                        status = "CRITICAL" if prob < 0.05 else "MARGINAL" if prob < 0.2 else "ACCEPTABLE"
                        f.write(f"  {constraint}: {prob:.3f} ({status})\n")
                f.write(f"  Joint Probability: {results['joint_probability']:.4f}\n")
            f.write("\n")

            # Recommendations
            f.write("OPTIMIZATION RECOMMENDATIONS:\n")
            f.write("-" * 30 + "\n")

            summary = recommendations['summary']
            f.write(f"Overall Coverage: {summary['overall_coverage']:.1%}\n")
            f.write(f"Critical Patterns: {summary['critical_patterns']}\n")
            f.write(f"Marginal Patterns: {summary['marginal_patterns']}\n")
            f.write(f"Acceptable Patterns: {summary['acceptable_patterns']}\n\n")

            rec_action = recommendations['recommended_action']
            f.write(f"RECOMMENDED ACTION: {rec_action['action']}\n")
            f.write(f"RATIONALE: {rec_action['rationale']}\n\n")

            # Detailed options
            f.write("DETAILED OPTIMIZATION OPTIONS:\n")
            f.write("-" * 32 + "\n")
            for option_name, option in recommendations['optimization_options'].items():
                f.write(f"\n{option_name.upper().replace('_', ' ')}:\n")
                f.write(f"Description: {option['description']}\n")
                f.write(f"Pros: {', '.join(option['pros'])}\n")
                f.write(f"Cons: {', '.join(option['cons'])}\n")
                if 'changes' in option:
                    f.write(f"Changes: {option['changes']}\n")

        print(f"Detailed report saved to: {report_path}")


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

    # Run complete analysis
    calibrator.run_full_analysis()


if __name__ == "__main__":
    main()
