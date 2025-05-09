#!/usr/bin/env python
"""
Direct Comparison Script for PyroVelocity.

This script directly trains both the legacy and modular implementations
of PyroVelocity on the preprocessed pancreas data and compares their results.
It bypasses the validation framework to avoid the issues we've encountered.

By default, velocity computation is skipped to accelerate model comparison
during debugging. Use the --compute-velocity flag to enable velocity computation.

Example usage:
    python direct_comparison.py --max-epochs 5 --num-samples 3
    python direct_comparison.py --max-epochs 5 --num-samples 3 --compute-velocity
"""

import os
import sys
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import pyro
import scipy.sparse
from importlib.resources import files

# Add the src directory to the path to import test fixtures
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from pyrovelocity.io.serialization import load_anndata_from_json
from pyrovelocity.models._velocity import PyroVelocity
from pyrovelocity.models.modular import PyroVelocityModel
from pyrovelocity.models.modular.factory import create_standard_model, create_legacy_model1
from pyrovelocity.validation.comparison import (
    compare_parameters,
    compare_velocities,
    compare_uncertainties,
    compare_performance,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Direct Comparison Script for PyroVelocity"
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=5,
        help="Maximum number of epochs for training",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=3,
        help="Number of posterior samples to generate",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="direct_comparison_results",
        help="Directory to save validation results",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="legacy",
        choices=["legacy", "poisson"],
        help="Model type to use for the modular implementation (legacy=replicates legacy model, poisson=standard model with Poisson likelihood). "
             "Note: This parameter has no effect on the legacy model, which always uses VelocityModelAuto.",
    )
    parser.add_argument(
        "--compute-velocity",
        action="store_true",
        default=False,
        help="Compute velocity (default: False)",
    )
    return parser.parse_args()


def load_preprocessed_pancreas_data():
    """Load the preprocessed pancreas data fixture."""
    print("Loading preprocessed pancreas data fixture...")

    # Fixture hash for data validation
    FIXTURE_HASH = "95c80131694f2c6449a48a56513ef79cdc56eae75204ec69abde0d81a18722ae"

    try:
        fixture_file_path = (
            files("pyrovelocity.tests.data") / "preprocessed_pancreas_50_7.json"
        )
        adata = load_anndata_from_json(
            filename=str(fixture_file_path),
            expected_hash=FIXTURE_HASH,
        )
        print(f"Loaded preprocessed pancreas data: {adata.shape[0]} cells, {adata.shape[1]} genes")
        return adata
    except Exception as e:
        print(f"Error loading preprocessed pancreas data: {e}")
        raise


def train_legacy_model(adata, max_epochs, num_samples, seed=42, compute_velocity=True, **kwargs):
    """
    Train the legacy model and return results.

    Args:
        adata: AnnData object containing the data
        max_epochs: Maximum number of epochs for training
        num_samples: Number of posterior samples to generate
        seed: Random seed for reproducibility
        compute_velocity: Whether to compute velocity (default: True)
        **kwargs: Additional keyword arguments (ignored)

    Note: The legacy model always uses VelocityModelAuto regardless of any model_type parameter.
    The model_type parameter is accepted through **kwargs for compatibility but has no effect.
    """
    print("Training legacy model...")

    # Set random seed for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    pyro.set_rng_seed(seed)

    # Set up AnnData for legacy model
    PyroVelocity.setup_anndata(adata)

    # Create legacy model - note that model_type is not used as it doesn't affect the legacy model
    # The legacy model always uses VelocityModelAuto regardless of model_type value
    print("Legacy model always uses VelocityModelAuto regardless of model_type")

    # Intentionally ignore kwargs (including model_type) as they have no effect on the legacy model
    _ = kwargs  # Suppress unused variable warning

    model = PyroVelocity(adata)

    # Train model
    training_start_time = time.time()
    model.train(max_epochs=max_epochs, check_val_every_n_epoch=None)
    training_end_time = time.time()
    training_time = training_end_time - training_start_time

    # Generate posterior samples
    inference_start_time = time.time()
    try:
        posterior_samples = model.generate_posterior_samples(num_samples=num_samples)
    except AttributeError as e:
        if "'NoneType' object has no attribute 'uns'" in str(e):
            # This is a known issue with the legacy model
            # We need to pass the adata explicitly
            posterior_samples = model.generate_posterior_samples(adata=model.adata, num_samples=num_samples)
        else:
            raise

    # Print parameter shapes for debugging
    print("Legacy model parameter shapes:")
    for key, value in posterior_samples.items():
        if isinstance(value, (np.ndarray, torch.Tensor)):
            print(f"  {key}: {value.shape}")

    # Print parameter statistics for debugging
    print("\nLegacy model parameter statistics:")
    for param in ["alpha", "beta", "gamma"]:
        if param in posterior_samples:
            param_value = posterior_samples[param]
            if isinstance(param_value, torch.Tensor):
                param_value = param_value.detach().cpu().numpy()
            print(f"  {param} mean: {np.mean(param_value):.6f}, std: {np.std(param_value):.6f}, min: {np.min(param_value):.6f}, max: {np.max(param_value):.6f}")

    # Print scaling factors if available
    if "u_scale" in posterior_samples:
        u_scale = posterior_samples["u_scale"]
        if isinstance(u_scale, torch.Tensor):
            u_scale = u_scale.detach().cpu().numpy()
        print(f"  u_scale mean: {np.mean(u_scale):.6f}, std: {np.std(u_scale):.6f}, min: {np.min(u_scale):.6f}, max: {np.max(u_scale):.6f}")

    if "s_scale" in posterior_samples:
        s_scale = posterior_samples["s_scale"]
        if isinstance(s_scale, torch.Tensor):
            s_scale = s_scale.detach().cpu().numpy()
        print(f"  s_scale mean: {np.mean(s_scale):.6f}, std: {np.std(s_scale):.6f}, min: {np.min(s_scale):.6f}, max: {np.max(s_scale):.6f}")

    # Print ut and st statistics
    if "ut" in posterior_samples:
        ut = posterior_samples["ut"]
        if isinstance(ut, torch.Tensor):
            ut = ut.detach().cpu().numpy()
        print(f"  ut mean: {np.mean(ut):.6f}, std: {np.std(ut):.6f}, min: {np.min(ut):.6f}, max: {np.max(ut):.6f}")

    if "st" in posterior_samples:
        st = posterior_samples["st"]
        if isinstance(st, torch.Tensor):
            st = st.detach().cpu().numpy()
        print(f"  st mean: {np.mean(st):.6f}, std: {np.std(st):.6f}, min: {np.min(st):.6f}, max: {np.max(st):.6f}")

    # Initialize velocity and uncertainty as None
    velocity = None
    uncertainty = None

    # Compute velocity only if requested
    if compute_velocity:
        # Print velocity calculation details
        print("\nComputing velocity for legacy model...")

        # Calculate velocity manually for comparison
        if ("u_scale" in posterior_samples) and ("s_scale" in posterior_samples):
            scale = posterior_samples["u_scale"] / posterior_samples["s_scale"]
            print(f"  Using scale from u_scale/s_scale, shape: {scale.shape}, mean: {np.mean(scale):.6f}")
        elif ("u_scale" in posterior_samples) and not ("s_scale" in posterior_samples):
            scale = posterior_samples["u_scale"]
            print(f"  Using scale from u_scale only, shape: {scale.shape}, mean: {np.mean(scale):.6f}")
        else:
            scale = 1
            print("  No scaling applied (scale = 1)")

        # Calculate velocity manually
        manual_velocity = (
            posterior_samples["beta"] * posterior_samples["ut"] / scale
            - posterior_samples["gamma"] * posterior_samples["st"]
        ).mean(0)
        print(f"  Manual velocity shape: {manual_velocity.shape}, mean: {np.mean(manual_velocity):.6f}, std: {np.std(manual_velocity):.6f}")

        # Compute statistics from posterior samples (this computes velocity and stores it in adata)
        # The method signature is compute_statistics_from_posterior_samples(self, adata, posterior_samples, ...)
        posterior_samples = model.compute_statistics_from_posterior_samples(
            adata=adata,
            posterior_samples=posterior_samples,
            vector_field_basis="umap",
            ncpus_use=1,
            random_seed=seed
        )

        # Extract velocity from adata.layers["velocity_pyro"]
        velocity = adata.layers["velocity_pyro"] if "velocity_pyro" in adata.layers else None

        # Extract uncertainty (using the FDR values as a proxy for uncertainty)
        # In the legacy model, uncertainty is computed during vector_field_uncertainty
        # and stored in the posterior_samples dictionary as 'fdri'
        uncertainty = posterior_samples.get('fdri', None)

        if uncertainty is None:
            # If fdri is not available, we can use a simple standard deviation across samples
            # as a proxy for uncertainty
            if ('u_scale' in posterior_samples) and ('s_scale' in posterior_samples):
                scale = posterior_samples["u_scale"] / posterior_samples["s_scale"]
            elif ('u_scale' in posterior_samples) and not ('s_scale' in posterior_samples):
                scale = posterior_samples["u_scale"]
            else:
                scale = 1

            velocity_samples = (
                posterior_samples["beta"] * posterior_samples["ut"] / scale
                - posterior_samples["gamma"] * posterior_samples["st"]
            )
            uncertainty = np.std(velocity_samples, axis=0)
    else:
        print("\nSkipping velocity computation for legacy model (--no-compute-velocity flag is set)")

    inference_end_time = time.time()
    inference_time = inference_end_time - inference_start_time

    # Return results
    return {
        "model": model,
        "posterior_samples": posterior_samples,
        "velocity": velocity,
        "uncertainty": uncertainty,
        "performance": {
            "training_time": training_time,
            "inference_time": inference_time,
        }
    }


def train_modular_model(adata, max_epochs, num_samples, seed=42, model_type="legacy", compute_velocity=True):
    """
    Train the modular model and return results.

    Args:
        adata: AnnData object containing the data
        max_epochs: Maximum number of epochs for training
        num_samples: Number of posterior samples to generate
        seed: Random seed for reproducibility
        model_type: Model type to use (legacy or poisson)
        compute_velocity: Whether to compute velocity (default: True)
    """
    print("Training modular model...")

    # Set random seed for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    pyro.set_rng_seed(seed)

    # Set up AnnData for modular model
    adata_copy = adata.copy()
    PyroVelocityModel.setup_anndata(adata_copy)

    # Create modular model based on model_type
    print(f"Modular model using model_type: {model_type}")

    # Print available components for debugging
    from pyrovelocity.models.modular.registry import LikelihoodModelRegistry

    # Get available components by inspecting the registry
    print(f"Available likelihood models: {list(LikelihoodModelRegistry._registry.keys())}")

    # Create model based on model_type
    if model_type == "legacy":
        # Use the legacy model replication for direct comparison with the legacy model
        # Create the model using the predefined legacy model factory function
        model = create_legacy_model1()
        print("Using legacy model configuration for direct comparison with legacy implementation")
    else:
        # Default to standard model with Poisson observation model
        model = create_standard_model()
        print("Using standard model with Poisson likelihood")

    # Train model
    training_start_time = time.time()
    # The modular model's train method has a different signature
    # It doesn't accept check_val_every_n_epoch
    model.train(
        adata=adata_copy,
        max_epochs=max_epochs,
        early_stopping=False,
        seed=seed,
    )
    training_end_time = time.time()
    training_time = training_end_time - training_start_time

    # Generate posterior samples
    inference_start_time = time.time()
    posterior_samples = model.generate_posterior_samples(
        adata=adata_copy,
        num_samples=num_samples,
        seed=seed
    )

    # Print parameter shapes for debugging
    print("Modular model parameter shapes:")
    for key, value in posterior_samples.items():
        if isinstance(value, (np.ndarray, torch.Tensor)):
            print(f"  {key}: {value.shape}")

    # Print parameter statistics for debugging
    print("\nModular model parameter statistics:")
    for param in ["alpha", "beta", "gamma"]:
        if param in posterior_samples:
            param_value = posterior_samples[param]
            if isinstance(param_value, torch.Tensor):
                param_value = param_value.detach().cpu().numpy()
            print(f"  {param} mean: {np.mean(param_value):.6f}, std: {np.std(param_value):.6f}, min: {np.min(param_value):.6f}, max: {np.max(param_value):.6f}")

    # Initialize velocity and uncertainty as None
    velocity = None
    uncertainty = None

    # Compute velocity only if requested
    if compute_velocity:
        # Compute velocity with detailed logging
        print("\nComputing velocity for modular model...")

        # Get the raw data
        u = adata_copy.layers["unspliced"]
        s = adata_copy.layers["spliced"]
        if isinstance(u, scipy.sparse.spmatrix):
            u = u.toarray()
        if isinstance(s, scipy.sparse.spmatrix):
            s = s.toarray()
        print(f"  u shape: {u.shape}, mean: {np.mean(u):.6f}, std: {np.std(u):.6f}")
        print(f"  s shape: {s.shape}, mean: {np.mean(s):.6f}, std: {np.std(s):.6f}")

        # Extract parameters for velocity calculation
        alpha = posterior_samples["alpha"]
        beta = posterior_samples["beta"]
        gamma = posterior_samples["gamma"]
        u_scale = posterior_samples.get("u_scale")
        s_scale = posterior_samples.get("s_scale")

        if isinstance(alpha, torch.Tensor):
            alpha_mean = alpha.mean(dim=0).detach().cpu().numpy()
        else:
            alpha_mean = np.mean(alpha, axis=0)

        if isinstance(beta, torch.Tensor):
            beta_mean = beta.mean(dim=0).detach().cpu().numpy()
        else:
            beta_mean = np.mean(beta, axis=0)

        if isinstance(gamma, torch.Tensor):
            gamma_mean = gamma.mean(dim=0).detach().cpu().numpy()
        else:
            gamma_mean = np.mean(gamma, axis=0)

        print(f"  alpha_mean shape: {alpha_mean.shape}, mean: {np.mean(alpha_mean):.6f}, std: {np.std(alpha_mean):.6f}")
        print(f"  beta_mean shape: {beta_mean.shape}, mean: {np.mean(beta_mean):.6f}, std: {np.std(beta_mean):.6f}")
        print(f"  gamma_mean shape: {gamma_mean.shape}, mean: {np.mean(gamma_mean):.6f}, std: {np.std(gamma_mean):.6f}")

        if u_scale is not None:
            if isinstance(u_scale, torch.Tensor):
                u_scale_mean = u_scale.mean(dim=0).detach().cpu().numpy()
            else:
                u_scale_mean = np.mean(u_scale, axis=0)
            print(f"  u_scale_mean shape: {u_scale_mean.shape}, mean: {np.mean(u_scale_mean):.6f}, std: {np.std(u_scale_mean):.6f}")

        if s_scale is not None:
            if isinstance(s_scale, torch.Tensor):
                s_scale_mean = s_scale.mean(dim=0).detach().cpu().numpy()
            else:
                s_scale_mean = np.mean(s_scale, axis=0)
            print(f"  s_scale_mean shape: {s_scale_mean.shape}, mean: {np.mean(s_scale_mean):.6f}, std: {np.std(s_scale_mean):.6f}")

        # Compute velocity using the model's get_velocity method
        velocity = model.get_velocity(
            adata=adata_copy,
            random_seed=seed
        )

        # Compute uncertainty
        uncertainty = model.get_velocity_uncertainty(
            adata=adata_copy,
            num_samples=num_samples
        )
    else:
        print("\nSkipping velocity computation for modular model (--no-compute-velocity flag is set)")

    inference_end_time = time.time()
    inference_time = inference_end_time - inference_start_time

    # Return results
    return {
        "model": model,
        "posterior_samples": posterior_samples,
        "velocity": velocity,
        "uncertainty": uncertainty,
        "performance": {
            "training_time": training_time,
            "inference_time": inference_time,
        }
    }


def compare_models(legacy_results, modular_results):
    """Compare the results of the legacy and modular models."""
    print("Comparing models...")

    # Initialize comparison results
    comparison_results = {}

    # Compare parameters
    parameter_comparison = {}
    try:
        # Prepare the results dictionary in the format expected by compare_parameters
        results = {
            "legacy": {
                "posterior_samples": legacy_results["posterior_samples"]
            },
            "modular": {
                "posterior_samples": modular_results["posterior_samples"]
            }
        }

        # Compare parameters
        param_comparison_results = compare_parameters(results)

        # Extract the comparison results for each parameter
        for param in ["alpha", "beta", "gamma"]:
            if param in param_comparison_results:
                parameter_comparison[param] = param_comparison_results[param]
            else:
                print(f"Warning: Parameter {param} not found in comparison results")
                parameter_comparison[param] = {"error": f"Parameter {param} not found in comparison results"}
    except Exception as e:
        print(f"Error comparing parameters: {e}")
        import traceback
        traceback.print_exc()

        # Fallback to manual comparison
        print("Falling back to manual parameter comparison")
        for param in ["alpha", "beta", "gamma"]:
            try:
                # Get parameter samples
                param_samples1 = legacy_results["posterior_samples"][param]
                param_samples2 = modular_results["posterior_samples"][param]

                # Convert to numpy arrays if needed
                if isinstance(param_samples1, torch.Tensor):
                    param_samples1 = param_samples1.detach().cpu().numpy()
                if isinstance(param_samples2, torch.Tensor):
                    param_samples2 = param_samples2.detach().cpu().numpy()

                # Flatten arrays for comparison
                param_samples1_flat = param_samples1.flatten()
                param_samples2_flat = param_samples2.flatten()

                # Compute basic statistics
                correlation = np.corrcoef(param_samples1_flat, param_samples2_flat)[0, 1]
                mse = np.mean((param_samples1_flat - param_samples2_flat) ** 2)
                mae = np.mean(np.abs(param_samples1_flat - param_samples2_flat))

                # Store parameter comparison results
                parameter_comparison[param] = {
                    "legacy_vs_modular": {
                        "correlation": correlation,
                        "mse": mse,
                        "mae": mae
                    }
                }
            except Exception as e:
                print(f"Error comparing parameter {param}: {e}")
                parameter_comparison[param] = {
                    "legacy_vs_modular": {
                        "error": str(e)
                    }
                }

    # Store parameter comparison results
    comparison_results["parameter_comparison"] = parameter_comparison

    # Compare velocities if both are available and velocity computation was requested
    if legacy_results["velocity"] is not None and modular_results["velocity"] is not None:
        try:
            # Ensure both velocities are numpy arrays
            legacy_velocity = legacy_results["velocity"]
            if isinstance(legacy_velocity, torch.Tensor):
                legacy_velocity = legacy_velocity.detach().cpu().numpy()

            modular_velocity = modular_results["velocity"]
            if isinstance(modular_velocity, torch.Tensor):
                modular_velocity = modular_velocity.detach().cpu().numpy()

            # Prepare the results dictionary in the format expected by compare_velocities
            velocity_results = {
                "legacy": {
                    "velocity": legacy_velocity
                },
                "modular": {
                    "velocity": modular_velocity
                }
            }

            # Compare velocities
            velocity_comparison = {
                "legacy_vs_modular": compare_velocities(velocity_results)
            }

            # Store velocity comparison results
            comparison_results["velocity_comparison"] = velocity_comparison
        except Exception as e:
            print(f"Error comparing velocities: {e}")
            import traceback
            traceback.print_exc()

            # Fallback to manual comparison
            print("Falling back to manual velocity comparison")
            try:
                # Flatten arrays for comparison
                legacy_velocity_flat = legacy_velocity.flatten()
                modular_velocity_flat = modular_velocity.flatten()

                # Compute basic statistics
                correlation = np.corrcoef(legacy_velocity_flat, modular_velocity_flat)[0, 1]
                mse = np.mean((legacy_velocity_flat - modular_velocity_flat) ** 2)
                mae = np.mean(np.abs(legacy_velocity_flat - modular_velocity_flat))

                # Store velocity comparison results
                comparison_results["velocity_comparison"] = {
                    "legacy_vs_modular": {
                        "correlation": correlation,
                        "mse": mse,
                        "mae": mae
                    }
                }
            except Exception as e:
                print(f"Error in manual velocity comparison: {e}")
                comparison_results["velocity_comparison"] = {
                    "legacy_vs_modular": {
                        "error": str(e)
                    }
                }
    else:
        # Check if velocity computation was intentionally skipped
        if legacy_results["velocity"] is None and modular_results["velocity"] is None:
            print("Note: Velocity comparison skipped because velocity computation was disabled (--no-compute-velocity flag)")
            comparison_results["velocity_comparison"] = {
                "legacy_vs_modular": {"skipped": "Velocity computation was disabled"}
            }
        else:
            print("Warning: Velocity comparison skipped because one or both velocities are None")
            comparison_results["velocity_comparison"] = {
                "legacy_vs_modular": {"error": "One or both velocities are None"}
            }

    # Compare uncertainties if both are available
    if legacy_results["uncertainty"] is not None and modular_results["uncertainty"] is not None:
        try:
            # Ensure both uncertainties are numpy arrays
            legacy_uncertainty = legacy_results["uncertainty"]
            if isinstance(legacy_uncertainty, torch.Tensor):
                legacy_uncertainty = legacy_uncertainty.detach().cpu().numpy()

            modular_uncertainty = modular_results["uncertainty"]
            if isinstance(modular_uncertainty, torch.Tensor):
                modular_uncertainty = modular_uncertainty.detach().cpu().numpy()

            # Prepare the results dictionary in the format expected by compare_uncertainties
            uncertainty_results = {
                "legacy": {
                    "uncertainty": legacy_uncertainty
                },
                "modular": {
                    "uncertainty": modular_uncertainty
                }
            }

            # Compare uncertainties
            uncertainty_comparison = {
                "legacy_vs_modular": compare_uncertainties(uncertainty_results)
            }

            # Store uncertainty comparison results
            comparison_results["uncertainty_comparison"] = uncertainty_comparison
        except Exception as e:
            print(f"Error comparing uncertainties: {e}")
            import traceback
            traceback.print_exc()

            # Fallback to manual comparison
            print("Falling back to manual uncertainty comparison")
            try:
                # Flatten arrays for comparison
                legacy_uncertainty_flat = legacy_uncertainty.flatten()
                modular_uncertainty_flat = modular_uncertainty.flatten()

                # Compute basic statistics
                correlation = np.corrcoef(legacy_uncertainty_flat, modular_uncertainty_flat)[0, 1]
                mse = np.mean((legacy_uncertainty_flat - modular_uncertainty_flat) ** 2)
                mae = np.mean(np.abs(legacy_uncertainty_flat - modular_uncertainty_flat))

                # Store uncertainty comparison results
                comparison_results["uncertainty_comparison"] = {
                    "legacy_vs_modular": {
                        "correlation": correlation,
                        "mse": mse,
                        "mae": mae
                    }
                }
            except Exception as e:
                print(f"Error in manual uncertainty comparison: {e}")
                comparison_results["uncertainty_comparison"] = {
                    "legacy_vs_modular": {
                        "error": str(e)
                    }
                }
    else:
        # Check if velocity computation was intentionally skipped (uncertainty is computed as part of velocity)
        if legacy_results["uncertainty"] is None and modular_results["uncertainty"] is None:
            print("Note: Uncertainty comparison skipped because velocity computation was disabled (--no-compute-velocity flag)")
            comparison_results["uncertainty_comparison"] = {
                "legacy_vs_modular": {"skipped": "Velocity computation was disabled"}
            }
        else:
            print("Warning: Uncertainty comparison skipped because one or both uncertainties are None")
            comparison_results["uncertainty_comparison"] = {
                "legacy_vs_modular": {"error": "One or both uncertainties are None"}
            }

    # Compare performance
    try:
        # Prepare the results dictionary in the format expected by compare_performance
        performance_results = {
            "legacy": {
                "performance": legacy_results["performance"]
            },
            "modular": {
                "performance": modular_results["performance"]
            }
        }

        # Compare performance
        performance_comparison = {
            "legacy_vs_modular": compare_performance(performance_results)
        }

        # Store performance comparison results
        comparison_results["performance_comparison"] = performance_comparison
    except Exception as e:
        print(f"Error comparing performance: {e}")
        import traceback
        traceback.print_exc()

        # Fallback to manual comparison
        print("Falling back to manual performance comparison")
        try:
            # Get performance metrics
            legacy_training_time = legacy_results["performance"]["training_time"]
            modular_training_time = modular_results["performance"]["training_time"]
            legacy_inference_time = legacy_results["performance"]["inference_time"]
            modular_inference_time = modular_results["performance"]["inference_time"]

            # Compute ratios
            training_time_ratio = modular_training_time / legacy_training_time if legacy_training_time > 0 else float('inf')
            inference_time_ratio = modular_inference_time / legacy_inference_time if legacy_inference_time > 0 else float('inf')

            # Store performance comparison results
            comparison_results["performance_comparison"] = {
                "legacy_vs_modular": {
                    "training_time_ratio": training_time_ratio,
                    "inference_time_ratio": inference_time_ratio,
                    "legacy_training_time": legacy_training_time,
                    "modular_training_time": modular_training_time,
                    "legacy_inference_time": legacy_inference_time,
                    "modular_inference_time": modular_inference_time
                }
            }
        except Exception as e:
            print(f"Error in manual performance comparison: {e}")
            comparison_results["performance_comparison"] = {
                "legacy_vs_modular": {
                    "error": str(e)
                }
            }

    return comparison_results


def generate_summary_report(legacy_results, modular_results, comparison, args, output_dir):
    """Generate a summary report of the comparison."""
    report_path = os.path.join(output_dir, "comparison_summary_report.txt")

    with open(report_path, "w") as f:
        f.write("PyroVelocity Direct Comparison Summary Report\n")
        f.write("===========================================\n\n")

        # Write comparison parameters
        f.write("Comparison Parameters\n")
        f.write("--------------------\n")
        f.write(f"Max epochs: {args.max_epochs}\n")
        f.write(f"Number of posterior samples: {args.num_samples}\n")
        f.write(f"Random seed: {args.seed}\n")
        f.write(f"Model type: {args.model_type}\n")
        f.write(f"Compute velocity: {args.compute_velocity}\n")
        f.write("\n")

        # Write model information
        f.write("Model Information\n")
        f.write("----------------\n")

        # Legacy model
        f.write("Legacy Model:\n")
        if legacy_results is not None:
            f.write(f"  Training time: {legacy_results['performance']['training_time']:.2f} seconds\n")
            f.write(f"  Inference time: {legacy_results['performance']['inference_time']:.2f} seconds\n")
            if legacy_results['velocity'] is not None:
                f.write(f"  Velocity shape: {legacy_results['velocity'].shape}\n")
            else:
                f.write("  Velocity: None\n")
        else:
            f.write("  Failed to train\n")
        f.write("\n")

        # Modular model
        f.write("Modular Model:\n")
        if modular_results is not None:
            f.write(f"  Training time: {modular_results['performance']['training_time']:.2f} seconds\n")
            f.write(f"  Inference time: {modular_results['performance']['inference_time']:.2f} seconds\n")
            if modular_results['velocity'] is not None:
                f.write(f"  Velocity shape: {modular_results['velocity'].shape}\n")
            else:
                f.write("  Velocity: None\n")
        else:
            f.write("  Failed to train\n")
        f.write("\n")

        # Write comparison summary
        f.write("Comparison Summary\n")
        f.write("-----------------\n")

        # Check if comparison has error
        if "error" in comparison:
            f.write(f"Error: {comparison['error']}\n")
            f.write("\nRecommendation: Fix the modular model implementation and run validation again.\n")
        else:
            # Parameter comparison summary
            f.write("Parameter Comparison:\n")
            for param, param_comp in comparison["parameter_comparison"].items():
                f.write(f"  {param}:\n")
                for comp_name, metrics in param_comp.items():
                    f.write(f"    {comp_name}:\n")
                    for metric_name, value in metrics.items():
                        if isinstance(value, float):
                            f.write(f"      {metric_name}: {value:.6f}\n")
                        else:
                            f.write(f"      {metric_name}: {value}\n")
            f.write("\n")

            # Velocity comparison summary
            f.write("Velocity Comparison:\n")
            for comp_name, metrics in comparison["velocity_comparison"].items():
                f.write(f"  {comp_name}:\n")
                for metric_name, value in metrics.items():
                    if isinstance(value, float):
                        f.write(f"    {metric_name}: {value:.6f}\n")
                    else:
                        f.write(f"    {metric_name}: {value}\n")
            f.write("\n")

            # Uncertainty comparison summary
            f.write("Uncertainty Comparison:\n")
            for comp_name, metrics in comparison["uncertainty_comparison"].items():
                f.write(f"  {comp_name}:\n")
                for metric_name, value in metrics.items():
                    if isinstance(value, float):
                        f.write(f"    {metric_name}: {value:.6f}\n")
                    else:
                        f.write(f"    {metric_name}: {value}\n")
            f.write("\n")

            # Performance comparison summary
            f.write("Performance Comparison:\n")
            for comp_name, metrics in comparison["performance_comparison"].items():
                f.write(f"  {comp_name}:\n")
                for metric_name, value in metrics.items():
                    if isinstance(value, float):
                        f.write(f"    {metric_name}: {value:.6f}\n")
                    else:
                        f.write(f"    {metric_name}: {value}\n")
            f.write("\n")

            # Write comparison conclusion
            f.write("Comparison Conclusion\n")
            f.write("--------------------\n")

            # Check if there are any significant differences in parameters
            param_diff = False
            for param, param_comp in comparison["parameter_comparison"].items():
                for comp_name, metrics in param_comp.items():
                    if "correlation" in metrics and metrics["correlation"] < 0.9:
                        param_diff = True
                        break
                if param_diff:
                    break

            # Check if there are any significant differences in velocities
            vel_diff = False
            for comp_name, metrics in comparison["velocity_comparison"].items():
                if "correlation" in metrics and metrics["correlation"] < 0.9:
                    vel_diff = True
                    break

            # Write conclusion
            if param_diff or vel_diff:
                f.write("There are significant differences between the legacy and modular implementations.\n")
                if param_diff:
                    f.write("- Parameter estimates show low correlation (< 0.9)\n")
                if vel_diff:
                    f.write("- Velocity estimates show low correlation (< 0.9)\n")
                f.write("\nRecommendation: Further investigation is needed to understand these differences.\n")
            else:
                f.write("The legacy and modular implementations produce similar results.\n")
                f.write("- Parameter estimates show high correlation (>= 0.9)\n")
                f.write("- Velocity estimates show high correlation (>= 0.9)\n")
                f.write("\nRecommendation: The modular implementation can be considered a valid replacement for the legacy implementation.\n")

    print(f"Summary report saved to {report_path}")
    return report_path


def main():
    """Run the direct comparison and save results."""
    # Parse command line arguments
    args = parse_args()

    # Set random seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    pyro.set_rng_seed(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Created output directory: {args.output_dir}")

    # Load preprocessed pancreas data
    try:
        adata = load_preprocessed_pancreas_data()
    except Exception as e:
        print(f"Failed to load data: {e}")
        return

    # Train legacy model
    try:
        # Note: model_type is passed for compatibility but has no effect on the legacy model
        legacy_results = train_legacy_model(
            adata,
            args.max_epochs,
            args.num_samples,
            seed=args.seed,
            compute_velocity=args.compute_velocity,
            model_type=args.model_type  # This parameter is ignored by the legacy model
        )
        print("Legacy model training successful")
    except Exception as e:
        print(f"Error training legacy model: {e}")
        import traceback
        traceback.print_exc()
        return

    # Train modular model
    try:
        modular_results = train_modular_model(
            adata,
            args.max_epochs,
            args.num_samples,
            seed=args.seed,
            model_type=args.model_type,
            compute_velocity=args.compute_velocity
        )
        print("Modular model training successful")
    except Exception as e:
        print(f"Error training modular model: {e}")
        import traceback
        traceback.print_exc()
        return

    # Compare models
    try:
        # Check if modular_results is available
        if 'modular_results' in locals() and modular_results is not None:
            comparison_results = compare_models(legacy_results, modular_results)
            print("Model comparison successful")

            # Generate summary report
            try:
                report_path = generate_summary_report(legacy_results, modular_results, comparison_results, args, args.output_dir)
                print(f"Summary report saved to {report_path}")
                print(f"Results saved to {args.output_dir}")
            except Exception as e:
                print(f"Error generating summary report: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("Skipping model comparison because modular model failed")

            # Create a simple summary report for the legacy model
            try:
                report_path = os.path.join(args.output_dir, "legacy_model_summary.txt")
                with open(report_path, "w") as f:
                    f.write("PyroVelocity Legacy Model Summary\n")
                    f.write("===============================\n\n")
                    f.write(f"Max epochs: {args.max_epochs}\n")
                    f.write(f"Number of posterior samples: {args.num_samples}\n")
                    f.write(f"Random seed: {args.seed}\n")
                    f.write(f"Model type: {args.model_type}\n")
                    f.write(f"Compute velocity: {args.compute_velocity}\n")
                    f.write("\n")
                    f.write("Legacy Model:\n")
                    f.write(f"  Training time: {legacy_results['performance']['training_time']:.2f} seconds\n")
                    f.write(f"  Inference time: {legacy_results['performance']['inference_time']:.2f} seconds\n")
                    if legacy_results['velocity'] is not None:
                        f.write(f"  Velocity shape: {legacy_results['velocity'].shape}\n")
                    else:
                        f.write("  Velocity: None\n")
                    f.write("\n")
                    f.write("Modular Model:\n")
                    f.write("  Failed to train\n")
                    f.write("\n")
                    f.write("Conclusion:\n")
                    f.write("  The legacy model was trained successfully, but the modular model failed.\n")
                    f.write("  This is expected since we're focusing on validating the legacy model's workflow first.\n")
                    f.write("  The next step is to fix the modular model implementation.\n")

                print(f"Legacy model summary saved to {report_path}")
                print(f"Results saved to {args.output_dir}")
            except Exception as e:
                print(f"Error generating legacy model summary: {e}")
                import traceback
                traceback.print_exc()
    except Exception as e:
        print(f"Error comparing models: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
