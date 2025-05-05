"""
Posterior analysis utilities for PyroVelocity PyTorch/Pyro modular implementation.

This module contains posterior analysis utilities, including:

- sample_posterior: Sample from the posterior
- compute_velocity: Compute RNA velocity from posterior samples
- compute_uncertainty: Compute uncertainty in RNA velocity
- analyze_posterior: Analyze posterior samples from either SVI or MCMC
- create_inference_data: Create ArviZ InferenceData object from posterior samples
- format_anndata_output: Format results into AnnData object
"""

import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pyro
import pyro.distributions as dist
import pyro.infer as infer
import torch
from anndata import AnnData
from beartype import beartype
from jaxtyping import Array, Float

from pyrovelocity.models.modular.inference.unified import (
    InferenceState,
    extract_posterior_samples,
    posterior_predictive,
)
from pyrovelocity.models.modular.model import PyroVelocityModel


@beartype
def sample_posterior(
    model: Union[Callable, PyroVelocityModel],
    state: InferenceState,
    num_samples: int = 1000,
    seed: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """
    Sample from the posterior.

    Args:
        model: Pyro model function or PyroVelocityModel
        state: Inference state
        num_samples: Number of posterior samples
        seed: Random seed

    Returns:
        Dictionary of posterior samples
    """
    # Extract posterior samples from inference state
    return extract_posterior_samples(state, num_samples)


@beartype
def compute_velocity(
    model: Union[Callable, PyroVelocityModel],
    posterior_samples: Dict[str, Union[torch.Tensor, np.ndarray]],
    adata: Optional[AnnData] = None,
    use_mean: bool = False,
) -> Dict[str, Union[torch.Tensor, np.ndarray]]:
    """
    Compute RNA velocity from posterior samples.

    Args:
        model: Pyro model function or PyroVelocityModel
        posterior_samples: Posterior samples from inference
        adata: AnnData object
        use_mean: Whether to use mean of posterior samples

    Returns:
        Dictionary containing velocity results
    """
    # Extract parameters from posterior samples
    alpha = posterior_samples.get("alpha")
    beta = posterior_samples.get("beta")
    gamma = posterior_samples.get("gamma")

    if alpha is None or beta is None or gamma is None:
        raise ValueError(
            "Posterior samples must contain alpha, beta, and gamma"
        )

    # Convert numpy arrays to torch tensors if needed
    if isinstance(alpha, np.ndarray):
        alpha = torch.tensor(alpha)
    if isinstance(beta, np.ndarray):
        beta = torch.tensor(beta)
    if isinstance(gamma, np.ndarray):
        gamma = torch.tensor(gamma)

    # Get unspliced and spliced counts
    if adata is not None:
        u_layer = adata.layers["unspliced"]
        s_layer = adata.layers["spliced"]
        # Handle sparse matrices
        if hasattr(u_layer, "toarray"):
            u = torch.tensor(u_layer.toarray())
            s = torch.tensor(s_layer.toarray())
        else:
            u = torch.tensor(u_layer)
            s = torch.tensor(s_layer)
    else:
        u = posterior_samples.get("u")
        s = posterior_samples.get("s")

        # Convert numpy arrays to torch tensors if needed
        if isinstance(u, np.ndarray):
            u = torch.tensor(u)
        if isinstance(s, np.ndarray):
            s = torch.tensor(s)

    if u is None or s is None:
        raise ValueError("Unable to get unspliced and spliced counts")

    # Compute steady state
    u_ss = alpha / beta
    s_ss = alpha / gamma

    # Compute velocity
    if use_mean:
        # Use mean of posterior samples
        alpha_mean = alpha.mean(dim=0)
        beta_mean = beta.mean(dim=0)
        gamma_mean = gamma.mean(dim=0)
        u_ss_mean = alpha_mean / beta_mean
        s_ss_mean = alpha_mean / gamma_mean
        velocity = beta_mean * (u - u_ss_mean) - gamma_mean * (s - s_ss_mean)

        # Compute latent time (pseudotime)
        # This is a simple implementation based on the ratio of unspliced to spliced
        # More sophisticated methods could be used

        # Handle 1D tensors
        if u.dim() == 1:
            u_norm = u / (u.max() + 1e-6)
            s_norm = s / (s.max() + 1e-6)
            latent_time = torch.tensor(1.0 - u_norm.mean() / (s_norm.mean() + 1e-6))
        else:
            u_norm = u / u.max(dim=1, keepdim=True)[0]
            s_norm = s / s.max(dim=1, keepdim=True)[0]
            latent_time = 1.0 - u_norm.mean(dim=1) / (s_norm.mean(dim=1) + 1e-6)

        return {
            "velocity": velocity,
            "alpha": alpha_mean,
            "beta": beta_mean,
            "gamma": gamma_mean,
            "u_ss": u_ss_mean,
            "s_ss": s_ss_mean,
            "latent_time": latent_time,
        }
    else:
        # Check if the shapes are compatible for broadcasting
        # If not, use the mean of the parameters
        if alpha.shape[0] != u.shape[0] and u.shape[0] > 1:
            print(f"Shape mismatch: alpha shape {alpha.shape}, u shape {u.shape}")
            print(f"Using mean of parameters for velocity calculation")
            # Use mean of posterior samples
            alpha_mean = alpha.mean(dim=0)
            beta_mean = beta.mean(dim=0)
            gamma_mean = gamma.mean(dim=0)
            u_ss_mean = alpha_mean / beta_mean
            s_ss_mean = alpha_mean / gamma_mean

            # Ensure the shapes are compatible for broadcasting
            if u.dim() == 1 and u_ss_mean.dim() > 1:
                # If u is 1D but u_ss_mean is multi-dimensional, flatten u_ss_mean
                u_ss_mean = u_ss_mean.flatten()
                s_ss_mean = s_ss_mean.flatten()
            elif u.dim() > 1 and u_ss_mean.dim() == 1:
                # If u is multi-dimensional but u_ss_mean is 1D, expand u_ss_mean
                if u.shape[1] == u_ss_mean.shape[0]:
                    # If the second dimension of u matches the size of u_ss_mean, expand along first dimension
                    u_ss_mean = u_ss_mean.unsqueeze(0).expand(u.shape[0], -1)
                    s_ss_mean = s_ss_mean.unsqueeze(0).expand(u.shape[0], -1)
                else:
                    # Otherwise, try to reshape to match
                    u_ss_mean = u_ss_mean.reshape(1, -1).expand(u.shape[0], -1)
                    s_ss_mean = s_ss_mean.reshape(1, -1).expand(u.shape[0], -1)

            # Compute velocity
            velocity = beta_mean * (u - u_ss_mean) - gamma_mean * (s - s_ss_mean)

            # Compute latent time (pseudotime)
            # This is a simple implementation based on the ratio of unspliced to spliced
            # More sophisticated methods could be used

            # Handle 1D tensors
            if u.dim() == 1:
                u_norm = u / (u.max() + 1e-6)
                s_norm = s / (s.max() + 1e-6)
                latent_time = torch.tensor(1.0 - u_norm.mean() / (s_norm.mean() + 1e-6))
            else:
                u_norm = u / u.max(dim=1, keepdim=True)[0]
                s_norm = s / s.max(dim=1, keepdim=True)[0]
                latent_time = 1.0 - u_norm.mean(dim=1) / (s_norm.mean(dim=1) + 1e-6)

            return {
                "velocity": velocity,
                "alpha": alpha,
                "beta": beta,
                "gamma": gamma,
                "u_ss": u_ss,
                "s_ss": s_ss,
                "latent_time": latent_time,
            }
        else:
            # Compute velocity for each posterior sample
            # We need to ensure the shapes are compatible for broadcasting
            # If alpha has shape [num_samples, n_genes] and u has shape [n_cells, n_genes],
            # we need to reshape for proper broadcasting
            if alpha.dim() == 2 and u.dim() == 2 and alpha.shape[0] != u.shape[0]:
                # Reshape u and s to [1, n_cells, n_genes] for broadcasting with
                # alpha, beta, gamma of shape [num_samples, 1, n_genes]
                u_reshaped = u.unsqueeze(0)  # [1, n_cells, n_genes]
                s_reshaped = s.unsqueeze(0)  # [1, n_cells, n_genes]

                # Reshape alpha, beta, gamma to [num_samples, 1, n_genes]
                alpha_reshaped = alpha.unsqueeze(1)  # [num_samples, 1, n_genes]
                beta_reshaped = beta.unsqueeze(1)    # [num_samples, 1, n_genes]
                gamma_reshaped = gamma.unsqueeze(1)  # [num_samples, 1, n_genes]

                # Reshape u_ss and s_ss to [num_samples, 1, n_genes]
                u_ss_reshaped = u_ss.unsqueeze(1)    # [num_samples, 1, n_genes]
                s_ss_reshaped = s_ss.unsqueeze(1)    # [num_samples, 1, n_genes]

                # Compute velocity with broadcasting
                # This will result in shape [num_samples, n_cells, n_genes]
                velocity = beta_reshaped * (u_reshaped - u_ss_reshaped) - gamma_reshaped * (s_reshaped - s_ss_reshaped)
            else:
                # Standard computation when shapes are compatible
                velocity = beta * (u - u_ss) - gamma * (s - s_ss)

            # Compute latent time (pseudotime)
            # This is a simple implementation based on the ratio of unspliced to spliced
            # More sophisticated methods could be used

            # Handle 1D tensors
            if u.dim() == 1:
                u_norm = u / (u.max() + 1e-6)
                s_norm = s / (s.max() + 1e-6)
                latent_time = torch.tensor(1.0 - u_norm.mean() / (s_norm.mean() + 1e-6))
            else:
                u_norm = u / u.max(dim=1, keepdim=True)[0]
                s_norm = s / s.max(dim=1, keepdim=True)[0]
                latent_time = 1.0 - u_norm.mean(dim=1) / (s_norm.mean(dim=1) + 1e-6)

            return {
                "velocity": velocity,
                "alpha": alpha,
                "beta": beta,
                "gamma": gamma,
                "u_ss": u_ss,
                "s_ss": s_ss,
                "latent_time": latent_time,
            }


@beartype
def compute_uncertainty(
    velocity_samples: Union[torch.Tensor, Dict[str, torch.Tensor]],
    method: str = "std",
) -> torch.Tensor:
    """
    Compute uncertainty in RNA velocity.

    Args:
        velocity_samples: Velocity samples from posterior, either a tensor or a dictionary with a 'velocity' key
        method: Method for computing uncertainty ("std", "quantile", "entropy")

    Returns:
        Uncertainty in RNA velocity
    """
    # Handle dictionary input
    if isinstance(velocity_samples, dict):
        if 'velocity' not in velocity_samples:
            raise ValueError("velocity_samples dictionary must contain a 'velocity' key")
        velocity_samples = velocity_samples['velocity']

    if method == "std":
        # Standard deviation across samples
        return velocity_samples.std(dim=0)
    elif method == "quantile":
        # Interquartile range across samples
        q75 = torch.quantile(velocity_samples, 0.75, dim=0)
        q25 = torch.quantile(velocity_samples, 0.25, dim=0)
        return q75 - q25
    elif method == "entropy":
        # Approximate entropy using histogram
        # This is a simplified version and may not be accurate for all distributions
        try:
            import numpy as np
            from scipy.stats import entropy

            # Convert to numpy for scipy
            velocity_np = velocity_samples.detach().cpu().numpy()
            # Compute entropy for each gene
            uncertainties = []
            for gene_idx in range(velocity_np.shape[1]):
                hist, _ = np.histogram(
                    velocity_np[:, gene_idx], bins=20, density=True
                )
                uncertainties.append(entropy(hist))
            return torch.tensor(uncertainties)
        except ImportError:
            warnings.warn("scipy not available, falling back to std method")
            return velocity_samples.std(dim=0)
    else:
        raise ValueError(f"Unsupported uncertainty method: {method}")


@beartype
def analyze_posterior(
    state: InferenceState,
    model: Union[Callable, PyroVelocityModel],
    adata: Optional[AnnData] = None,
    num_samples: int = 1000,
    compute_velocity_flag: bool = True,
    compute_uncertainty_flag: bool = True,
    uncertainty_method: str = "std",
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Analyze posterior samples from either SVI or MCMC.

    Args:
        state: Inference state
        model: Pyro model function or PyroVelocityModel
        adata: AnnData object
        num_samples: Number of posterior samples
        compute_velocity_flag: Whether to compute velocity
        compute_uncertainty_flag: Whether to compute uncertainty
        uncertainty_method: Method for computing uncertainty
        seed: Random seed

    Returns:
        Dictionary containing analysis results
    """
    # Sample from posterior
    posterior_samples = sample_posterior(model, state, num_samples, seed)

    # Initialize results dictionary
    results = {"posterior_samples": posterior_samples}

    # Compute velocity if requested
    if compute_velocity_flag:
        velocity_results = compute_velocity(model, posterior_samples, adata)
        # Update results dictionary with velocity results
        for k, v in velocity_results.items():
            results[k] = v

        # Compute uncertainty if requested
        if compute_uncertainty_flag and "velocity" in velocity_results:
            velocity = velocity_results["velocity"]
            # Convert numpy array to tensor if needed
            if isinstance(velocity, np.ndarray):
                velocity = torch.tensor(velocity)
            uncertainty = compute_uncertainty(velocity, uncertainty_method)
            results["uncertainty"] = uncertainty

    return results


@beartype
def create_inference_data(
    posterior_samples: Dict[str, torch.Tensor],
    observed_data: Optional[Dict[str, torch.Tensor]] = None,
    prior_samples: Optional[Dict[str, torch.Tensor]] = None,
) -> Any:
    """
    Create ArviZ InferenceData object from posterior samples.

    Args:
        posterior_samples: Posterior samples from inference
        observed_data: Observed data
        prior_samples: Prior samples

    Returns:
        ArviZ InferenceData object
    """
    try:
        import arviz as az
    except ImportError:
        warnings.warn("arviz not available, returning samples as dictionary")
        return {
            "posterior": posterior_samples,
            "observed_data": observed_data,
            "prior": prior_samples,
        }

    # Convert torch tensors to numpy arrays
    posterior_np = {
        k: v.detach().cpu().numpy() for k, v in posterior_samples.items()
    }
    observed_np = (
        {k: v.detach().cpu().numpy() for k, v in observed_data.items()}
        if observed_data
        else None
    )
    prior_np = (
        {k: v.detach().cpu().numpy() for k, v in prior_samples.items()}
        if prior_samples
        else None
    )

    # Create InferenceData object
    return az.from_dict(
        posterior=posterior_np,
        observed_data=observed_np,
        prior=prior_np,
    )


@beartype
def format_anndata_output(
    adata: AnnData,
    results: Dict[str, Any],
    model_name: str = "pyrovelocity",
) -> AnnData:
    """
    Format results into AnnData object.

    Args:
        adata: AnnData object
        results: Results from analyze_posterior
        model_name: Name of the model

    Returns:
        AnnData object with results added
    """
    # Create a copy of the AnnData object
    adata = adata.copy()

    # Add posterior samples to AnnData object
    if "posterior_samples" in results:
        for key, value in results["posterior_samples"].items():
            # Convert to numpy array
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()

            # Special handling for alpha, beta, gamma - store in var
            if key in ["alpha", "beta", "gamma"]:
                # If value is multi-dimensional, take the mean across samples
                if value.ndim > 1:
                    value_mean = value.mean(axis=0)
                else:
                    value_mean = value

                # Ensure the value has the right shape for var
                if value_mean.shape[0] != adata.n_vars:
                    # Transpose if needed
                    if value_mean.shape[0] == adata.n_obs:
                        value_mean = value_mean.T
                    else:
                        # Reshape if possible
                        try:
                            value_mean = value_mean.reshape(adata.n_vars)
                        except ValueError:
                            warnings.warn(f"Could not reshape {key} to match var dimensions")
                            # Still store the original in uns
                            adata.uns[f"{model_name}_{key}"] = value
                            continue

                # Store in var dataframe
                adata.var[f"{model_name}_{key}"] = value_mean

            # Store original in uns for all parameters
            adata.uns[f"{model_name}_{key}"] = value

    # Add velocity to AnnData object
    if "velocity" in results:
        velocity = results["velocity"]
        if isinstance(velocity, torch.Tensor):
            velocity = velocity.detach().cpu().numpy()
        # Reshape velocity to match AnnData dimensions
        if velocity.ndim == 1:
            velocity = velocity.reshape(1, -1)
            # Transpose if needed to match AnnData dimensions
            if (
                velocity.shape[1] == adata.n_obs
                and velocity.shape[0] != adata.n_vars
            ):
                velocity = velocity.T
        adata.layers[f"{model_name}_velocity"] = velocity

    # Add uncertainty to AnnData object
    if "uncertainty" in results:
        uncertainty = results["uncertainty"]
        if isinstance(uncertainty, torch.Tensor):
            uncertainty = uncertainty.detach().cpu().numpy()
        adata.var[f"{model_name}_uncertainty"] = uncertainty

    # Add latent time to AnnData object
    if "latent_time" in results:
        latent_time = results["latent_time"]
        if isinstance(latent_time, torch.Tensor):
            latent_time = latent_time.detach().cpu().numpy()
        adata.obs[f"{model_name}_latent_time"] = latent_time

    # Add steady state to AnnData object
    if "u_ss" in results and "s_ss" in results:
        u_ss = results["u_ss"]
        s_ss = results["s_ss"]
        if isinstance(u_ss, torch.Tensor):
            u_ss = u_ss.detach().cpu().numpy()
        if isinstance(s_ss, torch.Tensor):
            s_ss = s_ss.detach().cpu().numpy()
        # Reshape steady states to match AnnData dimensions
        if u_ss.ndim == 1:
            u_ss = u_ss.reshape(1, -1)
            # Transpose if needed to match AnnData dimensions
            if u_ss.shape[1] == adata.n_obs and u_ss.shape[0] != adata.n_vars:
                u_ss = u_ss.T
        if s_ss.ndim == 1:
            s_ss = s_ss.reshape(1, -1)
            # Transpose if needed to match AnnData dimensions
            if s_ss.shape[1] == adata.n_obs and s_ss.shape[0] != adata.n_vars:
                s_ss = s_ss.T
        adata.layers[f"{model_name}_u_ss"] = u_ss
        adata.layers[f"{model_name}_s_ss"] = s_ss

    # Add parameters to AnnData object
    for param in ["alpha", "beta", "gamma"]:
        if param in results:
            value = results[param]
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            # If value is multi-dimensional, take the mean across samples
            if value.ndim > 1:
                value = value.mean(axis=0)
            # Ensure the value has the right shape for var
            if value.shape[0] != adata.n_vars:
                # Transpose if needed
                if value.shape[0] == adata.n_obs:
                    value = value.T
                else:
                    # Reshape if possible
                    try:
                        value = value.reshape(adata.n_vars)
                    except ValueError:
                        warnings.warn(f"Could not reshape {param} to match var dimensions")
                        continue
            adata.var[f"{model_name}_{param}"] = value

    return adata
