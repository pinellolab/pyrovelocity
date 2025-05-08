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

import logging
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

import numpy as np
import pyro
import pyro.distributions as dist
import pyro.infer as infer
import scipy.sparse
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

    This function computes velocity using the same approach as the legacy implementation
    in compute_mean_vector_field.

    Args:
        model: Pyro model function or PyroVelocityModel
        posterior_samples: Posterior samples from inference
        adata: AnnData object
        use_mean: Whether to use mean of posterior samples

    Returns:
        Dictionary containing velocity results
    """
    # Print all keys in posterior_samples for debugging
    print(f"compute_velocity - posterior_samples keys: {list(posterior_samples.keys())}")

    # Extract parameters from posterior samples
    alpha = posterior_samples.get("alpha")
    beta = posterior_samples.get("beta")
    gamma = posterior_samples.get("gamma")
    u_scale = posterior_samples.get("u_scale")
    s_scale = posterior_samples.get("s_scale")

    # Print shapes for debugging
    print(f"compute_velocity - alpha shape: {alpha.shape if alpha is not None else None}")
    print(f"compute_velocity - beta shape: {beta.shape if beta is not None else None}")
    print(f"compute_velocity - gamma shape: {gamma.shape if gamma is not None else None}")
    print(f"compute_velocity - u_scale shape: {u_scale.shape if u_scale is not None else None}")
    print(f"compute_velocity - s_scale shape: {s_scale.shape if s_scale is not None else None}")

    # Try to get ut and st from posterior samples (like legacy implementation)
    # These are the latent variables, not the observed data
    ut = posterior_samples.get("ut")
    st = posterior_samples.get("st")

    # Print shapes for debugging
    print(f"compute_velocity - ut shape: {ut.shape if ut is not None else None}")
    print(f"compute_velocity - st shape: {st.shape if st is not None else None}")

    # Ensure ut and st have the correct shapes
    # In the legacy model, ut and st have shape [num_samples, n_cells, n_genes]

    # Convert numpy arrays to torch tensors if needed
    if ut is not None and isinstance(ut, np.ndarray):
        ut = torch.tensor(ut)
    if st is not None and isinstance(st, np.ndarray):
        st = torch.tensor(st)

    if ut is not None and ut.dim() > 3:
        # Remove extra dimensions
        print(f"Reshaping ut from {ut.shape} to [num_samples, n_cells, n_genes]")
        ut = ut.squeeze()
        # If we squeezed too much, add back necessary dimensions
        if ut.dim() < 3:
            if ut.dim() == 2:
                # Check if first dimension is num_samples or n_cells
                if beta is not None and ut.shape[0] == beta.shape[0]:
                    # [num_samples, n_genes] -> [num_samples, 1, n_genes]
                    ut = ut.unsqueeze(1)
                else:
                    # [n_cells, n_genes] -> [1, n_cells, n_genes]
                    ut = ut.unsqueeze(0)
            elif ut.dim() == 1:
                # [n_genes] -> [1, 1, n_genes]
                ut = ut.unsqueeze(0).unsqueeze(0)
        print(f"Reshaped ut to {ut.shape}")

    if st is not None and isinstance(st, torch.Tensor) and st.dim() > 3:
        # Remove extra dimensions
        print(f"Reshaping st from {st.shape} to [num_samples, n_cells, n_genes]")
        st = st.squeeze()
        # If we squeezed too much, add back necessary dimensions
        if st.dim() < 3:
            if st.dim() == 2:
                # Check if first dimension is num_samples or n_cells
                if gamma is not None and st.shape[0] == gamma.shape[0]:
                    # [num_samples, n_genes] -> [num_samples, 1, n_genes]
                    st = st.unsqueeze(1)
                else:
                    # [n_cells, n_genes] -> [1, n_cells, n_genes]
                    st = st.unsqueeze(0)
            elif st.dim() == 1:
                # [n_genes] -> [1, 1, n_genes]
                st = st.unsqueeze(0).unsqueeze(0)
        print(f"Reshaped st to {st.shape}")

    # If ut/st not available, compute them from the model parameters
    if ut is None or st is None:
        # Try to compute ut and st from the model parameters
        if alpha is not None and beta is not None and gamma is not None:
            # Get cell_time from posterior samples
            cell_time = posterior_samples.get("cell_time")

            if cell_time is not None:
                # Convert numpy arrays to torch tensors if needed
                if isinstance(alpha, np.ndarray):
                    alpha = torch.tensor(alpha)
                if isinstance(beta, np.ndarray):
                    beta = torch.tensor(beta)
                if isinstance(gamma, np.ndarray):
                    gamma = torch.tensor(gamma)
                if isinstance(cell_time, np.ndarray):
                    cell_time = torch.tensor(cell_time)

                # Compute steady state values
                u_inf = alpha / beta
                s_inf = alpha / gamma

                # Compute switching time (t0 + dt_switching)
                t0 = posterior_samples.get("t0")
                dt_switching = posterior_samples.get("dt_switching")

                if t0 is not None and dt_switching is not None:
                    if isinstance(t0, np.ndarray):
                        t0 = torch.tensor(t0)
                    if isinstance(dt_switching, np.ndarray):
                        dt_switching = torch.tensor(dt_switching)
                    switching = t0 + dt_switching
                else:
                    # If t0 or dt_switching are not available, use zeros
                    switching = torch.zeros_like(u_inf)

                # Compute ut and st based on the transcription model
                # For cells before switching time
                # Reshape for broadcasting
                if u_inf.dim() == 2 and cell_time.dim() == 2:
                    # Reshape u_inf from [num_samples, n_genes] to [num_samples, 1, n_genes]
                    u_inf_reshaped = u_inf.unsqueeze(1)
                    s_inf_reshaped = s_inf.unsqueeze(1)
                    beta_reshaped = beta.unsqueeze(1)
                    gamma_reshaped = gamma.unsqueeze(1)
                    switching_reshaped = switching.unsqueeze(1)

                    # Reshape cell_time from [num_samples, n_cells] to [num_samples, n_cells, 1]
                    if cell_time.shape[1] > 1:  # If cell_time has multiple cells
                        cell_time_reshaped = cell_time.unsqueeze(2)

                        # Expand to match the number of genes
                        n_genes = u_inf.shape[1]
                        cell_time_expanded = cell_time_reshaped.expand(-1, -1, n_genes)

                        # Compute ut and st
                        ut = u_inf_reshaped * (1 - torch.exp(-beta_reshaped * cell_time_expanded))
                        st = s_inf_reshaped * (1 - torch.exp(-gamma_reshaped * cell_time_expanded)) - (
                            u_inf_reshaped * beta_reshaped / (gamma_reshaped - beta_reshaped)
                        ) * (torch.exp(-beta_reshaped * cell_time_expanded) - torch.exp(-gamma_reshaped * cell_time_expanded))
                    else:
                        # If cell_time has only one cell, we need to handle it differently
                        logger.warning(
                            "cell_time has only one cell, computing ut/st with simplified approach."
                        )
                        # Extract data from AnnData object as a fallback
                        if adata is not None:
                            u = torch.tensor(adata.layers["unspliced"].toarray() if isinstance(adata.layers["unspliced"], scipy.sparse.spmatrix) else adata.layers["unspliced"])
                            s = torch.tensor(adata.layers["spliced"].toarray() if isinstance(adata.layers["spliced"], scipy.sparse.spmatrix) else adata.layers["spliced"])
                            ut = u
                            st = s
                        else:
                            # Try to get u and s from posterior samples
                            ut = posterior_samples.get("u")
                            st = posterior_samples.get("s")
                else:
                    # If dimensions don't match, fall back to using observed data
                    logger.warning(
                        "Dimension mismatch between u_inf and cell_time, computing ut/st with simplified approach."
                    )
                    # Extract data from AnnData object as a fallback
                    if adata is not None:
                        u = torch.tensor(adata.layers["unspliced"].toarray() if isinstance(adata.layers["unspliced"], scipy.sparse.spmatrix) else adata.layers["unspliced"])
                        s = torch.tensor(adata.layers["spliced"].toarray() if isinstance(adata.layers["spliced"], scipy.sparse.spmatrix) else adata.layers["spliced"])
                        ut = u
                        st = s
                    else:
                        # Try to get u and s from posterior samples
                        ut = posterior_samples.get("u")
                        st = posterior_samples.get("s")
            else:
                # If cell_time is not available, fall back to using observed data
                logger.warning(
                    "cell_time not found in posterior_samples, computing ut/st with simplified approach."
                )
                # Extract data from AnnData object as a fallback
                if adata is not None:
                    u = torch.tensor(adata.layers["unspliced"].toarray() if isinstance(adata.layers["unspliced"], scipy.sparse.spmatrix) else adata.layers["unspliced"])
                    s = torch.tensor(adata.layers["spliced"].toarray() if isinstance(adata.layers["spliced"], scipy.sparse.spmatrix) else adata.layers["spliced"])
                    ut = u
                    st = s
                else:
                    # Try to get u and s from posterior samples
                    ut = posterior_samples.get("u")
                    st = posterior_samples.get("s")
        else:
            # If alpha, beta, or gamma are not available, fall back to using observed data
            logger.warning(
                "alpha, beta, or gamma not found in posterior_samples, computing ut/st with simplified approach."
            )
            # Extract data from AnnData object as a fallback
            if adata is not None:
                u = torch.tensor(adata.layers["unspliced"].toarray() if isinstance(adata.layers["unspliced"], scipy.sparse.spmatrix) else adata.layers["unspliced"])
                s = torch.tensor(adata.layers["spliced"].toarray() if isinstance(adata.layers["spliced"], scipy.sparse.spmatrix) else adata.layers["spliced"])
                ut = u
                st = s
            else:
                # Try to get u and s from posterior samples
                ut = posterior_samples.get("u")
                st = posterior_samples.get("s")

    if ut is None or st is None:
        raise ValueError(
            "Either adata must be provided or posterior_samples must contain ut/st or u/s"
        )

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
    if isinstance(u_scale, np.ndarray):
        u_scale = torch.tensor(u_scale)
    if isinstance(s_scale, np.ndarray):
        s_scale = torch.tensor(s_scale)
    if isinstance(ut, np.ndarray):
        ut = torch.tensor(ut)
    if isinstance(st, np.ndarray):
        st = torch.tensor(st)

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

        # Print shapes for debugging
        print(f"compute_velocity - alpha_mean shape: {alpha_mean.shape}")
        print(f"compute_velocity - beta_mean shape: {beta_mean.shape}")
        print(f"compute_velocity - ut shape: {ut.shape}")
        print(f"compute_velocity - st shape: {st.shape}")

        # Ensure parameters have the correct shapes for broadcasting
        # We need to ensure:
        # - gene parameters (beta, gamma) have shape [n_genes] or [1, n_genes]
        # - cell-gene parameters (ut, st) have shape [n_cells, n_genes]

        # Reshape gene parameters if needed
        if beta_mean.dim() == 1:  # [n_genes]
            # This is the expected shape for gene parameters
            pass
        elif beta_mean.dim() == 2:  # [num_samples, n_genes]
            # Take mean across samples
            beta_mean = beta_mean.mean(dim=0)  # [n_genes]
            gamma_mean = gamma_mean.mean(dim=0)  # [n_genes]

        # Reshape ut and st if needed
        if ut.dim() > 2:
            # If ut has extra dimensions, reshape to [n_cells, n_genes]
            if ut.dim() == 3 and ut.shape[0] == 1:  # [1, n_cells, n_genes]
                ut = ut.squeeze(0)  # [n_cells, n_genes]
                st = st.squeeze(0)  # [n_cells, n_genes]
            elif ut.dim() == 3:  # [num_samples, n_cells, n_genes]
                # Take mean across samples
                ut = ut.mean(dim=0)  # [n_cells, n_genes]
                st = st.mean(dim=0)  # [n_cells, n_genes]
            else:
                # For more complex shapes, try to reshape to [n_cells, n_genes]
                print(f"Reshaping ut from {ut.shape} to [n_cells, n_genes]")
                # Get the last two dimensions which should be cells and genes
                n_cells = ut.shape[-2] if ut.dim() > 1 else 1
                n_genes = ut.shape[-1]
                ut = ut.reshape(n_cells, n_genes)
                st = st.reshape(n_cells, n_genes)

        # Calculate scaling factor if needed
        if u_scale is not None and s_scale is not None:
            # For Gaussian models with two scales
            if u_scale.dim() > 1:
                u_scale_mean = u_scale.mean(dim=0)  # [n_cells, 1]
                s_scale_mean = s_scale.mean(dim=0)  # [n_cells, 1]
            else:
                u_scale_mean = u_scale
                s_scale_mean = s_scale

            # Calculate scale as u_scale / s_scale
            scale = u_scale_mean / s_scale_mean  # [n_cells, 1]

            # Compute velocity with proper broadcasting
            # beta: [n_genes] or [1, n_genes]
            # ut: [n_cells, n_genes]
            # scale: [n_cells, 1]
            # Result: [n_cells, n_genes]
            velocity = beta_mean * ut / scale - gamma_mean * st
        elif u_scale is not None:
            # For Poisson Model 2 with one scale
            if u_scale.dim() > 1:
                scale = u_scale.mean(dim=0)  # [n_cells, 1]
            else:
                scale = u_scale

            # Compute velocity with proper broadcasting
            velocity = beta_mean * ut / scale - gamma_mean * st
        else:
            # For Poisson Model 1 with no scale
            # Compute velocity with proper broadcasting
            velocity = beta_mean * ut - gamma_mean * st

        # Print final velocity shape
        print(f"compute_velocity - velocity shape: {velocity.shape}")

        # Compute latent time (pseudotime)
        # This is a simple implementation based on the ratio of unspliced to spliced
        # More sophisticated methods could be used

        # Handle 1D tensors
        if ut.dim() == 1:
            u_norm = ut / (ut.max() + 1e-6)
            s_norm = st / (st.max() + 1e-6)
            latent_time = torch.tensor(1.0 - u_norm.mean() / (s_norm.mean() + 1e-6))
        else:
            u_norm = ut / ut.max(dim=1, keepdim=True)[0]
            s_norm = st / st.max(dim=1, keepdim=True)[0]
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
        # For the non-mean case, we want to compute velocity for each posterior sample
        # Print shapes for debugging
        print(f"compute_velocity - alpha shape: {alpha.shape}")
        print(f"compute_velocity - beta shape: {beta.shape}")
        print(f"compute_velocity - gamma shape: {gamma.shape}")
        print(f"compute_velocity - ut shape: {ut.shape}")
        print(f"compute_velocity - st shape: {st.shape}")

        # Simplify by using the mean of posterior samples
        # This is a reasonable approach for most cases
        print("Using mean of posterior samples for velocity calculation")
        alpha_mean = alpha.mean(dim=0) if alpha.dim() > 1 else alpha
        beta_mean = beta.mean(dim=0) if beta.dim() > 1 else beta
        gamma_mean = gamma.mean(dim=0) if gamma.dim() > 1 else gamma
        u_ss_mean = alpha_mean / beta_mean
        s_ss_mean = alpha_mean / gamma_mean

        # Reshape ut and st if needed
        if ut.dim() > 2:
            # If ut has extra dimensions, reshape to [n_cells, n_genes]
            if ut.dim() == 3 and ut.shape[0] == 1:  # [1, n_cells, n_genes]
                ut = ut.squeeze(0)  # [n_cells, n_genes]
                st = st.squeeze(0)  # [n_cells, n_genes]
            elif ut.dim() == 3:  # [num_samples, n_cells, n_genes]
                # Take mean across samples
                ut = ut.mean(dim=0)  # [n_cells, n_genes]
                st = st.mean(dim=0)  # [n_cells, n_genes]
            else:
                # For more complex shapes, try to reshape to [n_cells, n_genes]
                print(f"Reshaping ut from {ut.shape} to [n_cells, n_genes]")
                # Get the last two dimensions which should be cells and genes
                n_cells = ut.shape[-2] if ut.dim() > 1 else 1
                n_genes = ut.shape[-1]
                ut = ut.reshape(n_cells, n_genes)
                st = st.reshape(n_cells, n_genes)

        # Calculate scaling factor if needed
        if u_scale is not None and s_scale is not None:
            # For Gaussian models with two scales
            if u_scale.dim() > 1:
                u_scale_mean = u_scale.mean(dim=0)  # [n_cells, 1]
                s_scale_mean = s_scale.mean(dim=0)  # [n_cells, 1]
            else:
                u_scale_mean = u_scale
                s_scale_mean = s_scale

            # Calculate scale as u_scale / s_scale
            scale = u_scale_mean / s_scale_mean  # [n_cells, 1]

            # Compute velocity with proper broadcasting
            # beta_mean: [n_genes]
            # ut: [n_cells, n_genes]
            # scale: [n_cells, 1]
            # Result: [n_cells, n_genes]
            velocity = beta_mean * ut / scale - gamma_mean * st
        elif u_scale is not None:
            # For Poisson Model 2 with one scale
            if u_scale.dim() > 1:
                scale = u_scale.mean(dim=0)  # [n_cells, 1]
            else:
                scale = u_scale

            # Compute velocity with proper broadcasting
            velocity = beta_mean * ut / scale - gamma_mean * st
        else:
            # For Poisson Model 1 with no scale
            # Compute velocity with proper broadcasting
            velocity = beta_mean * ut - gamma_mean * st

        # Print final velocity shape
        print(f"compute_velocity - velocity shape: {velocity.shape}")

        # Compute latent time (pseudotime)
        # This is a simple implementation based on the ratio of unspliced to spliced
        # More sophisticated methods could be used

        # Handle 1D tensors
        if ut.dim() == 1:
            u_norm = ut / (ut.max() + 1e-6)
            s_norm = st / (st.max() + 1e-6)
            latent_time = torch.tensor(1.0 - u_norm.mean() / (s_norm.mean() + 1e-6))
        else:
            u_norm = ut / ut.max(dim=1, keepdim=True)[0]
            s_norm = st / st.max(dim=1, keepdim=True)[0]
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
