import os
from os import PathLike

import arviz as az
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
import xarray as xr
from arviz import InferenceData
from beartype import beartype
from beartype.typing import Dict
from beartype.typing import List
from beartype.typing import Literal
from beartype.typing import Tuple
from jaxtyping import ArrayLike
from jaxtyping import Float
from jaxtyping import jaxtyped
from numpyro.infer import MCMC
from numpyro.infer import NUTS
from numpyro.infer import Predictive
from returns.result import Result
from returns.result import Success
from returns.result import safe

from pyrovelocity.logging import configure_logging
from pyrovelocity.models._deterministic_simulation import (
    solve_transcription_splicing_model,
)


__all__ = [
    "deterministic_transcription_splicing_probabilistic_model",
    "generate_test_data_for_deterministic_model_inference",
    "generate_prior_inference_data",
    "generate_posterior_inference_data",
    "save_inference_plots",
]

logger = configure_logging(__name__)

TimeTensor = Float[
    ArrayLike,
    # Cells | Timepoints per cell
    # ------|---------------------
    # 60    | 1
    "number_of_cells \
     number_of_timepoints",
]
MultiModalTranscriptomeTensor = Float[
    ArrayLike,
    # Genes | Cells | Timepoints per cell| Modalities
    # ------|-------|--------------------|-------------------
    # 1     | 60    | 1                  | 2
    "number_of_genes \
     number_of_cells \
     number_of_timepoints \
     number_of_modalities",
]


@jaxtyped(typechecker=beartype)
def deterministic_transcription_splicing_probabilistic_model(
    times: TimeTensor,
    data: MultiModalTranscriptomeTensor,
):
    num_genes, num_cells, num_timepoints, num_modalities = data.shape

    # priors
    initial_conditions = numpyro.sample(
        "initial_conditions",
        dist.LogNormal(loc=jnp.log(0.23), scale=0.9),
        sample_shape=(num_genes, num_modalities),
    )
    gamma = numpyro.sample(
        "gamma",
        dist.LogNormal(loc=jnp.log(1.13), scale=0.35),
        sample_shape=(num_genes,),
    )

    def model_solver(gene_index, cell_index):
        init_cond = initial_conditions[gene_index]
        rate = gamma[gene_index]
        t = times[cell_index, :]
        solution = solve_transcription_splicing_model(
            t,
            init_cond,
            (rate,),
        )
        return solution.ys

    # predictions = jax.vmap(
    #     jax.vmap(
    #         model_solver,
    #         in_axes=(0, None),
    #     ),
    #     in_axes=(None, 0),
    # )(
    #     jnp.arange(num_genes),
    #     jnp.arange(num_cells),
    # )
    predictions = jax.vmap(
        lambda g: jax.vmap(
            lambda c: model_solver(g, c),
            in_axes=0,
        )(jnp.arange(num_cells)),
        in_axes=0,
    )(jnp.arange(num_genes))

    logger.info(
        f"\nPredictions Shape: {predictions.shape}\n"
        f"Data Shape: {data.shape}\n\n"
    )

    sigma = numpyro.sample(
        "sigma",
        dist.HalfNormal(scale=1.0),
        sample_shape=(num_modalities,),
    )
    sigma_expanded = sigma.reshape(1, 1, 1, num_modalities)

    numpyro.sample(
        "observations",
        dist.Normal(
            predictions,
            sigma_expanded,
        ),
        obs=data,
    )


@beartype
def generate_test_data_for_deterministic_model_inference(
    num_genes: int,
    num_cells: int,
    num_timepoints: int,
    num_modalities: int,
    log_time_start: float = -1,
    log_time_end: float = 1,
    noise_levels: Tuple[float, float] = (0.05, 0.05),
    initial_conditions_params: Tuple[float, float] = (0.23, 0.9),
    gamma_params: Tuple[float, float] = (1.13, 0.35),
) -> Tuple[
    TimeTensor,
    MultiModalTranscriptomeTensor,
    int,
    int,
    int,
    int,
]:
    """
    Generate synthetic data for testing inference in the deterministic
    transcription-splicing model.

    Args:
        num_cells (Integer): Number of cells
        num_genes (Integer): Number of genes
        num_timepoints (Integer): Number of timepoints
        num_modalities (Integer): Number of modalities
        log_time_start (float): Log-space start for time points.
        log_time_end (float): Log-space end for time points.
        noise_levels (Tuple[float, float]):
            Standard deviations for observational noise on each modality.
        initial_conditions_params (Tuple[float, float]):
            Parameters (mean, std) for log-normal initial conditions.
        gamma_params (Tuple[float, float]):
            Parameters (mean, std) for log-normal gamma distribution.


    Returns:
        Tuple[
            TimeTensor,
            MultiModalTranscriptomeTensor,
            Integer,
            Integer,
            Integer,
        ]: Tuple containing the time array, data tensor, and the number of
            cells, genes, and timepoints.

    Example:
        >>> # xdoctest: +SKIP
        >>> times, data, num_cells, num_genes, num_timepoints, num_modalities = (
        >>> generate_test_data_for_deterministic_model_inference(
        ...      num_genes=1,
        ...      num_cells=3,
        ...      num_timepoints=4,
        ...      num_modalities=2,
        >>> )
    """
    rng_key = jax.random.PRNGKey(0)
    rng_key, rng_key_ = jax.random.split(rng_key)

    # simulate timepoints for each cell to represent observational sampling
    total_timepoints = num_timepoints * num_cells
    all_times = jnp.logspace(
        log_time_start,
        log_time_end,
        total_timepoints,
    )
    time_indices = jax.random.choice(
        rng_key_,
        total_timepoints,
        shape=(num_cells, num_timepoints),
        replace=False,
    )
    time_indices = jnp.sort(time_indices, axis=1)
    row_order = jnp.argsort(time_indices[:, 0])
    # Each row is sorted
    # The rows are sorted by the first column
    # Array([[ 0,  2,  4, 10],
    #        [ 1,  5,  8, 11],
    #        [ 3,  6,  7,  9]], dtype=int32)
    time_indices = time_indices[row_order]
    times = all_times[time_indices]

    # Flatten times and sort them globally while retaining
    # the original order associated to the observation times
    # of each cell
    flat_times = times.flatten()
    sorted_indices = jnp.argsort(flat_times)
    sorted_times = flat_times[sorted_indices]

    initial_conditions = numpyro.sample(
        "initial_conditions",
        dist.LogNormal(
            loc=jnp.log(initial_conditions_params[0]),
            scale=initial_conditions_params[1],
        ),
        sample_shape=(num_genes, num_modalities),
        rng_key=rng_key_,
    )
    gamma = numpyro.sample(
        "gamma",
        dist.LogNormal(
            loc=jnp.log(gamma_params[0]),
            scale=gamma_params[1],
        ),
        sample_shape=(num_genes,),
        rng_key=rng_key_,
    )

    def model_solver(ts, init_cond, rate):
        solution = solve_transcription_splicing_model(
            ts=ts.reshape(-1),
            initial_state=init_cond,
            params=(rate,),
        ).ys

        return solution[sorted_indices.argsort()].reshape(
            num_cells, num_timepoints, num_modalities
        )

    solutions = jax.vmap(
        model_solver,
        in_axes=(None, 0, 0),
    )(
        sorted_times,
        initial_conditions,
        gamma,
    )

    noisy_solutions = solutions + jax.random.normal(
        rng_key_, solutions.shape
    ) * jnp.array(noise_levels)

    logger.info(f"Generated test data tensor shape: {noisy_solutions.shape}")
    logger.info(f"Generated test time array shape: {times.shape}")

    return (
        times,
        noisy_solutions,
        num_genes,
        num_cells,
        num_timepoints,
        num_modalities,
    )


@beartype
def create_inference_data_labels(
    num_genes: int,
    num_cells: int,
    num_timepoints: int,
    modalities: List[str],
) -> Tuple[
    Dict[str, ArrayLike],
    Dict[str, List[str]],
]:
    """
    Creates labels for dimensions and coordinates for ArviZ InferenceData
    structures used here in probabilistic inference reporting with `numpyro` and
    `arviz`.

    Args:
    num_genes (int): Number of genes.
    num_cells (int): Number of cells.
    num_timepoints (int): Number of timepoints.
    modalities (List[str]): List of modalities (e.g., ["pre-mRNA", "mRNA"]).

    Returns:
    Tuple[Dict[str, np.ndarray], Dict[str, List[str]]]:
        A tuple containing two dictionaries, one for coordinates and one for
        dimensions suitable for ArviZ InferenceData construction.
    """
    idata_coords = {
        "genes": np.arange(num_genes),
        "cells": np.arange(num_cells),
        "timepoints": np.arange(num_timepoints),
        "modalities": modalities,
    }

    idata_dims = {
        "initial_conditions": ["genes", "modalities"],
        "gamma": ["genes"],
        "sigma": ["modalities"],
        "observations": ["genes", "cells", "timepoints", "modalities"],
    }

    return idata_coords, idata_dims


@beartype
def generate_prior_inference_data(
    times: TimeTensor,
    data: MultiModalTranscriptomeTensor,
    num_chains: int,
    num_samples: int,
    num_genes: int,
    num_cells: int,
    num_timepoints: int,
    num_modalities: int,
) -> InferenceData:
    """
    Generate samples from the prior predictive distribution based on the
    deterministic transcription-splicing model.

    Args:
        times (TimeTensor):
            Time points for each cell, shape (num_cells, num_timepoints).
        data (MultiModalTranscriptomeTensor):
            Actual observed data to include for comparison.
        num_samples (int): Number of prior samples to generate.
        num_cells (int): Number of cells.
        num_genes (int): Number of genes.
        num_timepoints (int): Number of timepoints per cell.
        num_modalities (int):
            Number of modalities (e.g., types of measurements).

    Returns:
        InferenceData:
            An ArviZ InferenceData object containing the generated prior samples
            and the actual observed data for comparison.
    """
    model = deterministic_transcription_splicing_probabilistic_model
    rng_key = jax.random.PRNGKey(0)

    predictive = Predictive(
        model,
        num_samples=num_chains * num_samples,
        batch_ndims=1,
        parallel=False,
    )
    prior_predictions = predictive(rng_key, times=times, data=data)

    modality_labels = ["pre-mRNA", "mRNA"]
    assert len(modality_labels) == num_modalities

    coords, dims = create_inference_data_labels(
        num_genes=num_genes,
        num_cells=num_cells,
        num_timepoints=num_timepoints,
        modalities=modality_labels,
    )

    idata = az.from_numpyro(
        posterior=None,
        prior=prior_predictions,
        coords=coords,
        dims=dims,
        posterior_predictive={
            "observations": prior_predictions["observations"]
        },
    )

    observed_data = xr.Dataset(
        {
            "observations": (
                ["genes", "cells", "timepoints", "modalities"],
                data,
            ),
        }
    )

    idata.add_groups({"observed_data": observed_data})

    return idata


def generate_posterior_inference_data(
    times: TimeTensor,
    data: MultiModalTranscriptomeTensor,
    num_chains: int,
    num_samples: int,
    num_genes: int,
    num_cells: int,
    num_timepoints: int,
    num_modalities: int,
    num_warmup: int = 500,
) -> InferenceData:
    model = deterministic_transcription_splicing_probabilistic_model
    rng_key = jax.random.PRNGKey(0)

    prior_predictive = Predictive(
        model,
        num_samples=num_chains * num_samples,
        batch_ndims=1,
        parallel=False,
    )
    prior_predictions = prior_predictive(rng_key, times=times, data=data)

    kernel = NUTS(model)
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=True,
    )

    mcmc.run(rng_key, times=times, data=data)
    mcmc.print_summary()

    posterior_samples = mcmc.get_samples()
    posterior_predictive = Predictive(model, posterior_samples)
    rng_key, _ = jax.random.split(rng_key)
    posterior_predictions = posterior_predictive(
        rng_key, times=times, data=data
    )

    modality_labels = ["pre-mRNA", "mRNA"]
    assert len(modality_labels) == num_modalities

    coords, dims = create_inference_data_labels(
        num_genes=num_genes,
        num_cells=num_cells,
        num_timepoints=num_timepoints,
        modalities=modality_labels,
    )
    idata = az.from_numpyro(
        posterior=mcmc,
        prior=prior_predictions,
        posterior_predictive=posterior_predictions,
        coords=coords,
        dims=dims,
    )

    return idata


@beartype
def save_inference_plots(
    idata_prior: InferenceData,
    idata_posterior: InferenceData,
    output_dir: PathLike | str,
) -> Result[Literal[True], Exception]:
    """
    Generate and save plots for both prior and posterior inference data.

    Args:
    idata_prior (arviz.InferenceData): Inference data from prior predictive checks.
    idata_posterior (arviz.InferenceData): Inference data from posterior predictive checks.
    output_dir (str): Directory path where plots will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)

    light_gray = "#bcbcbc"

    with plt.style.context(["pyrovelocity.styles.common"]):
        az.plot_ppc(idata_posterior, group="prior")
        plt.savefig(f"{output_dir}/prior_predictive_checks.png")
        plt.savefig(f"{output_dir}/prior_predictive_checks.pdf")
        plt.close()

        az.plot_ppc(idata_posterior, group="posterior")
        plt.savefig(f"{output_dir}/posterior_predictive_checks.png")
        plt.savefig(f"{output_dir}/posterior_predictive_checks.pdf")
        plt.close()

        variables = ["initial_conditions", "gamma", "sigma"]

        for var in variables:
            az.plot_posterior(
                idata_prior,
                var_names=[var],
                group="prior",
                kind="hist",
                color=light_gray,
                round_to=2,
            )
            plt.savefig(f"{output_dir}/prior_{var}.png")
            plt.savefig(f"{output_dir}/prior_{var}.pdf")
            plt.close()

            az.plot_posterior(
                idata_posterior,
                var_names=[var],
                group="posterior",
                kind="hist",
                color=light_gray,
                round_to=2,
            )
            plt.savefig(f"{output_dir}/posterior_{var}.png")
            plt.savefig(f"{output_dir}/posterior_{var}.pdf")
            plt.close()

            az.plot_forest(idata_posterior, var_names=[var])
            plt.savefig(f"{output_dir}/forest_{var}.png")
            plt.savefig(f"{output_dir}/forest_{var}.pdf")
            plt.close()

        az.plot_trace(
            idata_posterior,
            rug=True,
        )
        plt.savefig(f"{output_dir}/trace_plots.png")
        plt.savefig(f"{output_dir}/trace_plots.pdf")
        plt.close()

    return Success(True)
