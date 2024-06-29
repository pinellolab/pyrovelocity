from __future__ import annotations

import shutil
from os import PathLike
from pathlib import Path

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
from beartype.typing import (
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
)
from einops import rearrange
from jaxtyping import ArrayLike, Float, jaxtyped
from matplotlib import colors
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from matplotlib.figure import Figure
from numpyro.infer import MCMC, NUTS, Predictive
from xarray import DataArray  # type-checking

from pyrovelocity.logging import configure_logging
from pyrovelocity.models._deterministic_simulation import (
    solve_transcription_splicing_model,
)

__all__ = [
    "deterministic_transcription_splicing_probabilistic_model",
    "generate_test_data_for_deterministic_model_inference",
    "generate_prior_inference_data",
    "generate_posterior_inference_data",
    "plot_sample_phase_portraits",
    "plot_sample_trajectories",
    "plot_sample_trajectories_with_percentiles",
    "save_inference_plots",
]

logger = configure_logging(__name__)

# Timepoints refers to timepoints per cell
TimeTensor = Float[
    ArrayLike,
    # Cells | Timepoints |
    # ------|------------|
    # 3     | 4          |
    # 10    | 1          |
    "number_of_cells \
     number_of_timepoints",
]
MultiModalTranscriptomeTensor = Float[
    ArrayLike,
    # Genes | Cells | Timepoints | Modalities |
    # ------|-------|------------|------------|
    # 2     | 3     | 4          | 2          |
    # 1     | 10    | 1          | 2          |
    "number_of_genes \
     number_of_cells \
     number_of_timepoints \
     number_of_modalities",
]


@beartype
def solve_model_for_each_gene(
    initial_conditions: jnp.ndarray,
    gamma: jnp.ndarray,
    times: jnp.ndarray,
    num_genes: int,
) -> ArrayLike:
    """
    Solve the deterministic transcription-splicing model for all genes across
    a unified time vector.

    Args:
        initial_conditions (jnp.ndarray):
            Initial conditions for each gene and modality.
        gamma (jnp.ndarray): Decay rates for each gene.
        times (jnp.ndarray): Unified time points for the model.
        num_genes (int): Total number of genes.

    Returns:
        jnp.ndarray:
            The model's predictions across all genes and the unified timepoints.
    """

    def model_solver(gene_index: int) -> jnp.ndarray:
        init_cond = initial_conditions[gene_index]
        rate = gamma[gene_index]
        return solve_transcription_splicing_model(
            times,
            init_cond,
            (rate,),
        ).ys

    return jax.vmap(model_solver)(jnp.arange(num_genes))


@beartype
def sort_times_over_all_cells(
    times: TimeTensor,
) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
    """
    Generates a unified sorted time vector from the provided times matrix.

    If the times were not unique, it may be necessary to use jnp.unique;

    all_times = jnp.unique(flat_times[sorted_indices])

    however, we assume that the times are unique here.

    Args:
        times (jnp.ndarray): Time matrix with shape (num_cells, num_timepoints).

    Returns:
        Tuple of:
            - sorted_flat_times (jnp.ndarray): Unified sorted time vector.
            - sorted_to_original_indices (jnp.ndarray):
                Indices mapping the positions from the sorted times back to
                their original positions in the matrix.
            - original_to_sorted_indices (jnp.ndarray):
                Permutation indices used to sort the original times into
                sorted_flat_times.
    """
    flat_times = times.flatten()
    original_to_sorted_indices = jnp.argsort(flat_times)
    sorted_flat_times = flat_times[original_to_sorted_indices]

    sorted_to_original_indices = jnp.searchsorted(sorted_flat_times, times)

    return (
        sorted_flat_times,
        sorted_to_original_indices,
        original_to_sorted_indices,
    )


@jaxtyped(typechecker=beartype)
def deterministic_transcription_splicing_probabilistic_model(
    times: TimeTensor,
    data: MultiModalTranscriptomeTensor,
    data_observation_flag: bool = True,
):
    num_genes, num_cells, num_timepoints, num_modalities = data.shape

    all_times, time_indices, _ = sort_times_over_all_cells(times)

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

    # This could be a deterministic site,
    #
    # numpyro.deterministic("times", times)
    #
    # but then it would not be recorded with
    # is_observed=True in the posterior samples.
    # This property is used by ArviZ to extract
    # observed_data.
    #
    numpyro.sample(
        "times",
        dist.Delta(times),
        obs=times,
    )
    #
    # When times becomes a latent random variable
    # as opposed to an observed deterministic one,
    # then it can be sampled, e.g.,
    #
    # times = numpyro.sample(
    #     "times",
    #     dist.Uniform(low=0, high=1),
    #     sample_shape=(num_cells, num_timepoints),
    # )

    predictions = solve_model_for_each_gene(
        initial_conditions=initial_conditions,
        gamma=gamma,
        times=all_times,
        num_genes=num_genes,
    )

    predictions_reordered = predictions[:, time_indices.ravel()]

    predictions_rearranged = rearrange(
        predictions_reordered,
        "genes (cells timepoints) modalities -> genes cells timepoints modalities",
        cells=num_cells,
        timepoints=num_timepoints,
        genes=num_genes,
    )

    logger.debug(
        f"\nPredictions shape: {predictions.shape}\n"
        f"\nReordered predictions shape: {predictions_reordered.shape}\n"
        f"\nRearranged predictions shape: {predictions_rearranged.shape}\n"
        f"Data Shape: {data.shape}\n\n"
    )

    # likelihood
    sigma = numpyro.sample(
        "sigma",
        dist.HalfNormal(scale=0.1),
        sample_shape=(num_modalities,),
    )
    sigma_expanded = sigma.reshape(1, 1, 1, num_modalities)

    log_predictions = jnp.log(predictions_rearranged)
    numpyro.sample(
        "observations",
        dist.LogNormal(
            loc=log_predictions,
            scale=sigma_expanded,
        ),
        obs=data if data_observation_flag else None,
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
    initial_conditions_params: Tuple[float, float] = (0.1, 0.1),
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
    # The dimensions are 'cells timepoints' or
    # 'num_cells num_timepoints'
    # which is '3 4' in the following example
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

    noise = jax.random.normal(rng_key_, solutions.shape) * (
        solutions * jnp.array(noise_levels).reshape(1, 1, 1, num_modalities)
    )
    noisy_solutions = solutions + noise

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
    times: TimeTensor,
    modalities: List[str],
) -> Tuple[
    Dict[str, ArrayLike],
    Dict[str, List[str]],
]:
    """
    Creates labels for dimensions and coordinates for ArviZ InferenceData
    structures used here in probabilistic inference reporting with `numpyro` and
    `arviz`.

    Note that `timepoints` represents the number of timepoints per cell, and
    `times` represents the values of the observed timepoints for each cell.

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
        "initial_conditions": [
            "genes",
            "modalities",
        ],
        "gamma": ["genes"],
        "sigma": ["modalities"],
        "times": [
            "cells",
            "timepoints",
        ],
        "observations": [
            "genes",
            "cells",
            "timepoints",
            "modalities",
        ],
    }

    return idata_coords, idata_dims


@beartype
def generate_predictive_samples(
    model: Callable,
    times: TimeTensor,
    data: MultiModalTranscriptomeTensor,
    num_chains: int = 1,
    num_samples: Optional[int] = None,
    posterior_samples: Optional[Dict[str, ArrayLike]] = None,
    data_observation_flag: bool = False,
    rng_key: ArrayLike = jax.random.PRNGKey(0),
) -> Dict[str, ArrayLike]:
    """
    Creates a predictive model to generate predictive samples.

    The value of the num_samples argument used internal to
    `numpyro.infer.Predictive` will be inferred from the shape of the
    posterior_samples and this will override the num_samples argument.

    Args:
        model (Callable): The model function to use for predictions.
        times (TimeTensor):
            Time points for each cell, used as an input to the model.
        data (MultiModalTranscriptomeTensor):
            The data tensor to compare against the model.
        num_chains (int): Number of MCMC chains to run.
        num_samples (Optional[int]): Number of samples to generate per chain.
        posterior_samples (Optional[Dict[str, ArrayLike]]):
            Samples from the posterior distribution.
        data_observation_flag (bool):
            Flag to determine if data should be observed.
        rng_key (ArrayLike): Jax pseudo-random number generator key.

    Returns:
        Dict[str, ArrayLike]: A dictionary of predictions from the model.
    """
    rng_key, rng_key_ = jax.random.split(rng_key)

    predictive = Predictive(
        model=model,
        posterior_samples=posterior_samples,
        num_samples=num_chains * num_samples if num_samples else None,
        batch_ndims=1,
        parallel=False,
    )

    predictions = predictive(
        rng_key_,
        times=times,
        data=data,
        data_observation_flag=data_observation_flag,
    )

    return predictions


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

    prior_predictions = generate_predictive_samples(
        model=model,
        times=times,
        data=data,
        num_chains=num_chains,
        num_samples=num_samples,
        data_observation_flag=False,
        rng_key=rng_key,
    )

    modality_labels = ["pre-mRNA", "mRNA"]
    assert len(modality_labels) == num_modalities

    coords, dims = create_inference_data_labels(
        num_genes=num_genes,
        num_cells=num_cells,
        num_timepoints=num_timepoints,
        times=times,
        modalities=modality_labels,
    )

    idata = az.from_numpyro(
        posterior=None,
        prior=prior_predictions,
        coords=coords,
        dims=dims,
        posterior_predictive={
            "observations": prior_predictions["observations"],
            "times": prior_predictions["times"],
        },
    )

    observed_data = xr.Dataset(
        {
            "times": (
                ["cells", "timepoints"],
                times,
            ),
            "observations": (
                ["genes", "cells", "timepoints", "modalities"],
                data,
            ),
        }
    )

    idata.add_groups({"observed_data": observed_data})

    structure_description = print_inference_data_structure(idata)
    logger.info(f"\nPrior Inference Data\n\n" + structure_description)

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

    prior_predictions = generate_predictive_samples(
        model=model,
        times=times,
        data=data,
        num_chains=num_chains,
        num_samples=num_samples,
        data_observation_flag=False,
        rng_key=rng_key,
    )

    kernel = NUTS(
        model,
        init_strategy=numpyro.infer.init_to_feasible,
    )
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=True,
    )

    rng_key, rng_key_ = jax.random.split(rng_key)
    mcmc.run(rng_key_, times=times, data=data)
    mcmc.print_summary()

    posterior_samples = mcmc.get_samples(group_by_chain=False)

    posterior_predictions = generate_predictive_samples(
        model=model,
        times=times,
        data=data,
        num_chains=num_chains,
        num_samples=None,
        posterior_samples=posterior_samples,
        data_observation_flag=False,
        rng_key=rng_key,
    )

    modality_labels = ["pre-mRNA", "mRNA"]
    assert len(modality_labels) == num_modalities

    coords, dims = create_inference_data_labels(
        num_genes=num_genes,
        num_cells=num_cells,
        num_timepoints=num_timepoints,
        times=times,
        modalities=modality_labels,
    )
    idata = az.from_numpyro(
        posterior=mcmc,
        prior=prior_predictions,
        posterior_predictive=posterior_predictions,
        coords=coords,
        dims=dims,
    )

    structure_description = print_inference_data_structure(idata)
    logger.info(f"\nPosterior Inference Data\n\n" + structure_description)

    return idata


@beartype
def save_figure(
    name: str,
    output_dir: PathLike,
):
    for ext in ["png", "pdf"]:
        plt.savefig(output_dir / f"{name}.{ext}")
    plt.close()


@beartype
def save_figure_object(
    fig: Figure,
    name: str,
    output_dir: PathLike,
):
    """
    Saves a matplotlib figure to disk.

    Args:
        fig (Figure): The figure to save.
        name (str): Base filename to use for saving the figure.
        output_dir (PathLike): Directory path where the figure will be saved.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "pdf"]:
        fig.savefig(output_dir / f"{name}.{ext}")
    plt.close(fig)


@beartype
def save_inference_plots(
    idata_posterior: InferenceData,
    output_dir: PathLike | str,
) -> bool:
    """
    Generate and save plots for both prior and posterior inference data.

    Args:
        idata_posterior (arviz.InferenceData):
            Inference data from posterior predictive checks.
        output_dir (str): Directory path where plots will be saved.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    light_gray = "#bcbcbc"

    with plt.style.context("pyrovelocity.styles.common"):
        if not shutil.which("latex"):
            plt.rc("text", usetex=False)

        az.plot_ppc(idata_posterior, group="prior")
        save_figure(
            name="prior_predictive_checks",
            output_dir=output_dir,
        )

        az.plot_ppc(idata_posterior, group="posterior")
        save_figure(
            name="posterior_predictive_checks",
            output_dir=output_dir,
        )

        variables = ["initial_conditions", "gamma", "sigma"]
        for var in variables:
            az.plot_posterior(
                idata_posterior,
                var_names=[var],
                group="prior",
                kind="hist",
                color=light_gray,
                round_to=2,
            )
            save_figure(
                name=f"prior_{var}",
                output_dir=output_dir,
            )

            az.plot_posterior(
                idata_posterior,
                var_names=[var],
                group="posterior",
                kind="hist",
                color=light_gray,
                round_to=2,
            )
            save_figure(
                name=f"posterior_{var}",
                output_dir=output_dir,
            )

            az.plot_forest(idata_posterior, var_names=[var])
            save_figure(
                name=f"forest_{var}",
                output_dir=output_dir,
            )

        az.plot_trace(idata_posterior, rug=True)
        save_figure(
            name="trace_plots",
            output_dir=output_dir,
        )

        figs = plot_sample_trajectories(idata_posterior)
        for idx, fig in enumerate(figs):
            save_figure_object(
                fig=fig,
                name=f"sample_trajectories_{idx}",
                output_dir=output_dir,
            )

        figs = plot_sample_trajectories_with_percentiles(idata_posterior)
        for idx, fig in enumerate(figs):
            save_figure_object(
                fig=fig,
                name=f"sample_trajectories_percentiles_{idx}",
                output_dir=output_dir,
            )

        figs = plot_sample_phase_portraits(
            idata_posterior,
            colormap_name="RdBu",
        )
        for idx, fig in enumerate(figs):
            save_figure_object(
                fig=fig,
                name=f"sample_phase_portraits_{idx}",
                output_dir=output_dir,
            )

    return True


@beartype
def plot_sample_phase_portraits(
    idata: InferenceData,
    trajectories_index: int | slice = slice(None),
    colormap_name: str = "cividis",
) -> List[Figure]:
    """
    Plots the phase portrait of mRNA (x-axis) vs pre-mRNA (y-axis) for all genes.

    Args:
        idata (InferenceData):
            The posterior predictive data containing simulations.
        trajectories_index (int | slice):
            Index for the specific trajectories to plot.
        colormap_name (str):
            Name of the matplotlib color map to use for the time color bar.

    Returns:
        List[Figure]:
            A list of matplotlib Figure objects containing the plots.
    """
    figs = []
    genes = idata.observed_data.observations.coords["genes"].values

    for gene_index in genes:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        observed_data = idata.observed_data.observations.sel(genes=gene_index)
        sample_data = idata.posterior_predictive.observations.sel(
            chain=0, draw=trajectories_index, genes=gene_index
        )

        observed_times = idata.observed_data.times.values

        (
            sorted_flat_times,
            sorted_to_original_indices,
            original_to_sorted_indices,
        ) = sort_times_over_all_cells(jnp.array(observed_times))

        mRNA_obs = rearrange(
            observed_data.sel(modalities="mRNA").values, "c t -> (c t)"
        )[original_to_sorted_indices.ravel()]
        pre_mRNA_obs = rearrange(
            observed_data.sel(modalities="pre-mRNA").values, "c t -> (c t)"
        )[original_to_sorted_indices.ravel()]

        for ax, scale in zip(axes, ["linear", "log"]):
            if scale == "log":
                ax.set_xscale("log")
                ax.set_yscale("log")
                colorbar_norm = colors.LogNorm(
                    vmin=sorted_flat_times.min(), vmax=sorted_flat_times.max()
                )
            else:
                colorbar_norm = colors.Normalize(
                    vmin=sorted_flat_times.min(), vmax=sorted_flat_times.max()
                )

            sc = ax.scatter(
                mRNA_obs,
                pre_mRNA_obs,
                c=sorted_flat_times,
                cmap=colormap_name,
                norm=colorbar_norm,
                label="Observed",
                marker=".",
                s=12**2,
                edgecolor="none",
                alpha=1,
            )
            ax.plot(
                mRNA_obs,
                pre_mRNA_obs,
                "k-",
                label="Observed Trajectory",
                alpha=0.6,
            )

            for draw_index in range(sample_data.sizes["draw"]):
                mRNA_samples = rearrange(
                    sample_data.isel(draw=draw_index)
                    .sel(modalities="mRNA")
                    .values,
                    "c t -> (c t)",
                )[original_to_sorted_indices.ravel()]
                pre_mRNA_samples = rearrange(
                    sample_data.isel(draw=draw_index)
                    .sel(modalities="pre-mRNA")
                    .values,
                    "c t -> (c t)",
                )[original_to_sorted_indices.ravel()]

                ax.scatter(
                    mRNA_samples,
                    pre_mRNA_samples,
                    c=sorted_flat_times,
                    cmap=colormap_name,
                    norm=colorbar_norm,
                    label="Sampled" if draw_index == 0 else "_nolegend_",
                    alpha=0.3,
                    marker="2",
                    s=6**2,
                    edgecolor="none",
                )
                ax.plot(
                    mRNA_samples,
                    pre_mRNA_samples,
                    "gray",
                    label="Sampled Trajectory"
                    if draw_index == 0
                    else "_nolegend_",
                    alpha=0.3,
                )

            ax.set_xlabel("mRNA Expression")
            ax.set_ylabel("pre-mRNA Expression")
            ax.set_title(
                f"Gene {gene_index} - Phase Portrait ({scale.capitalize()} Scale)"
            )

            plt.colorbar(sc, ax=ax, label="Time")

        plt.tight_layout()
        figs.append(fig)
        plt.show()

    return figs


@beartype
def plot_trajectory_with_percentiles(
    ax: Axes,
    times: ArrayLike,
    data: ArrayLike,
    sample_data: ArrayLike,
    modality: str,
    modality_color: str,
    percentile_interval_widths: List[float],
    color_map_indices: ArrayLike,
    percentiles_color_maps: Dict[str, Colormap],
    samples_transparency: float,
    label: str,
    cell_index: int = 0,
):
    """
    Helper function to plot trajectories and percentile bands.
    """
    ax.plot(
        times,
        data,
        label=label,
        color=modality_color,
        marker=".",
        ms=12,
    )

    for i, percentile in enumerate(percentile_interval_widths):
        color = percentiles_color_maps[modality](color_map_indices[i])
        pct_lower = np.percentile(sample_data, 50 - percentile / 2, axis=0)
        pct_upper = np.percentile(sample_data, 50 + percentile / 2, axis=0)
        poly_label = (
            f"{round(50-percentile/2)} - {round(50+percentile/2)}%"
            if cell_index == 0
            else "_nolegend_"
        )
        ax.fill_between(
            times,
            pct_lower,
            pct_upper,
            color=color,
            alpha=samples_transparency,
            label=poly_label,
        )


@beartype
def plot_sample_trajectories_with_percentiles(
    idata: InferenceData,
    trajectories_index: int | slice = slice(None),
    num_percentile_intervals: int = 5,
    modality_line_colors: Dict[str, str] = {
        "pre-mRNA": "gray",
        "mRNA": "green",
    },
    percentiles_color_maps: Dict[str, Colormap] = {
        "pre-mRNA": plt.cm.Greys,
        "mRNA": plt.cm.Greens,
    },
    samples_transparency: float = 0.2,
) -> List[Figure]:
    """
    Plots sample trajectories over time for all genes, modality, and cell
    combinations, including percentile bands.

    Args:
        idata (InferenceData):
            The posterior predictive data containing simulations.
        trajectories_index (int | slice):
            Index for the specific trajectories to plot.
        num_percentile_intervals (int):
            Number of percentile intervals to plot.
        modality_line_colors (Dict[str, str]):
            Color mapping for modalities.

    Returns:
        List[Figure]:
            A list of matplotlib Figure objects containing the plots.
    """
    figs = []
    genes = idata.observed_data.observations.coords["genes"].values
    percentile_interval_widths = list(
        np.linspace(
            start=0,
            stop=100,
            num=2 + num_percentile_intervals,
        )
    )[1:-1][::-1]
    color_map_indices = np.linspace(0.2, 1, num_percentile_intervals)

    for gene_index in genes:
        fig, ax = plt.subplots(figsize=(12, 8))

        observed_data = idata.observed_data.observations.sel(
            genes=gene_index
        ).values
        observed_times = idata.observed_data.times.values

        sample_data = idata.posterior_predictive.observations.sel(
            chain=0, draw=trajectories_index, genes=gene_index
        ).values

        observed_times_shape = observed_times.shape
        if observed_times_shape[1] == 1:
            observed_times = rearrange(observed_times, "c t -> (c t)")
            observed_data = rearrange(observed_data, "c t m -> (c t) m")
            sample_data = rearrange(sample_data, "s c t m -> s (c t) m")

        for modality_index, modality in enumerate(
            idata.observed_data.observations.coords["modalities"].values
        ):
            modality_color = modality_line_colors.get(modality, "blue")
            if observed_times_shape[1] > 1:
                for cell_index in range(observed_times.shape[0]):
                    plot_trajectory_with_percentiles(
                        ax=ax,
                        times=observed_times[cell_index],
                        data=observed_data[cell_index, ..., modality_index],
                        sample_data=sample_data[
                            :, cell_index, ..., modality_index
                        ],
                        modality=modality,
                        modality_color=modality_color,
                        percentile_interval_widths=percentile_interval_widths,
                        color_map_indices=color_map_indices,
                        percentiles_color_maps=percentiles_color_maps,
                        samples_transparency=samples_transparency,
                        label=f"{modality}"
                        if cell_index == 0
                        else "_nolegend_",
                        cell_index=cell_index,
                    )
            else:
                plot_trajectory_with_percentiles(
                    ax=ax,
                    times=observed_times,
                    data=observed_data[..., modality_index],
                    sample_data=sample_data[..., modality_index],
                    modality=modality,
                    modality_color=modality_color,
                    percentile_interval_widths=percentile_interval_widths,
                    color_map_indices=color_map_indices,
                    percentiles_color_maps=percentiles_color_maps,
                    samples_transparency=samples_transparency,
                    label=modality,
                )

        ax.set_xlabel("Time")
        ax.set_ylabel("Expression")
        ax.set_title(
            f"Gene {gene_index} - Observed and Predicted Trajectories with Percentiles"
        )
        ax.legend()
        ax.grid(True)
        figs.append(fig)

    return figs


"""
Convert interfaces to use Sum type following upgrade to
python >=3.11 and expression >=5

from expression import case, tag, tagged_union

@tagged_union
class InferenceStage:
    tag: Literal["posterior", "prior"] = tag()

    posterior: Literal[True] = case()
    prior: Literal[True] = case()

    @staticmethod
    def Posterior() -> InferenceStage:
        return InferenceStage(posterior=True)

    @staticmethod
    def Prior() -> InferenceStage:
        return InferenceStage(prior=True)


@beartype
def plot_sample_trajectories(
    idata: InferenceData,
    inference_stage: InferenceStage = InferenceStage.Posterior(),
    trajectories_index: int | slice = slice(None),
    num_trajectories: int = 100,
    color_map: Dict[str, str] = {"pre-mRNA": "gray", "mRNA": "green"},
    samples_transparency: float = 0.3,
) -> List[Figure]:
    ...
    genes = idata.observed_data.observations.coords["genes"].values

    for gene_index in genes:
        ...

        match inference_stage:
            case InferenceStage(tag="posterior"):
                sample_data = idata.posterior_predictive.observations.sel(
                    chain=0, draw=trajectories_index, genes=gene_index
                ).values
            case InferenceStage(tag="prior"):
                sample_data = idata.prior_predictive.observations.sel(
                    chain=0, draw=trajectories_index, genes=gene_index
            7    ).values

    # inference_stage: InferenceStage = InferenceStage.Posterior(),
"""


@beartype
def plot_sample_trajectories(
    idata: InferenceData,
    trajectories_index: int | slice = slice(None),
    num_trajectories: int = 100,
    color_map: Dict[str, str] = {"pre-mRNA": "gray", "mRNA": "green"},
    samples_transparency: float = 0.3,
) -> List[Figure]:
    """
    Plots sample trajectories over time for all genes, modality, and cell
    combinations.

    Args:
        idata (InferenceData):
            The inference data object containing the observational and predictive data.
        trajectories_index (int | slice):
            Index for the specific trajectories to plot.
        num_trajectories (int):
            Number of random trajectories to plot for clarity.

    Returns:
        Result[List[Figure], Exception]:
            Success containing a list of Figure objects if plots are created
            without errors, otherwise the error wrapped in a Failure.
    """
    figs = []
    genes = idata.observed_data.observations.coords["genes"].values

    for gene_index in genes:
        fig, ax = plt.subplots(figsize=(12, 8))

        observed_data = idata.observed_data.observations.sel(
            genes=gene_index
        ).values
        observed_times = idata.observed_data.times.values

        sample_data = idata.posterior_predictive.observations.sel(
            chain=0, draw=trajectories_index, genes=gene_index
        ).values

        if observed_times.shape[1] == 1:
            observed_times = rearrange(observed_times, "c t -> (c t)")
            observed_data = rearrange(observed_data, "c t m -> (c t) m")
            sample_data = rearrange(sample_data, "s c t m -> s (c t) m")

        for modality_index, modality in enumerate(
            idata.observed_data.observations.coords["modalities"].values
        ):
            lines = ax.plot(
                observed_times,
                observed_data[..., modality_index],
                label=modality,
                color=color_map.get(modality, "blue"),
                marker=".",
                ms=12,
            )
            ax.legend(
                [lines[0]],
                [modality],
            )

            if num_trajectories > sample_data.shape[0]:
                logger.debug(
                    f"\nRequested number of trajectories ({num_trajectories}) "
                    f"exceeds available samples ({sample_data.shape[0]}).\n"
                    "Adjusting to maximum available.\n"
                )
                num_trajectories = sample_data.shape[0]
            else:
                sample_data = sample_data[slice(0, num_trajectories)]

            for idx in range(num_trajectories):
                ax.plot(
                    observed_times,
                    sample_data[idx, ..., modality_index],
                    alpha=samples_transparency,
                    color=color_map.get(modality, "blue"),
                    marker="2",
                    ms=12,
                )

        ax.set_xlabel("Time")
        ax.set_ylabel("Expression")
        ax.set_title(f"Gene {gene_index} - Observed and Predicted Trajectories")
        ax.legend()
        ax.grid(True)
        figs.append(fig)

    return figs


@beartype
def print_inference_data_structure(idata: InferenceData) -> str:
    """
    Generates a formatted string describing the structure of an InferenceData
    object, including the dimensions and sizes of variables in each group.

    Args:
        idata (InferenceData): The InferenceData object to describe.

    Returns:
        str: A formatted string with the complete structure description.
    """
    groups = idata.groups()
    info_lines = ["\nOverview of InferenceData structure:"]
    for group in groups:
        data_group = getattr(idata, group)
        info_lines.append(f"\nGroup: {group}")
        info_lines.append("  Variables and their dimensions:")
        for var_name, data_array in data_group.data_vars.items():
            dims_info = ", ".join(
                [f"{dim}={data_group[dim].size}" for dim in data_array.dims]
            )
            info_lines.append(f"  {var_name}: ({dims_info})")
        info_lines.append("")

    return "\n".join(info_lines).strip() + "\n"
