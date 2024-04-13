import arviz as az
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import xarray as xr
from arviz import InferenceData
from beartype import beartype
from beartype.typing import Tuple
from jaxtyping import ArrayLike
from jaxtyping import Float
from jaxtyping import Integer
from jaxtyping import jaxtyped
from numpyro.infer import MCMC
from numpyro.infer import NUTS
from numpyro.infer import Predictive

from pyrovelocity.logging import configure_logging
from pyrovelocity.models._deterministic_simulation import (
    solve_transcription_splicing_model,
)


__all__ = [
    "deterministic_transcription_splicing_probabilistic_model",
    "generate_test_data_for_deterministic_model_inference",
    "generate_prior_predictive_samples",
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
    num_cells, num_genes, num_timepoints, num_modalities = data.shape

    # priors
    initial_conditions = numpyro.sample(
        "initial_conditions",
        dist.LogNormal(loc=jnp.log(1.0), scale=0.1),
        sample_shape=(num_genes, num_modalities),
    )
    gamma = numpyro.sample(
        "gamma",
        dist.LogNormal(loc=jnp.log(2.0), scale=0.5),
        sample_shape=(num_genes,),
    )
    sigma = numpyro.sample(
        "sigma",
        dist.HalfNormal(scale=1.0),
        sample_shape=(num_modalities,),
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

    predictions = jax.vmap(
        jax.vmap(
            model_solver,
            in_axes=(0, None),
        ),
        in_axes=(None, 0),
    )(
        jnp.arange(num_genes),
        jnp.arange(num_cells),
    )

    logger.info(
        f"\nPredictions Shape: {predictions.shape}\n"
        f"Data Shape: {data.shape}\n\n"
    )
    sigma_expanded = sigma.reshape(1, 1, 1, 2)

    # with
    # numpyro.plate("gene_plate", num_genes, dim=-3),
    # numpyro.plate("cell_plate", num_cells, dim=-2),
    # numpyro.plate("time_plate", num_timepoints, dim=-1):
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
    initial_conditions_params: Tuple[float, float] = (1.0, 0.1),
    gamma_params: Tuple[float, float] = (2.0, 0.5),
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
        noise_levels (Tuple[float, float]): Standard deviations for observational noise on each modality.
        initial_conditions_params (Tuple[float, float]): Parameters (mean, std) for log-normal initial conditions.
        gamma_params (Tuple[float, float]): Parameters (mean, std) for log-normal gamma distribution.


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

    # Preparing for vectorized ODE solving
    def model_solver(ts, init_cond, rate):
        solution = solve_transcription_splicing_model(
            ts=ts.reshape(-1),  # Flatten the time array for all cells
            initial_state=init_cond,
            params=(rate,),
        ).ys

        return solution[sorted_indices.argsort()].reshape(
            num_cells, num_timepoints, num_modalities
        )

    # Compute the ODE solutions for all genes
    solutions = jax.vmap(
        model_solver,
        in_axes=(None, 0, 0),
    )(
        sorted_times,
        initial_conditions,
        gamma,
    )

    # Adding observational noise
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
def generate_prior_predictive_samples(
    times: TimeTensor,
    data: MultiModalTranscriptomeTensor,
    num_samples: int,
    num_cells: int,
    num_genes: int,
    num_timepoints: int,
) -> InferenceData:
    """
    Generate samples from the prior predictive distribution of the deterministic
    transcription-splicing model.

    Args:
        times (TimeTensor): Array of time points for each cell.
        data (MultiModalTranscriptomeTensor): Observed data tensor with dimensions
            [num_cells, num_genes, num_timepoints, num_modalities].
        num_samples (Integer): Number of prior samples to generate.
        num_cells (Integer): Number of cells in the data.
        num_genes (Integer): Number of genes in the data.
        num_timepoints (Integer): Number of timepoints per cell.

    Returns:
        az.InferenceData: An ArviZ InferenceData object containing the prior samples.
    """
    # Prepare the model function from the stored module
    model = deterministic_transcription_splicing_probabilistic_model

    # Setup the Predictive object for prior sampling
    rng_key = jax.random.PRNGKey(0)
    predictive = Predictive(model, num_samples=num_samples)

    # Generate prior predictive samples
    prior_predictions = predictive(rng_key, times=times, data=data)

    # Construct the InferenceData object
    idata = az.from_numpyro(
        posterior=None,
        prior=prior_predictions,
        coords={
            "cell": jnp.arange(num_cells),
            "gene": jnp.arange(num_genes),
            "timepoint": jnp.arange(num_timepoints),
        },
        dims={
            "u_obs": ["cell", "gene", "timepoint"],
            "s_obs": ["cell", "gene", "timepoint"],
        },
    )

    # Optionally, add observed data for diagnostics and comparison
    observed_data = xr.Dataset(
        {
            "u_obs": (["cell", "gene", "timepoint"], data[..., 0]),
            "s_obs": (["cell", "gene", "timepoint"], data[..., 1]),
        },
        coords={
            "cell": jnp.arange(num_cells),
            "gene": jnp.arange(num_genes),
            "timepoint": jnp.arange(num_timepoints),
        },
    )
    idata.add_groups({"observed_data": observed_data})

    return idata
