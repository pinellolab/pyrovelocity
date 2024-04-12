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
     number_of_timepoints_per_cell",
]
MultiModalTranscriptomeTensor = Float[
    ArrayLike,
    # Cells | Genes per cell | Timepoints per cell| Modalities per cell
    # ------|----------------|--------------------|-------------------
    # 60    | 1              | 1                  | 2
    "number_of_cells \
     number_of_genes_per_cell \
     number_of_timepoints_per_cell \
     number_of_modalities_per_cell",
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
        dist.LogNormal(loc=jnp.log(1.0), scale=0.1).expand(
            [num_genes, num_modalities]
        ),
    )
    rate_params = numpyro.sample(
        "rate_params",
        dist.LogNormal(loc=jnp.log(2.0), scale=0.5).expand([num_genes]),
    )
    noise_std = numpyro.sample(
        "noise_std", dist.HalfNormal(scale=1.0).expand([2])
    )

    for i in range(num_genes):
        u0_i, s0_i = initial_conditions[i]
        gamma_i = rate_params[i]
        sigma_u, sigma_s = noise_std

        for j in range(num_cells):
            t_j = times[j]
            assert t_j.shape == (num_timepoints,)

            initial_state = jnp.array([u0_i, s0_i])

            # numerically integrate the ODEs to estimate
            # alues of u^* and s^* at each timepoint
            solution = solve_transcription_splicing_model(
                t_j, initial_state, jnp.array([gamma_i])
            )
            u_pred, s_pred = solution.ys[:, 0], solution.ys[:, 1]

            # observations for
            # [
            #  each gene i,
            #  and cell j,
            #  at given time points,
            #  for each modality,
            # ]
            u_obs = data[j, i, :, 0]
            s_obs = data[j, i, :, 1]

            # likelihood
            numpyro.sample(
                f"u_obs_{i}_{j}", dist.Normal(u_pred, sigma_u), obs=u_obs
            )
            numpyro.sample(
                f"s_obs_{i}_{j}", dist.Normal(s_pred, sigma_s), obs=s_obs
            )


@beartype
def generate_test_data_for_deterministic_model_inference(
    num_cells: int,
    num_genes: int,
    num_timepoints: int,
    num_modalities: int,
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
        >>> time_array, data, num_cells, num_genes, num_timepoints = (
        >>> generate_test_data_for_deterministic_model_inference(
        ...      num_cells=3,
        ...      num_genes=1,
        ...      num_timepoints=5,
        ...      num_modalities=2,
        >>> )
    """
    # simulate timepoints for each cell as a proxy for sampling
    times = jnp.logspace(-1, 1, num_timepoints)
    time_array = jnp.tile(times, (num_cells, 1))

    # set common initial conditions and parameters
    initial_conditions = jnp.array([0.1, 0.1])
    params = jnp.array([1.00])

    # initialize data tensor
    data = jnp.zeros((num_cells, num_genes, len(times), num_modalities))

    rng_key = jax.random.PRNGKey(0)
    rng_key, rng_key_ = jax.random.split(rng_key)

    for i in range(num_genes):
        # solve the ODEs for each gene
        solution = solve_transcription_splicing_model(
            times, initial_conditions, params
        )
        u_simulated, s_simulated = solution.ys[:, 0], solution.ys[:, 1]

        # add noise to the simulated data
        noise_level_u = 0.05
        noise_level_s = 0.05
        u_noisy = dist.Normal(u_simulated, noise_level_u).sample(rng_key_)
        s_noisy = dist.Normal(s_simulated, noise_level_s).sample(rng_key_)

        for j in range(num_cells):
            data = data.at[j, i, :, 0].set(u_noisy)
            data = data.at[j, i, :, 1].set(s_noisy)

    logger.info(f"\nTest data tensor shape: {data.shape}\n")
    logger.info(f"\nTest data time tensor shape: {time_array.shape}\n")
    return (
        time_array,
        data,
        num_cells,
        num_genes,
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
