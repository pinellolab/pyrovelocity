import json
import multiprocessing
import os
import uuid
from pathlib import Path
from typing import Tuple

import mlflow
import scanpy as sc
from mlflow import MlflowClient

from pyrovelocity.io.compressedpickle import CompressedPickle
from pyrovelocity.logging import configure_logging
from pyrovelocity.models._velocity import PyroVelocity
from pyrovelocity.utils import mae_evaluate
from pyrovelocity.utils import pretty_print_dict
from pyrovelocity.utils import print_anndata


__all__ = ["postprocess_dataset"]

logger = configure_logging(__name__)


def postprocess_dataset(
    data_model: str,
    data_model_path: str | Path,
    trained_data_path: str | Path,
    model_path: str | Path,
    posterior_samples_path: str | Path,
    metrics_path: str | Path,
    vector_field_basis: str,
    number_posterior_samples: int,
) -> Tuple[Path, Path]:
    """
    Postprocess dataset computing vector field uncertainty for a given set of posterior samples.

    Args:
        data_model (str): string containing the data set and model identifier, e.g. simulated_model1
        data_model_path (str | Path): top level directory for this data-model pair, e.g. models/simulated_model1
        trained_data_path (str | Path): path to the trained data, e.g. models/simulated_model1/trained.h5ad
        model_path (str | Path): path to the model, e.g. models/simulated_model1/model
        posterior_samples_path (str | Path): path to the posterior samples, e.g. models/simulated_model1/posterior_samples.pkl.zst
        metrics_path (str | Path): path to the metrics, e.g. models/simulated_model1/metrics.json
        vector_field_basis (str): basis for the vector field, e.g. umap

    Returns:
        str: path to the pyrovelocity output data

    Examples:
        >>> from pyrovelocity.tasks.postprocess import postprocess_dataset # xdoctest: +SKIP
        >>> tmp = getfixture("tmp_path") # xdoctest: +SKIP
        >>> postprocess_dataset(
        ...     "simulated_model1",
        ...     "models/simulated_model1",
        ...     "models/simulated_model1/trained.h5ad",
        ...     "models/simulated_model1/model",
        ...     "models/simulated_model1/posterior_samples.pkl.zst",
        ...     "models/simulated_model1/metrics.json",
        ...     "leiden",
        ...     3,
        ... ) # xdoctest: +SKIP
    """

    Path(data_model_path).mkdir(parents=True, exist_ok=True)
    pyrovelocity_data_path = os.path.join(
        data_model_path, "pyrovelocity.pkl.zst"
    )
    postprocessed_data_path = os.path.join(
        data_model_path, "postprocessed.h5ad"
    )

    number_of_cpus_to_use = min(
        23, max(1, round(multiprocessing.cpu_count() * 0.8))
    )
    logger.info(f"using {number_of_cpus_to_use} cpus")

    logger.info(f"Loading trained data: {trained_data_path}")
    adata = sc.read(trained_data_path)
    print_anndata(adata)

    logger.info(f"Loading posterior samples: {posterior_samples_path}")
    posterior_samples = CompressedPickle.load(posterior_samples_path)

    # TODO: parameterize the number of posterior samples to use
    posterior_samples_size = {
        len(value) for _, value in posterior_samples.items()
    }
    if len(posterior_samples_size) == 1:
        posterior_samples_size = posterior_samples_size.pop()
    else:
        pretty_print_dict(
            {key: value.shape for key, value in posterior_samples.items()}
        )
        raise ValueError(
            f"Posterior samples have different sizes: {posterior_samples_size}"
        )

    if number_posterior_samples < posterior_samples_size:
        logger.info(
            f"Using {number_posterior_samples} posterior samples from {posterior_samples_size} posterior samples"
        )
        posterior_samples = {
            key: value[:number_posterior_samples]
            for key, value in posterior_samples.items()
        }
    else:
        logger.info(
            f"Using all {posterior_samples_size} posterior samples because {number_posterior_samples} >= {posterior_samples_size}"
        )
    pretty_print_dict(
        {key: value.shape for key, value in posterior_samples.items()}
    )

    logger.info(f"Loading model data: {model_path}")
    trained_model = PyroVelocity.load_model(model_path, adata)

    if os.path.exists(model_path) and os.path.isfile(pyrovelocity_data_path):
        logger.info(
            f"{trained_data_path}\n{model_path}\n{pyrovelocity_data_path}\nall exist"
        )
    else:
        logger.info(f"Postprocessing model data: {data_model}")

        # mlflow v2.1.1 12/26/2022 autolog only supports pytorch lightning
        # mlflow.pytorch.autolog(log_every_n_epoch=200, log_models=False, silent=False)
        with mlflow.start_run(
            run_name=f"{data_model}-{uuid.uuid4().hex[:7]}"
        ) as run:
            mlflow.set_tag(
                "mlflow.runName", f"{data_model}-{run.info.run_id[:7]}"
            )
            print(f"Active run_id: {run.info.run_id}")

            pretty_print_dict(
                {key: value.shape for key, value in posterior_samples.items()}
            )

            mae_df = mae_evaluate(posterior_samples, adata)
            mlflow.log_metric("MAE", mae_df["MAE"].mean())

            logger.info("Computing vector field uncertainty")

            pyrovelocity_data = (
                trained_model.compute_statistics_from_posterior_samples(
                    adata,
                    posterior_samples,
                    vector_field_basis=vector_field_basis,
                    ncpus_use=number_of_cpus_to_use,
                )
            )
            logger.info(
                "Data attributes after computation of vector field uncertainty"
            )
            print_anndata(adata)

            run_id = run.info.run_id

        logger.info(f"Saving pyrovelocity data: {pyrovelocity_data_path}")
        CompressedPickle.save(pyrovelocity_data_path, pyrovelocity_data)

        logger.info(f"Saving postprocessed data: {postprocessed_data_path}")
        adata.write(postprocessed_data_path)

        r = mlflow.get_run(run_id)
        _update_json(r, metrics_path)
        mlflow_log_message = _generate_string_from_logged_info(r)
        logger.info(mlflow_log_message)

    return (pyrovelocity_data_path, postprocessed_data_path)


def _update_json(r: mlflow.entities.run.Run, metrics_path: str) -> None:
    metrics_path = Path(metrics_path)
    if metrics_path.is_file():
        with metrics_path.open() as file:
            existing_data = json.load(file)
    else:
        existing_data = {}

    existing_data.update(r.data.metrics)

    with metrics_path.open("w") as file:
        json.dump(existing_data, file, indent=4)


def _generate_string_from_logged_info(r: mlflow.entities.run.Run) -> None:
    tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [
        f.path for f in MlflowClient().list_artifacts(r.info.run_id, "model")
    ]

    return (
        f"\nrun_id: {r.info.run_id},\n"
        f"artifacts: {artifacts},\n"
        f"params: {r.data.params},\n"
        f"metrics: {r.data.metrics},\n"
        f"tags: {tags}\n\n"
    )
