import json
import multiprocessing
import os
import uuid
from logging import Logger
from pathlib import Path

import hydra
import mlflow
import torch
from mlflow import MlflowClient
from omegaconf import DictConfig
from pyrovelocity.api import train_model
from pyrovelocity.config import print_config_tree
from pyrovelocity.data import load_data
from pyrovelocity.io.compressedpickle import CompressedPickle
from pyrovelocity.utils import (
    filter_startswith_dict,
    get_pylogger,
    print_anndata,
    print_attributes,
)

"""Loads processed data and trains and saves model.

Inputs:
  "data/processed/{data_set_name}_processed.h5ad"

Outputs:
  data:
    "models/{data_model}/trained.h5ad"
    "models/{data_model}/pyrovelocity.pkl"
  models:
    models/{data_model}/model/
    ├── attr.pkl
    ├── model_params.pt
    ├── param_store_test.pt
    └── var_names.csv
"""


def train(conf: DictConfig, logger: Logger) -> None:
    for data_model in conf.train_models:
        ###########
        # load data
        ###########
        data_model_conf = conf.model_training[data_model]
        processed_path = data_model_conf.input_data_path

        trained_data_path = data_model_conf.trained_data_path
        model_path = data_model_conf.model_path
        posterior_samples_path = data_model_conf.posterior_samples_path
        pyrovelocity_data_path = data_model_conf.pyrovelocity_data_path
        vector_field_basis = data_model_conf.vector_field_parameters.basis
        metrics_path = data_model_conf.metrics_path
        run_info_path = data_model_conf.run_info_path

        print_config_tree(data_model_conf, logger, ())
        logger.info(f"\n\nTraining model(s) in: {data_model}\n\n")

        ncpus_use = min(23, max(1, round(multiprocessing.cpu_count() * 0.8)))
        print("ncpus_use:", ncpus_use)
        logger.info(
            f"\n\nVerifying existence of paths for:\n\n"
            f"  model data: {data_model_conf.path}\n"
        )
        Path(data_model_conf.path).mkdir(parents=True, exist_ok=True)

        if os.path.isfile(processed_path):
            logger.info(f"Loading data: {processed_path}")
            adata = load_data(processed_path=processed_path)
            print_attributes(adata)
            print_anndata(adata)
        else:
            logger.error(f"Input data: {processed_path} does not exist")
            raise FileNotFoundError(
                f"Check {processed_path} output of preprocessing stage."
            )

        #############
        # train model
        #############

        if torch.cuda.is_available():
            accelerators = list(range(torch.cuda.device_count()))
            if len(accelerators) >= 4:
                gpu_id = data_model_conf.gpu_id
            else:
                gpu_id = accelerators.pop()
        else:
            gpu_id = False

        print(f"GPU ID: {gpu_id}")

        if os.path.exists(model_path) and os.path.isfile(
            pyrovelocity_data_path
        ):
            logger.info(
                f"{processed_path}\n{model_path}\n{pyrovelocity_data_path}\nall exist"
            )
        else:
            logger.info(f"Training model: {data_model}")

            # UPDATE: v2.1.1 12/26/2022 autolog only supports pytorch lightning
            # mlflow.pytorch.autolog(log_every_n_epoch=200, log_models=False, silent=False)
            with mlflow.start_run(
                run_name=f"{data_model}-{uuid.uuid4().hex[:7]}"
            ) as run:
                mlflow.set_tag(
                    "mlflow.runName", f"{data_model}-{run.info.run_id[:7]}"
                )
                print(f"Active run_id: {run.info.run_id}")
                mlflow.log_params(data_model_conf.training_parameters)

                trained_model, posterior_samples = train_model(
                    adata,
                    **dict(
                        filter_startswith_dict(
                            data_model_conf.training_parameters
                        ),
                        use_gpu=gpu_id,
                    ),
                )

                logger.info("Data attributes after model training")

                run_id = run.info.run_id

            logger.info(f"Saving pyrovelocity data: {pyrovelocity_data_path}")
            CompressedPickle.save(
                posterior_samples_path,
                posterior_samples,
            )

            ##############
            # save metrics
            ##############

            r = mlflow.get_run(run_id)

            Path(metrics_path).write_text(json.dumps(r.data.metrics, indent=4))

            Path(run_info_path).write_text(
                json.dumps(r.to_dictionary()["info"], indent=4)
            )

            print_logged_info(r)

            #############
            # postprocess
            #############

            if "pancreas" in data_model:
                logger.info("checking shared time")

                def check_shared_time(posterior_samples, adata):
                    adata.obs["cell_time"] = (
                        posterior_samples["cell_time"].squeeze().mean(0)
                    )
                    adata.obs["1-Cytotrace"] = 1 - adata.obs["cytotrace"]

                check_shared_time(posterior_samples, adata)

            print_attributes(adata)
            print_anndata(adata)

            ##################
            # save checkpoints
            ##################

            logger.info(f"Saving trained data: {trained_data_path}")
            adata.write(trained_data_path)

            logger.info(f"Saving model: {model_path}")
            trained_model.save_model(model_path, overwrite=True)


def print_logged_info(r: mlflow.entities.run.Run) -> None:
    tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [
        f.path for f in MlflowClient().list_artifacts(r.info.run_id, "model")
    ]
    print(f"run_id: {r.info.run_id}")
    print(f"artifacts: {artifacts}")
    print(f"params: {r.data.params}")
    print(f"metrics: {r.data.metrics}")
    print(f"tags: {tags}")


def get_least_busy_gpu():
    from gpustat import GPUStatCollection

    gpu_stats = GPUStatCollection.new_query()
    least_busy_gpu = min(gpu_stats, key=lambda gpu: len(gpu.processes))

    return least_busy_gpu.index


@hydra.main(version_base="1.2", config_path=".", config_name="config.yaml")
def main(conf: DictConfig) -> None:
    """Train model
    Args:
        config {DictConfig}: hydra configuration
    """

    logger = get_pylogger(name="TRAIN", log_level=conf.base.log_level)
    train(conf, logger)


if __name__ == "__main__":
    main()
