import json
import multiprocessing
import os
import pickle
import uuid
from logging import Logger
from pathlib import Path

import hydra
import mlflow
import scvelo as scv
import torch
from mlflow import MlflowClient
from omegaconf import DictConfig

# from pyrovelocity.api import train_model
from pyrovelocity.config import print_config_tree
from pyrovelocity.io.compressedpickle import CompressedPickle
# from pyrovelocity.data import load_data
# from pyrovelocity.utils import filter_startswith_dict
from pyrovelocity.utils import get_pylogger
from pyrovelocity.utils import mae_evaluate
from pyrovelocity.utils import pretty_print_dict
from pyrovelocity.utils import print_anndata
from pyrovelocity.utils import print_attributes
from pyrovelocity._velocity import PyroVelocity


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


def postprocess(conf: DictConfig, logger: Logger) -> None:
    for data_model in conf.train_models:
        ###########
        # load data
        ###########
        data_model_conf = conf.model_training[data_model]
        processed_path = data_model_conf.input_data_path

        trained_data_path = data_model_conf.trained_data_path
        model_path = data_model_conf.model_path
        pyrovelocity_data_path = data_model_conf.pyrovelocity_data_path
        posterior_samples_path = data_model_conf.posterior_samples_path
        vector_field_basis = data_model_conf.vector_field_parameters.basis
        metrics_path = data_model_conf.metrics_path
        # run_info_path = data_model_conf.run_info_path

        print_config_tree(data_model_conf, logger, ())

        logger.info(f"\n\nPostprocessing model data for: {data_model_conf}\n\n")

        ncpus_use = min(23, max(1, round(multiprocessing.cpu_count() * 0.8)))
        print("ncpus_use:", ncpus_use)
        # logger.info(
        #     f"\n\nVerifying existence of paths for:\n\n"
        #     f"  model data: {data_model_conf.path}\n"
        # )
        # Path(data_model_conf.path).mkdir(parents=True, exist_ok=True)

        # if os.path.isfile(processed_path):
        #     logger.info(f"Loading data: {processed_path}")
        #     adata = load_data(processed_path=processed_path)
        #     print_attributes(adata)
        # else:
        #     logger.error(f"Input data: {processed_path} does not exist")
        #     raise FileNotFoundError(
        #         f"Check {processed_path} output of preprocessing stage."
        #     )

        logger.info(f"Loading trained data: {trained_data_path}")
        adata = scv.read(trained_data_path)
        print_anndata(adata)

        logger.info(f"Loading pyrovelocity data: {pyrovelocity_data_path}")
        # with open(pyrovelocity_data_path, "rb") as f:
        #     posterior_samples = pickle.load(f)
        posterior_samples = CompressedPickle.load(posterior_samples_path)

        logger.info(f"Loading model data: {model_path}")
        trained_model = PyroVelocity.load_model(model_path, adata)

        #############
        # postprocess
        #############

        if os.path.exists(model_path) and os.path.isfile(pyrovelocity_data_path):
            logger.info(
                f"{processed_path}\n{model_path}\n{pyrovelocity_data_path}\nall exist"
            )
        else:
            logger.info(f"Postprocessing model data: {data_model}")

            # UPDATE: v2.1.1 12/26/2022 autolog only supports pytorch lightning
            # mlflow.pytorch.autolog(log_every_n_epoch=200, log_models=False, silent=False)
            with mlflow.start_run(
                run_name=f"{data_model}-{uuid.uuid4().hex[:7]}"
            ) as run:
                mlflow.set_tag("mlflow.runName", f"{data_model}-{run.info.run_id[:7]}")
                print(f"Active run_id: {run.info.run_id}")
                mlflow.log_params(data_model_conf.training_parameters)

                # trained_model, posterior_samples = train_model(
                #     adata,
                #     **dict(
                #         filter_startswith_dict(data_model_conf.training_parameters),
                #         use_gpu=gpu_id,
                #     ),
                # )

                # logger.info("Data attributes after model training")
                pretty_print_dict(posterior_samples)

                mae_df = mae_evaluate(posterior_samples, adata)
                mlflow.log_metric("MAE", mae_df["MAE"].mean())

                logger.info("Computing vector field uncertainty")

                pyrovelocity_data = (
                    trained_model.compute_statistics_from_posterior_samples(
                        adata,
                        posterior_samples,
                        vector_field_basis=vector_field_basis,
                        ncpus_use=ncpus_use,
                    )
                )
                logger.info(
                    "Data attributes after computation of vector field uncertainty"
                )
                pretty_print_dict(posterior_samples)
                print(posterior_samples.keys())

                run_id = run.info.run_id

            logger.info(f"Saving pyrovelocity data: {pyrovelocity_data_path}")
            # trained_model.save_pyrovelocity_data(
            #     pyrovelocity_data, pyrovelocity_data_path
            # )
            CompressedPickle.save(
                pyrovelocity_data_path, pyrovelocity_data
            )

            ################
            # update metrics
            ################

            r = mlflow.get_run(run_id)

            # Path(metrics_path).write_text(json.dumps(r.data.metrics, indent=4))
            update_json(r, metrics_path)

            # Path(run_info_path).write_text(
            #     json.dumps(r.to_dictionary()["info"], indent=4)
            # )

            print_logged_info(r)




def update_json(r: mlflow.entities.run.Run, metrics_path: str) -> None:
    metrics_path = Path(metrics_path)
    if metrics_path.is_file():
        with metrics_path.open() as file:
            existing_data = json.load(file)
    else:
        existing_data = {}

    existing_data.update(r.data.metrics)

    with metrics_path.open('w') as file:
        json.dump(existing_data, file, indent=4)


def print_logged_info(r: mlflow.entities.run.Run) -> None:
    tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in MlflowClient().list_artifacts(r.info.run_id, "model")]
    print(f"run_id: {r.info.run_id}")
    print(f"artifacts: {artifacts}")
    print(f"params: {r.data.params}")
    print(f"metrics: {r.data.metrics}")
    print(f"tags: {tags}")


@hydra.main(version_base="1.2", config_path=".", config_name="config.yaml")
def main(conf: DictConfig) -> None:
    """Train model
    Args:
        config {DictConfig}: hydra configuration
    """

    logger = get_pylogger(name="POSTPROCESS", log_level=conf.base.log_level)
    postprocess(conf, logger)


if __name__ == "__main__":
    main()
