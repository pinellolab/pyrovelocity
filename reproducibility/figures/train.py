import multiprocessing
import os
import pickle
from logging import Logger
from pathlib import Path
from typing import Text

import hydra
import torch
from omegaconf import DictConfig

from pyrovelocity.api import train_model
from pyrovelocity.config import print_config_tree
from pyrovelocity.data import load_data
from pyrovelocity.plot import compute_mean_vector_field
from pyrovelocity.plot import vector_field_uncertainty
from pyrovelocity.utils import get_pylogger
from pyrovelocity.utils import print_attributes


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
    for data_model in conf.model_training.train:
        ###########
        # load data
        ###########
        data_model_conf = conf.model_training[data_model]
        processed_path = data_model_conf.input_data_path

        trained_data_path = data_model_conf.trained_data_path
        model_path = data_model_conf.model_path
        pyrovelocity_data_path = data_model_conf.pyrovelocity_data_path
        vector_field_basis = data_model_conf.vector_field_parameters.basis

        ncpus_use = min(30, max(1, round(multiprocessing.cpu_count() * 0.8)))

        logger.info(
            f"\n\nVerifying existence of paths for:\n\n"
            f"  model data: {data_model_conf.path}\n"
        )
        Path(data_model_conf.path).mkdir(parents=True, exist_ok=True)

        if os.path.isfile(processed_path):
            logger.info(f"Loading data: {processed_path}")
            adata = load_data(processed_path=processed_path)
            print_attributes(adata)
        else:
            logger.error(f"Input data: {processed_path} does not exist")
            raise Exception(f"Check {processed_path} output of preprocessing stage.")

        #############
        # train model
        #############

        if torch.cuda.is_available():
            accelerators = [d for d in range(torch.cuda.device_count())]
            gpu_id = accelerators.pop()
        else:
            gpu_id = False

        if os.path.exists(model_path) and os.path.isfile(pyrovelocity_data_path):
            logger.info(
                f"{processed_path}\n{model_path}\n{pyrovelocity_data_path}\nall exist"
            )
        else:
            logger.info(f"Training model: {data_model}")
            adata_model_pos = train_model(
                adata, use_gpu=gpu_id, **data_model_conf.training_parameters
            )
            logger.info(f"Data attributes after model training")
            print_attributes(adata_model_pos[1])

            logger.info(f"computing vector field uncertainty")
            v_map_all, embeds_radian, fdri = vector_field_uncertainty(
                adata, adata_model_pos[1], basis=vector_field_basis, n_jobs=ncpus_use
            )
            logger.info(
                f"Data attributes after computation of vector field uncertainty"
            )
            print_attributes(adata_model_pos[1])

            #############
            # postprocess
            #############

            if "pancreas" in data_model:
                logger.info(f"checking shared time")

                def check_shared_time(adata_model_pos, adata):
                    adata.obs["cell_time"] = (
                        adata_model_pos[1]["cell_time"].squeeze().mean(0)
                    )
                    adata.obs["1-Cytotrace"] = 1 - adata.obs["cytotrace"]

                check_shared_time(adata_model_pos, adata)

            logger.info(f"computing mean vector field")
            compute_mean_vector_field(
                pos=adata_model_pos[1],
                adata=adata,
                basis=vector_field_basis,
                n_jobs=ncpus_use,
            )
            embed_mean = adata.obsm[f"velocity_pyro_{vector_field_basis}"]
            logger.info(f"Data attributes after computation of mean vector field")
            print_attributes(adata)

            ##################
            # save checkpoints
            ##################

            logger.info(f"Saving trained data: {trained_data_path}")
            adata.write(trained_data_path)

            logger.info(f"Saving model: {model_path}")
            adata_model_pos[0].save(model_path, overwrite=True)
            result_dict = {
                "adata_model_pos": adata_model_pos[1],
                "v_map_all": v_map_all,
                "embeds_radian": embeds_radian,
                "fdri": fdri,
                "embed_mean": embed_mean,
            }

            logger.info(f"Saving pyrovelocity data: {pyrovelocity_data_path}")
            with open(pyrovelocity_data_path, "wb") as f:
                pickle.dump(result_dict, f)


@hydra.main(version_base="1.2", config_path=".", config_name="config.yaml")
def main(conf: DictConfig) -> None:
    """Train model
    Args:
        config {DictConfig}: hydra configuration
    """

    logger = get_pylogger(name="TRAIN", log_level=conf.base.log_level)
    print_config_tree(conf, logger, ())

    logger.info(f"\n\nTraining model(s) in: {conf.model_training.train}\n\n")

    train(conf, logger)


if __name__ == "__main__":
    main()
