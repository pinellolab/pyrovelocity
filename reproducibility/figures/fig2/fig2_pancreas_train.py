import argparse
import os
import pickle
from logging import Logger
from pathlib import Path
from typing import Text

from omegaconf import DictConfig

from pyrovelocity.api import train_model
from pyrovelocity.config import config_setup
from pyrovelocity.data import load_data
from pyrovelocity.plot import compute_mean_vector_field
from pyrovelocity.plot import vector_field_uncertainty
from pyrovelocity.utils import get_pylogger
from pyrovelocity.utils import print_attributes


"""Loads processed pancreas data and trains and saves model1 model.

Inputs:
  "pancreas_scvelo_fitted_2000_30.h5ad" via load_data()

Outputs:
  data:
    "fig2_pancreas_processed.h5ad"
    "fig2_pancreas_data.pkl"
  models:
    Fig2_pancreas_model/
    ├── attr.pkl
    ├── model_params.pt
    ├── param_store_test.pt
    └── var_names.csv
"""


def train(conf: DictConfig, logger: Logger) -> None:
    ###########
    # load data
    ###########

    processed_path = conf.data_external.scvelo.pancreas.derived.rel_path

    trained_data_path = conf.model_training.pancreas_model1.trained_data_path
    model_path = conf.model_training.pancreas_model1.model_path
    pyrovelocity_data_path = conf.model_training.pancreas_model1.pyrovelocity_data_path

    logger.info(f"Loading data: {processed_path}")
    adata = load_data(processed_path=processed_path)
    print_attributes(adata)

    #############
    # train model
    #############

    if os.path.exists(model_path) and os.path.isfile(pyrovelocity_data_path):
        logger.info(
            f"{processed_path}\n{model_path}\n{pyrovelocity_data_path}\nall exist"
        )
    else:
        logger.info(f"Training model...")
        adata_model_pos = train_model(
            adata,
            max_epochs=4000,
            svi_train=False,
            log_every=1000,
            patient_init=45,
            batch_size=-1,
            use_gpu=0,
            include_prior=True,
            offset=False,
            library_size=True,
            patient_improve=1e-4,
            guide_type="auto_t0_constraint",
            train_size=1.0,
        )
        logger.info(f"Data attributes after model training")
        print_attributes(adata_model_pos)

        logger.info(f"computing vector field uncertainty")
        v_map_all, embeds_radian, fdri = vector_field_uncertainty(
            adata, adata_model_pos[1], basis="umap"
        )
        logger.info(f"Data attributes after computation of vector field uncertainty")
        print_attributes(adata_model_pos)

        #############
        # postprocess
        #############

        logger.info(f"checking shared time")

        def check_shared_time(adata_model_pos, adata):
            adata.obs["cell_time"] = adata_model_pos[1]["cell_time"].squeeze().mean(0)
            adata.obs["1-Cytotrace"] = 1 - adata.obs["cytotrace"]

        check_shared_time(adata_model_pos, adata)

        logger.info(f"computing mean vector field")
        basis = "umap"
        compute_mean_vector_field(pos=adata_model_pos[1], adata=adata, basis=basis)
        embed_mean = adata.obsm[f"velocity_pyro_{basis}"]
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


def main(config_path: str) -> None:
    """Train model
    Args:
        config_path {Text}: path to config
    """
    conf = config_setup(config_path)

    logger = get_pylogger(name="TRAIN", log_level=conf.base.log_level)
    logger.info(
        f"\n\nVerifying existence of paths for:\n\n"
        f"  processed data: {conf.data_external.processed_path}\n"
        f"  model data: {conf.model_training.pancreas_model1.path}\n"
    )
    Path(conf.data_external.processed_path).mkdir(parents=True, exist_ok=True)
    Path(conf.model_training.pancreas_model1.path).mkdir(parents=True, exist_ok=True)

    train(conf, logger)


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args = args_parser.parse_args()

    main(config_path=args.config)
