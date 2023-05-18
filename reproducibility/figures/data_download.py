import os
from logging import Logger
from pathlib import Path
from typing import Text

import hydra
import scanpy as scp
import scvelo as scv
from omegaconf import DictConfig

from pyrovelocity.config import print_config_tree
from pyrovelocity.utils import generate_sample_data
from pyrovelocity.utils import get_pylogger
from pyrovelocity.utils import print_anndata
from pyrovelocity.utils import print_attributes


def download_datasets(conf: DictConfig, logger: Logger) -> None:
    # for source in conf.sources:
    for data_set in conf.process_data:
        data_set_conf = conf.data_sets[data_set]
        source = data_set_conf.source
        data_url = data_set_conf.url
        dl_root = data_set_conf.dl_root
        dl_path = data_set_conf.dl_path
        data_path = data_set_conf.rel_path

        logger.info(
            f"\n\nVerifying existence of path for:\n\n"
            f"  downloaded data: {data_set_conf.dl_root}\n"
            #     f"  processed data: {conf.data_external.processed_path}\n"
        )
        Path(data_set_conf.dl_root).mkdir(parents=True, exist_ok=True)
        # Path(conf.data_external.processed_path).mkdir(parents=True, exist_ok=True)

        logger.info(
            f"\n\nVerifying {data_set} data:\n\n"
            f"  from url {data_url}\n"
            f"  temporarily downloaded to {dl_path}\n"
            f"  stored in {data_path}\n"
        )

        if os.path.isfile(data_path) and os.access(data_path, os.R_OK):
            logger.info(f"{data_path} exists")
        else:
            logger.info(f"downloading {data_path} ...")
            if source == "scvelo":
                dl_method = getattr(scv.datasets, data_set)
                adata = dl_method()  # e.g. scv.datasets.pancreas()
            elif source == "simulate":
                adata = generate_sample_data(
                    n_obs=3000,
                    n_vars=1000,
                    noise_model="gillespie",
                    random_seed=99,
                )
                adata.write(data_path)
            else:
                adata = scp.read(
                    data_path,
                    backup_url=data_url,
                )

            print_attributes(adata)
            print_anndata(adata)
            if dl_path != data_path:
                os.replace(
                    dl_path,
                    data_path,
                )
                try:
                    os.rmdir(dl_root)
                except OSError as e:
                    logger.warn(f"{dl_root} : {e.strerror}")
            if os.path.isfile(data_path) and os.access(data_path, os.R_OK):
                logger.info(f"successfully downloaded {data_path}")
            else:
                logger.warn(f"cannot find and read {data_path}")


@hydra.main(version_base="1.2", config_path=".", config_name="config.yaml")
def main(conf: DictConfig) -> None:
    """Load external data.
    Args:
        conf {DictConfig}: hydra configuration
    """

    logger = get_pylogger(name="DATA_LOAD", log_level=conf.base.log_level)
    print_config_tree(conf, logger, ())

    download_datasets(conf, logger)


if __name__ == "__main__":
    main()
