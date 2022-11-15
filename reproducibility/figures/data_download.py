import argparse
import os
from logging import Logger
from pathlib import Path
from typing import Text

import scvelo as scv
from config import config_setup
from omegaconf import DictConfig

from pyrovelocity.utils import get_pylogger
from pyrovelocity.utils import print_attributes


def download_scvelo_datasets(conf: DictConfig, logger: Logger) -> None:
    for data_set in conf.download:
        logger.info(
            f"\n\nVerifying {data_set} data:\n\n"
            f"  from url {conf[data_set].url}\n"
            f"  temporarily downloaded to {conf[data_set].dl_path}\n"
            f"  stored in {conf[data_set].rel_path}\n"
        )

        if os.path.isfile(conf[data_set].rel_path) and os.access(
            conf[data_set].rel_path, os.R_OK
        ):
            logger.info(f"{conf[data_set].rel_path} exists")
        else:
            logger.info(f"downloading {conf[data_set].rel_path} ...")
            dl_method = getattr(scv.datasets, data_set)
            adata = dl_method()  # e.g. scv.datasets.pancreas()
            print_attributes(adata)
            os.replace(
                conf[data_set].dl_path,
                conf[data_set].rel_path,
            )
            os.rmdir(conf[data_set].dl_root)
            logger.info(f"downloaded {conf[data_set].rel_path}")


def data_download(conf: DictConfig) -> None:
    """Load external data.
    Args:
        config_path {Text}: path to config
    """

    logger = get_pylogger(name="DATA_LOAD", log_level=conf.base.log_level)
    logger.info(
        f"\n\nVerifying existence of paths for:\n\n"
        f"  external data: {conf.data_external.root_path}\n"
        f"  processed data: {conf.data_processed.root_path}\n"
    )
    Path(conf.data_external.root_path).mkdir(parents=True, exist_ok=True)
    Path(conf.data_processed.root_path).mkdir(parents=True, exist_ok=True)

    download_scvelo_datasets(conf.data_external.scvelo, logger)


def main(config_path: str) -> None:
    conf = config_setup(config_path)
    data_download(conf)


if __name__ == "__main__":

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args = args_parser.parse_args()

    main(config_path=args.config)
