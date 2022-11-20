import argparse
import os
from logging import Logger
from pathlib import Path
from typing import Text

import scanpy as scp
import scvelo as scv
from omegaconf import DictConfig

from pyrovelocity.config import config_setup
from pyrovelocity.utils import get_pylogger
from pyrovelocity.utils import print_attributes


def download_datasets(conf: DictConfig, logger: Logger) -> None:
    for source in conf.sources:
        for data_set in conf[source].download:
            data_url = conf[source][data_set].url
            dl_root = conf[source][data_set].dl_root
            dl_path = conf[source][data_set].dl_path
            data_path = conf[source][data_set].rel_path

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
                else:
                    adata = scp.read(
                        data_path,
                        backup_url=data_url,
                    )

                print_attributes(adata)
                if dl_path != data_path:
                    os.replace(
                        dl_path,
                        data_path,
                    )
                    try:
                        os.rmdir(dl_root)
                    except OSError as e:
                        logger.warn(f"{dl_root} : {e.strerror}")
                        pass

                if os.path.isfile(data_path) and os.access(data_path, os.R_OK):
                    logger.info(f"successfully downloaded {data_path}")
                else:
                    logger.warn(f"cannot find and read {data_path}")


def main(config_path: str) -> None:
    """Load external data.
    Args:
        config_path {Text}: path to config
    """
    conf = config_setup(config_path)

    logger = get_pylogger(name="DATA_LOAD", log_level=conf.base.log_level)
    logger.info(
        f"\n\nVerifying existence of paths for:\n\n"
        f"  external data: {conf.data_external.root_path}\n"
        f"  processed data: {conf.data_external.processed_path}\n"
    )
    Path(conf.data_external.root_path).mkdir(parents=True, exist_ok=True)
    Path(conf.data_external.processed_path).mkdir(parents=True, exist_ok=True)

    download_datasets(conf.data_external, logger)


if __name__ == "__main__":

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args = args_parser.parse_args()

    main(config_path=args.config)
