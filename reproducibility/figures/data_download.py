import argparse
import os
from logging import Logger
from pathlib import Path
from typing import Text

import scanpy as scp
import scvelo as scv
from config import config_setup
from omegaconf import DictConfig

from pyrovelocity.utils import get_pylogger
from pyrovelocity.utils import print_attributes


def download_datasets(conf: DictConfig, logger: Logger) -> None:
    for source in conf.sources:
        for data_set in conf[source].download:
            logger.info(
                f"\n\nVerifying {data_set} data:\n\n"
                f"  from url {conf[source][data_set].url}\n"
                f"  temporarily downloaded to {conf[source][data_set].dl_path}\n"
                f"  stored in {conf[source][data_set].rel_path}\n"
            )

            if os.path.isfile(conf[source][data_set].rel_path) and os.access(
                conf[source][data_set].rel_path, os.R_OK
            ):
                logger.info(f"{conf[source][data_set].rel_path} exists")
            else:
                logger.info(f"downloading {conf[source][data_set].rel_path} ...")
                if source == "scvelo":
                    dl_method = getattr(scv.datasets, data_set)
                    adata = dl_method()  # e.g. scv.datasets.pancreas()
                else:
                    adata = scp.read(
                        conf[source][data_set].rel_path,
                        backup_url=conf[source][data_set].url,
                    )

                print_attributes(adata)
                if conf[source][data_set].dl_path != conf[source][data_set].rel_path:
                    os.replace(
                        conf[source][data_set].dl_path,
                        conf[source][data_set].rel_path,
                    )
                    try:
                        os.rmdir(conf[source][data_set].dl_root)
                    except OSError as e:
                        logger.warn(f"{conf[source][data_set].dl_root} : {e.strerror}")
                        pass

                if os.path.isfile(conf[source][data_set].rel_path) and os.access(
                    conf[source][data_set].rel_path, os.R_OK
                ):
                    logger.info(
                        f"successfully downloaded {conf[source][data_set].rel_path}"
                    )
                else:
                    logger.warn(
                        f"cannot find and read {conf[source][data_set].rel_path}"
                    )


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
