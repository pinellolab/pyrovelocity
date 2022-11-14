import argparse
import os
from pathlib import Path
from typing import Text

import scvelo as scv
from config import config_setup

from pyrovelocity.utils import get_pylogger
from pyrovelocity.utils import print_attributes


def data_load(config_path: str) -> None:
    """Load external data.
    Args:
        config_path {Text}: path to config
    """
    conf = config_setup(config_path)

    logger = get_pylogger(name="DATA_LOAD", log_level=conf.base.log_level)

    logger.info(
        f"\n\nVerifying existence of paths for:\n\n"
        f"  external data: {conf.data_external.root_path}\n"
        f"  processed data: {conf.data_processed.root_path}\n"
    )
    Path(conf.data_external.root_path).mkdir(parents=True, exist_ok=True)
    Path(conf.data_processed.root_path).mkdir(parents=True, exist_ok=True)

    logger.info(
        f"\n\nDownloading pancreas data:\n\n"
        f"  from url {conf.data_external.pancreas.url}\n"
        f"  to {conf.data_external.pancreas.dl_path}\n"
        f"  storing in {conf.data_external.pancreas.rel_path}\n"
    )

    if os.path.isfile(conf.data_external.pancreas.rel_path) and os.access(
        conf.data_external.pancreas.rel_path, os.R_OK
    ):
        logger.info(f"{conf.data_external.pancreas.rel_path} exists")
    else:
        logger.info(f"downloading {conf.data_external.pancreas.rel_path} ...")
        # adata = load_data(data="pancreas")
        adata = scv.datasets.pancreas()
        print_attributes(adata)
        os.replace(
            conf.data_external.pancreas.dl_path,
            conf.data_external.pancreas.rel_path,
        )
        os.rmdir(conf.data_external.pancreas.dl_root)
        logger.info(f"downloaded {conf.data_external.pancreas.rel_path}")


if __name__ == "__main__":

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args = args_parser.parse_args()

    data_load(config_path=args.config)
