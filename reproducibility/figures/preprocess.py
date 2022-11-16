import argparse
import os
from logging import Logger
from pathlib import Path
from typing import Text

from config import config_setup
from omegaconf import DictConfig

import pyrovelocity.data
from pyrovelocity.utils import get_pylogger
from pyrovelocity.utils import print_attributes


def preprocess(conf: DictConfig, logger: Logger) -> None:
    for source in conf.sources:
        for data_set in conf[source].process:

            data_path = conf[source][data_set].rel_path
            processed_path = conf[source][data_set].derived.rel_path
            process_method = conf[source][data_set].derived.process_method

            logger.info(
                f"\n\nPreprocessing {data_set} data :\n\n"
                f"  from external: {data_path}\n"
                f"  to processed: {processed_path}\n"
                f"  using method: {process_method}\n"
            )

            if os.path.isfile(processed_path) and os.access(processed_path, os.R_OK):
                logger.info(f"{processed_path} exists")
            else:
                logger.info(f"generating {processed_path} ...")
                process_method_fn = getattr(pyrovelocity.data, process_method)
                adata = process_method_fn(
                    data=data_path,
                    processed_path=processed_path,
                )
                print_attributes(adata)

                if os.path.isfile(processed_path) and os.access(
                    processed_path, os.R_OK
                ):
                    logger.info(f"successfully generated {processed_path}")
                else:
                    logger.warn(f"cannot find and read {processed_path}")


def main(config_path: str) -> None:
    """Preprocess data.
    Args:
        config_path {Text}: path to config
    """
    conf = config_setup(config_path)

    logger = get_pylogger(name="PREPROCESS", log_level=conf.base.log_level)
    logger.info(
        f"\n\nVerifying existence of paths for:\n\n"
        f"  external data: {conf.data_external.root_path}\n"
        f"  processed data: {conf.data_external.processed_path}\n"
    )
    Path(conf.data_external.root_path).mkdir(parents=True, exist_ok=True)
    Path(conf.data_external.processed_path).mkdir(parents=True, exist_ok=True)

    preprocess(conf.data_external, logger)


if __name__ == "__main__":

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args = args_parser.parse_args()

    main(config_path=args.config)
