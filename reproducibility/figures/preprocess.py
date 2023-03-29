import os
from logging import Logger
from pathlib import Path
from typing import Text

import hydra
from omegaconf import DictConfig

import pyrovelocity.data
from pyrovelocity.config import print_config_tree
from pyrovelocity.utils import get_pylogger
from pyrovelocity.utils import print_attributes


def preprocess(conf: DictConfig, logger: Logger) -> None:
    """Preprocess data.

    Args:
        conf (DictConfig): omegaconf configuration
        logger (Logger): PREPROCESS logger

    Examples:
        preprocess(conf.data_external, logger)
    """
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


@hydra.main(version_base="1.2", config_path=".", config_name="config.yaml")
def main(conf: DictConfig) -> None:
    """Preprocess data.
    Args:
        conf {DictConfig}: hydra configuration
    """

    logger = get_pylogger(name="PREPROCESS", log_level=conf.base.log_level)
    print_config_tree(conf, logger, ())

    logger.info(
        f"\n\nVerifying existence of paths for:\n\n"
        f"  external data: {conf.data_external.root_path}\n"
        f"  processed data: {conf.data_external.processed_path}\n"
    )
    Path(conf.data_external.root_path).mkdir(parents=True, exist_ok=True)
    Path(conf.data_external.processed_path).mkdir(parents=True, exist_ok=True)

    preprocess(conf.data_external, logger)


if __name__ == "__main__":
    main()
