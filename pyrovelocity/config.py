import argparse
from pathlib import Path
from typing import Sequence
from typing import Text
from typing import Union

import rich.console
import rich.pretty
import rich.syntax
import rich.tree
from omegaconf import DictConfig
from omegaconf import ListConfig
from omegaconf import OmegaConf
from pytorch_lightning.utilities import rank_zero_only

from pyrovelocity.utils import get_pylogger


def config_setup(config_path: str) -> Union[DictConfig, ListConfig]:
    """Convert template into concrete configuration file.
    Args:
        config_path {Text}: path to config
    """

    logger = get_pylogger(name="CONF")

    template_config_path = config_path.replace("config.yaml", "template-config.yaml")
    conf = OmegaConf.load(template_config_path)
    with open(config_path, "w") as conf_file:
        OmegaConf.save(config=conf, f=conf_file, resolve=True)

    conf = OmegaConf.load(config_path)
    print_config_tree(conf, logger, ())
    return conf

    # with open(config_path, "r") as conf_file:
    #     syntax = rich.syntax.Syntax(conf_file.read(), "yaml")

    # console = rich.console.Console()
    # console.print(syntax)


@rank_zero_only
def print_config_tree(
    cfg: DictConfig,
    logger,
    print_order: Sequence[str] = (
        "datamodule",
        "model",
        "callbacks",
        "logger",
        "trainer",
        "paths",
        "extras",
    ),
    resolve: bool = False,
    save_to_file: bool = False,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.
        print_order (Sequence[str], optional): Determines in what order config components are printed.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
        save_to_file (bool, optional): Whether to export config to the hydra output folder.
    """

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    queue = []

    # add fields from `print_order` to queue
    for field in print_order:
        queue.append(field) if field in cfg else logger.warning(
            f"Field '{field}' not found in config. Skipping '{field}' config printing..."
        )

    # add all the other fields to queue (not specified in `print_order`)
    for field in cfg:
        if field not in queue:
            queue.append(field)

    # generate config tree from queue
    for field in queue:
        branch = tree.add(field, style=style, guide_style=style)

        config_group = cfg[field]
        if isinstance(config_group, DictConfig):
            branch_content = OmegaConf.to_yaml(config_group, resolve=resolve)
        else:
            branch_content = str(config_group)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    # print config tree
    rich.print(tree)

    # save config tree to file
    if save_to_file:
        with open(Path(cfg.paths.output, "config_tree.log"), "w") as file:
            rich.print(tree, file=file)


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args = args_parser.parse_args()

    config_setup(config_path=args.config)
