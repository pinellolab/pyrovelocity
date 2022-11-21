import hydra
from omegaconf import DictConfig
from omegaconf import OmegaConf

from pyrovelocity.config import print_config_tree
from pyrovelocity.utils import get_pylogger


@hydra.main(version_base="1.2", config_path=".", config_name="template-config.yaml")
def main(conf: DictConfig) -> None:
    """Generate pipeline configuration.
    Args:
        conf {DictConfig}: OmegaConf configuration for hydra
    """
    logger = get_pylogger(name="CONF", log_level=conf.base.log_level)
    logger.info(f"\n\nResolving global configuration file\n\n")
    OmegaConf.resolve(conf)
    print_config_tree(conf, logger, ())
    OmegaConf.save(config=conf, f="config.yaml")


if __name__ == "__main__":
    main()
