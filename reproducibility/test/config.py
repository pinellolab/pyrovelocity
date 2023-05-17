import hydra
from hydra.core.config_store import ConfigStore
from hydra_zen import save_as_yaml
from omegaconf import DictConfig
from omegaconf import OmegaConf

# from pyrovelocity.config import hydra_zen_configure
from pyrovelocity.config import print_config_tree
from pyrovelocity.config import test_hydra_zen_configure as hydra_zen_configure
from pyrovelocity.utils import get_pylogger


@hydra.main(version_base="1.2", config_name="config")
def main(conf: DictConfig) -> None:
    """Generate pipeline configuration.
    Args:
        conf {DictConfig}: OmegaConf configuration for hydra
    """
    logger = get_pylogger(name="CONF", log_level="INFO")
    logger.info("\n\nResolving global configuration file\n\n")
    OmegaConf.resolve(conf)
    print_config_tree(conf, logger, ())
    save_as_yaml(conf, "config.yaml")


if __name__ == "__main__":
    # register the configuration in hydra's ConfigStore
    config = hydra_zen_configure()
    cs = ConfigStore.instance()
    cs.store(name="config", node=config)
    main()
