import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf.dictconfig import DictConfig


def get_app_config() -> DictConfig:
    GlobalHydra.instance().clear()
    hydra.initialize(
        version_base="1.2", config_path="../../reproducibility/figures"
    )
    return hydra.compose(config_name="config")
