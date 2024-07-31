from typing import Tuple

from beartype import beartype
from flytekit.configuration import Config
from flytekit.interaction.string_literals import literal_map_string_repr
from flytekit.remote.remote import FlyteRemote
from omegaconf import DictConfig, OmegaConf

from pyrovelocity.utils import print_config_tree

__all__ = ["get_remote_task_results"]


@beartype
def get_remote_task_results(
    execution_id: str,
    task_id: str,
    endpoint: str = "flyte.cluster.pyrovelocity.net",
    remote_uri_protocol: str = "flyte://v1",
    remote_project: str = "pyrovelocity",
    remote_domain: str = "development",
) -> Tuple[DictConfig, DictConfig]:
    """
    Get the inputs and outputs of a remote Flyte task.

    Args:
        execution_id (str): Flyte execution id.
        task_id (str): Flyte task id. Defaults to "f26c1pjy-0-dn1-0-dn6".
        endpoint (str): Cluster endpoint url. Defaults to "flyte.cluster.pyrovelocity.net".
        remote_uri_protocol (str): Flyte remote uri protocol. Defaults to "flyte://v1".
        remote_project (str): Flyte remote project. Defaults to "pyrovelocity".
        remote_domain (str): Flyte remote domain. Defaults to "development".

    Returns:
        Tuple[DictConfig, DictConfig]: Autocompletable inputs and outputs of
            the remote task execution.

    Example:
        >>> # xdoctest: +REQUIRES(env:NETWORK_ACCESS==true)
        >>> from pyrovelocity.io.cluster import get_remote_task_results
        ...
        >>> execution_id = "pyrovelocity-2021-09-01-16-00-00-000000-utc"
        >>> task_id = "f26c1pjy-0-dn1-0-dn6"
        ...
        >>> (
        ...     model1_postprocessing_inputs,
        ...     model1_postprocessing_outputs,
        ... ) = get_remote_task_results(
        ...     execution_id=execution_id,
        ...     task_id=task_id,
        ... )
    """
    remote = FlyteRemote(
        Config.for_endpoint(endpoint=endpoint),
    )
    execution_uri = (
        f"{remote_uri_protocol}/{remote_project}/{remote_domain}/{execution_id}"
    )
    task_uri = f"{execution_uri}/{task_id}"
    inputs = remote.get(f"{task_uri}/i")
    outputs = remote.get(f"{task_uri}/o")
    inputs_dict = literal_map_string_repr(inputs.literals)
    inputs_dictconfig = OmegaConf.create(inputs_dict)
    print_config_tree(inputs_dict)

    outputs_dict = literal_map_string_repr(outputs.literals)
    outputs_dictconfig = OmegaConf.create(outputs_dict)
    print_config_tree(outputs_dict)
    return inputs_dictconfig, outputs_dictconfig
