from typing import Any, Dict, Optional, Tuple
from unittest.mock import Mock

import pytest
from beartype import beartype
from flytekit.configuration import Config
from omegaconf import DictConfig
from pytest_mock import MockerFixture

from pyrovelocity.io.cluster import get_remote_task_results


class MockPrimitive:
    def __init__(self, value: Any):
        self.string_value: Optional[str] = (
            value if isinstance(value, str) else None
        )
        self.integer: Optional[int] = value if isinstance(value, int) else None
        self.float_value: Optional[float] = (
            value if isinstance(value, float) else None
        )
        self.boolean: Optional[bool] = (
            value if isinstance(value, bool) else None
        )


class MockScalar:
    def __init__(self, value: Any):
        self.primitive: MockPrimitive = MockPrimitive(value)


class MockLiteral:
    def __init__(self, value: Any):
        self.scalar: MockScalar = MockScalar(value)


class MockLiteralMap:
    def __init__(self, literals: Dict[str, Any]):
        self.literals: Dict[str, MockLiteral] = {
            k: MockLiteral(v) for k, v in literals.items()
        }


class MockFlyteRemote:
    @beartype
    def __init__(self, config: Any) -> None:
        self.config: Any = config

    @beartype
    def get(self, uri: str) -> MockLiteralMap:
        if uri.endswith("/i"):
            return MockLiteralMap({"input_key": "input_value"})
        elif uri.endswith("/o"):
            return MockLiteralMap({"output_key": "output_value"})
        raise ValueError(f"Unexpected URI: {uri}")


@pytest.fixture
@beartype
def mock_flyte_remote(mocker: MockerFixture) -> Mock:
    mock_config = mocker.Mock()
    mocker.patch.object(Config, "for_endpoint", return_value=mock_config)
    mock_remote = mocker.Mock(spec=MockFlyteRemote)
    mock_remote.get.side_effect = MockFlyteRemote(mock_config).get
    mocker.patch(
        "pyrovelocity.io.cluster.FlyteRemote", return_value=mock_remote
    )
    return mock_remote


@pytest.fixture
@beartype
def mock_print_config_tree(mocker: MockerFixture) -> Mock:
    return mocker.patch("pyrovelocity.io.cluster.print_config_tree")


@pytest.mark.parametrize(
    "execution_id, task_id, endpoint, protocol, project, domain",
    [
        (
            "exec-1",
            "task-1",
            "custom.endpoint",
            "protocol://v1",
            "custom-project",
            "custom-domain",
        ),
        (
            "exec-2",
            "task-2",
            "flyte.cluster.testorg.net",
            "flyte://v1",
            "testorg",
            "development",
        ),
    ],
)
@beartype
def test_get_remote_task_results(
    mocker: MockerFixture,
    mock_flyte_remote: Mock,
    mock_print_config_tree: Mock,
    execution_id: str,
    task_id: str,
    endpoint: str,
    protocol: str,
    project: str,
    domain: str,
) -> None:
    result: Tuple[DictConfig, DictConfig] = get_remote_task_results(
        execution_id, task_id, endpoint, protocol, project, domain
    )

    assert isinstance(result, Tuple)
    assert len(result) == 2
    assert all(isinstance(item, DictConfig) for item in result)

    inputs, outputs = result

    Config.for_endpoint.assert_called_once_with(endpoint=endpoint)

    expected_base_uri: str = (
        f"{protocol}/{project}/{domain}/{execution_id}/{task_id}"
    )
    mock_flyte_remote.get.assert_any_call(f"{expected_base_uri}/i")
    mock_flyte_remote.get.assert_any_call(f"{expected_base_uri}/o")

    assert "input_key" in inputs
    assert "output_key" in outputs
    assert inputs["input_key"] == "input_value"
    assert outputs["output_key"] == "output_value"

    assert mock_print_config_tree.call_count == 2
    mock_print_config_tree.assert_any_call({"input_key": "input_value"})
    mock_print_config_tree.assert_any_call({"output_key": "output_value"})


@beartype
def test_get_remote_task_results_type_checking() -> None:
    with pytest.raises(Exception):
        get_remote_task_results(123, "task_id")

    with pytest.raises(Exception):
        get_remote_task_results("execution_id", 123)

    with pytest.raises(Exception):
        get_remote_task_results("execution_id", "task_id", endpoint=123)
