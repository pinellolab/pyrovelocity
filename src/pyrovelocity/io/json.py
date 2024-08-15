import json
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict

from beartype import beartype
from returns.result import Failure, Result, Success

__all__ = [
    "load_json",
    "merge_json",
    "combine_json_files",
    "add_duration_to_run_info",
]


@beartype
def load_json(file_path: Path) -> Result[Dict[str, Any], Exception]:
    """
    Load JSON data from a file.

    Args:
        file_path: Path to the JSON file.

    Returns:
        Result[Dict[str, Any], Exception]: A result object containing the loaded JSON data or an exception.

    Examples:
        >>> import tempfile
        >>> tmp = getfixture('tmp_path')  # Get a temporary directory path
        >>> json_file = tmp / 'test.json'
        >>> json_file.write_text('{"key": "value"}')  # Create a test JSON file
        >>> result = load_json(json_file)
        >>> print(f"Result: {result}")
        Result: <Success: {'key': 'value'}>
        >>> assert result.unwrap() == {"key": "value"}
        >>> non_existent_file = tmp / 'non_existent.json'
        >>> result = load_json(non_existent_file)
        >>> print(f"Result: {result}")
        Result: <Failure: ...
    """
    try:
        with file_path.open("r") as file:
            return Success(json.load(file))
    except Exception as e:
        return Failure(e)


@beartype
def merge_json(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two JSON dictionaries.

    Args:
        dict1: The first JSON dictionary.
        dict2: The second JSON dictionary.

    Returns:
        Dict[str, Any]: The merged dictionary.

    Examples:
        >>> dict1 = {"a": 1, "b": 2}
        >>> dict2 = {"b": 3, "c": 4}
        >>> result = merge_json(dict1, dict2)
        >>> print(result)
        {'a': 1, 'b': 3, 'c': 4}
        >>> assert result == {"a": 1, "b": 3, "c": 4}
    """
    return {**dict1, **dict2}


@beartype
def combine_json_files(
    file1: str | Path,
    file2: str | Path,
    output_file: str | Path,
) -> Result[None, Exception]:
    """
    Combine two JSON files and write the result to a new file.

    Args:
        file1: Path to the first JSON file.
        file2: Path to the second JSON file.
        output_file: Path to the output JSON file.

    Returns:
        Result[None, Exception]: A result object encapsulating success or failure.

    Examples:
        >>> tmp = getfixture('tmp_path')  # Get a temporary directory path
        >>> file1 = tmp / 'file1.json'
        >>> file2 = tmp / 'file2.json'
        >>> output_file = tmp / 'combined.json'
        >>> file1.write_text('{"a": 1, "b": 2}')
        >>> file2.write_text('{"b": 3, "c": 4}')
        >>> result = combine_json_files(file1, file2, output_file)
        >>> print(f"Result: {result}")
        Result: <Success: None>
        >>> assert output_file.exists()
        >>> with open(output_file, 'r') as f:
        ...     content = json.load(f)
        ...     assert content == {"a": 1, "b": 3, "c": 4}
    """
    try:
        data1_result = load_json(Path(file1))
        data2_result = load_json(Path(file2))

        if isinstance(data1_result, Failure):
            return data1_result
        if isinstance(data2_result, Failure):
            return data2_result

        data1 = data1_result.unwrap()
        data2 = data2_result.unwrap()

        combined_data = merge_json(data1, data2)

        with output_file.open("w") as file:
            json.dump(combined_data, file, indent=2)

        return Success(None)
    except Exception as e:
        return Failure(e)

