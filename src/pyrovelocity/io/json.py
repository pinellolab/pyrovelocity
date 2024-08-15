import json
import re
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from beartype import beartype
from beartype.typing import Any, Dict, Tuple
from returns.result import Failure, Result, Success
from rich.console import Console
from rich.table import Table

__all__ = [
    "load_json",
    "merge_json",
    "combine_json_files",
    "add_duration_to_run_info",
    "generate_tables",
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


@beartype
def add_duration_to_run_info(
    file_path: Path | str = "metrics.json"
) -> Result[None, Exception]:
    """
    Load a JSON file containing run information, add a duration field,
    and overwrite the file with the updated information.

    The duration is calculated as the difference between 'end_time' and 'start_time',
    and is added as a new field in the format "HH:MM:SS".

    Args:
        file_path (Path | str): Path to the JSON file. Defaults to "metrics.json".

    Returns:
        Result[None, Exception]: A Result object containing None if successful,
                                 or an Exception if an error occurred.

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> # Create a temporary JSON file
        >>> with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as temp_file:
        ...     json.dump({
        ...         "start_time": 1723744878851,
        ...         "end_time": 1723744887595,
        ...         "other_field": "value"
        ...     }, temp_file)
        ...     temp_file_path = Path(temp_file.name)
        >>> # Test the function
        >>> result = add_duration_to_run_info(temp_file_path)
        >>> print(f"Result: {result}")
        Result: <Success: None>
        >>> # Verify the file was updated correctly
        >>> with open(temp_file_path, 'r') as f:
        ...     updated_data = json.load(f)
        >>> print(f"Duration added: {updated_data['duration']}")
        Duration added: 00:00:08
        >>> assert 'duration' in updated_data
        >>> assert updated_data['duration'] == "00:00:08"
        >>> assert updated_data['other_field'] == "value"
        >>> # Clean up
        >>> temp_file_path.unlink()

        >>> # Test with non-existent file
        >>> result = add_duration_to_run_info("non_existent.json")
        >>> print(f"Result: {result}")
        Result: <Failure: ...

        >>> # Test with invalid JSON
        >>> with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as temp_file:
        ...     temp_file.write("Invalid JSON")
        ...     temp_file_path = Path(temp_file.name)
        >>> result = add_duration_to_run_info(temp_file_path)
        >>> print(f"Result: {result}")
        Result: <Failure: ...
        >>> # Clean up
        >>> temp_file_path.unlink()
    """
    file_path = Path(file_path)

    try:
        with file_path.open("r") as file:
            data: Dict[str, Any] = json.load(file)

        start_time = data.get("start_time")
        end_time = data.get("end_time")

        if start_time is not None and end_time is not None:
            duration = timedelta(milliseconds=end_time - start_time)
            hours, remainder = divmod(duration.seconds, 3600)
            minutes, seconds = divmod(remainder, 60)

            data["duration"] = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        else:
            return Failure(
                ValueError(
                    "Unable to calculate duration. Start time or end time is missing."
                )
            )

        with file_path.open("w") as file:
            json.dump(data, file, indent=2)

        return Success(None)

    except Exception as e:
        return Failure(e)


@beartype
def generate_tables(json_data: Dict[str, Any]) -> Tuple[str, str, str, Table]:
    """
    Generate LaTeX, HTML, and rich tables from the provided JSON data.

    Args:
        json_data (Dict[str, Any]):
            The JSON data containing metrics for different datasets and models.

    Returns:
        Tuple[str, str, Table]:
            A tuple containing the LaTeX table string, HTML table string,
            and rich Table object.

    Examples:
        >>> json_data = {
        ...     "simulated_model1-123": {"run_name": "simulated_model1-123", "-ELBO": -7.339, "MAE": 1.094},
        ...     "simulated_model2-456": {"run_name": "simulated_model2-456", "-ELBO": -7.512, "MAE": 1.123},
        ...     "pancreas_model1-789": {"run_name": "pancreas_model1-789", "-ELBO": -8.234, "MAE": 0.987},
        ...     "pancreas_model2-012": {"run_name": "pancreas_model2-012", "-ELBO": -8.123, "MAE": 0.998}
        ... }
        >>> latex, html, markdown, rich_table = generate_tables(json_data)
    """

    dataset_metrics = defaultdict(lambda: defaultdict(dict))
    for run_name, metrics in json_data.items():
        match = re.match(r"(\w+)_(model\d+)-", run_name)
        if match:
            dataset, model = match.groups()
            dataset_metrics[dataset][model]["-ELBO"] = metrics.get(
                "-ELBO", "N/A"
            )
            dataset_metrics[dataset][model]["MAE"] = metrics.get("MAE", "N/A")

    data = []
    for dataset, models in dataset_metrics.items():
        row = {"Dataset": dataset}
        for model in ["model1", "model2"]:
            if model in models:
                row[f"{model.capitalize()} -ELBO"] = models[model].get(
                    "-ELBO", "N/A"
                )
                row[f"{model.capitalize()} MAE"] = models[model].get(
                    "MAE", "N/A"
                )
            else:
                row[f"{model.capitalize()} -ELBO"] = "N/A"
                row[f"{model.capitalize()} MAE"] = "N/A"
        data.append(row)

    df = pd.DataFrame(data)

    df.columns = pd.MultiIndex.from_tuples(
        [
            ("Dataset", ""),
            ("Model 1", "-ELBO"),
            ("Model 1", "MAE"),
            ("Model 2", "-ELBO"),
            ("Model 2", "MAE"),
        ]
    )

    latex_table = df.to_latex(
        index=False,
        multicolumn=True,
        multicolumn_format="c",
        column_format="l|cc|cc",
    )
    latex_table = latex_table.replace("\\midrule", "\\midrule\n\\cline{2-5}")

    html_table = """
    <table border="1" class="dataframe">
      <thead>
        <tr>
          <th>Dataset</th>
          <th colspan="2">Model 1</th>
          <th colspan="2">Model 2</th>
        </tr>
        <tr>
          <th></th>
          <th>-ELBO</th>
          <th>MAE</th>
          <th>-ELBO</th>
          <th>MAE</th>
        </tr>
      </thead>
      <tbody>
        {rows}
      </tbody>
    </table>
    """

    rows = ""
    for _, row in df.iterrows():
        rows += f"""
        <tr>
          <td>{row[("Dataset", "")]}</td>
          <td>{row[("Model 1", "-ELBO")]:.4f}</td>
          <td>{row[("Model 1", "MAE")]:.4f}</td>
          <td>{row[("Model 2", "-ELBO")]:.4f}</td>
          <td>{row[("Model 2", "MAE")]:.4f}</td>
        </tr>
        """

    html_table = html_table.format(rows=rows)

    markdown_table = (
        "| Dataset   | Model 1   |          | Model 2   |          |\n"
    )
    markdown_table += (
        "|-----------|-----------|----------|-----------|----------|\n"
    )
    markdown_table += (
        "|           | -ELBO     | MAE      | -ELBO     | MAE      |\n"
    )
    for _, row in df.iterrows():
        markdown_table += (
            f"| {row['Dataset', '']} "
            f"| {row['Model 1', '-ELBO']:.4f} | {row['Model 1', 'MAE']:.4f} "
            f"| {row['Model 2', '-ELBO']:.4f} | {row['Model 2', 'MAE']:.4f} |\n"
        )

    rich_table = Table(title="Model Comparison")
    rich_table.add_column("Dataset", style="cyan", no_wrap=True)
    rich_table.add_column("Model 1", style="magenta", no_wrap=True)
    rich_table.add_column("", style="magenta", no_wrap=True)
    rich_table.add_column("Model 2", style="green", no_wrap=True)
    rich_table.add_column("", style="green", no_wrap=True)
    rich_table.add_row("", "-ELBO", "MAE", "-ELBO", "MAE")

    for _, row in df.iterrows():
        rich_table.add_row(
            row[("Dataset", "")],
            f"{row[('Model 1', '-ELBO')]:.4f}"
            if isinstance(row[("Model 1", "-ELBO")], float)
            else str(row[("Model 1", "-ELBO")]),
            f"{row[('Model 1', 'MAE')]:.4f}"
            if isinstance(row[("Model 1", "MAE")], float)
            else str(row[("Model 1", "MAE")]),
            f"{row[('Model 2', '-ELBO')]:.4f}"
            if isinstance(row[("Model 2", "-ELBO")], float)
            else str(row[("Model 2", "-ELBO")]),
            f"{row[('Model 2', 'MAE')]:.4f}"
            if isinstance(row[("Model 2", "MAE")], float)
            else str(row[("Model 2", "MAE")]),
        )

    console = Console()
    console.print(latex_table)
    console.print(html_table)
    console.print(markdown_table)
    console.print(rich_table)

    return (
        latex_table,
        html_table,
        markdown_table,
        rich_table,
    )
