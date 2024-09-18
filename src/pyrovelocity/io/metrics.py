import json
import re
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from beartype import beartype
from beartype.typing import Any, Dict, List, Tuple
from returns.result import Failure, Result, Success
from rich.console import Console
from rich.table import Table

__all__ = [
    "load_json",
    "merge_json",
    "combine_json_files",
    "add_duration_to_run_info",
    "generate_tables",
    "generate_and_save_metric_tables",
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
def generate_metric_tables(
    json_data: Dict[str, Any], separate: bool = False
) -> Tuple[Dict[str, Dict[str, str] | str], Dict[str, Table]]:
    """
    Generate LaTeX, HTML, and Markdown tables for ELBO and MAE metrics from the provided JSON data.

    This function can generate either combined tables (ELBO and MAE together) or separate tables for each metric.

    Args:
        json_data (Dict[str, Any]): The JSON data containing metrics for different datasets and models.
        separate (bool): If True, generate separate tables for ELBO and MAE. If False, generate combined tables. Default is False.

    Returns:
        Tuple[Dict[str, str], Dict[str, Table]]: A tuple containing two dictionaries:
            1. A dictionary with keys 'latex', 'html', and 'markdown', each containing the respective table strings.
            2. A dictionary with rich Table objects.

    Examples:
        >>> json_data = {
        ...     "simulated_model1-123": {"run_name": "simulated_model1-123", "-ELBO": -7.339, "MAE": 1.094},
        ...     "simulated_model2-456": {"run_name": "simulated_model2-456", "-ELBO": -7.512, "MAE": 1.123},
        ...     "pancreas_model1-789": {"run_name": "pancreas_model1-789", "-ELBO": -8.234, "MAE": 0.987},
        ...     "pancreas_model2-012": {"run_name": "pancreas_model2-012", "-ELBO": -8.123, "MAE": 0.998}
        ... }
        >>> table_strings, rich_tables = generate_metric_tables(json_data)
        >>> separate_table_strings, separate_rich_tables = generate_metric_tables(json_data, separate=True)
    """
    dataset_metrics = process_json_data(json_data)

    if separate:
        return generate_separate_tables(dataset_metrics)
    else:
        return generate_combined_tables(dataset_metrics)


def process_json_data(
    json_data: Dict[str, Any]
) -> Dict[str, Dict[str, Dict[str, float]]]:
    dataset_metrics = defaultdict(lambda: defaultdict(dict))
    for run_name, metrics in json_data.items():
        match = re.match(r"(\w+)_(model\d+)-", run_name)
        if match:
            dataset, model = match.groups()
            dataset = dataset.replace("_", " ")
            if "simulated" in dataset.lower():
                continue
            dataset_metrics[dataset][model]["ELBO"] = metrics.get(
                "-ELBO", "N/A"
            )
            dataset_metrics[dataset][model]["MAE"] = metrics.get("MAE", "N/A")
    return dataset_metrics


def generate_combined_tables(
    dataset_metrics: Dict[str, Dict[str, Dict[str, float]]]
) -> Tuple[Dict[str, str], Dict[str, Table]]:
    data = []
    for dataset, models in dataset_metrics.items():
        row = {"Dataset": dataset}
        for model in ["model1", "model2"]:
            formatted_model = format_model_name(model)
            if model in models:
                row[f"{formatted_model} ELBO"] = models[model].get(
                    "ELBO", "N/A"
                )
                row[f"{formatted_model} MAE"] = models[model].get("MAE", "N/A")
            else:
                row[f"{formatted_model} ELBO"] = "N/A"
                row[f"{formatted_model} MAE"] = "N/A"
        data.append(row)

    df = pd.DataFrame(data)

    latex_table = generate_latex_table(df)
    html_table = generate_html_table(df)
    markdown_table = generate_markdown_table(df)
    rich_table = generate_rich_table(df)

    console = Console()
    console.print(rich_table)

    return {
        "latex": latex_table,
        "html": html_table,
        "markdown": markdown_table,
    }, {"combined": rich_table}


def format_model_name(model: str) -> str:
    return f"Model {model[-1]}"


def generate_separate_tables(
    dataset_metrics: Dict[str, Dict[str, Dict[str, float]]]
) -> Tuple[Dict[str, Dict[str, str]], Dict[str, Table]]:
    data_elbo, data_mae = [], []
    for dataset, models in dataset_metrics.items():
        row_elbo = {"Dataset": dataset}
        row_mae = {"Dataset": dataset}
        for model in ["model1", "model2"]:
            formatted_model = format_model_name(model)
            if model in models:
                row_elbo[formatted_model] = models[model].get("ELBO", "N/A")
                row_mae[formatted_model] = models[model].get("MAE", "N/A")
            else:
                row_elbo[formatted_model] = "N/A"
                row_mae[formatted_model] = "N/A"
        data_elbo.append(row_elbo)
        data_mae.append(row_mae)

    df_elbo = pd.DataFrame(data_elbo)
    df_mae = pd.DataFrame(data_mae)

    latex_elbo = generate_latex_table(df_elbo, separate=True)
    latex_mae = generate_latex_table(df_mae, separate=True)
    html_elbo = generate_html_table(df_elbo, separate=True)
    html_mae = generate_html_table(df_mae, separate=True)
    markdown_elbo = generate_markdown_table(df_elbo, separate=True)
    markdown_mae = generate_markdown_table(df_mae, separate=True)
    rich_table_elbo = generate_rich_table(
        df_elbo, title="ELBO Comparison", separate=True
    )
    rich_table_mae = generate_rich_table(
        df_mae, title="MAE Comparison", separate=True
    )

    console = Console()
    console.print(rich_table_elbo)
    console.print(rich_table_mae)

    return {
        "latex": {"ELBO": latex_elbo, "MAE": latex_mae},
        "html": {"ELBO": html_elbo, "MAE": html_mae},
        "markdown": {"ELBO": markdown_elbo, "MAE": markdown_mae},
    }, {"ELBO": rich_table_elbo, "MAE": rich_table_mae}


def wrap_latex_table(table_content: str) -> str:
    """
    Wrap the LaTeX table content in a complete LaTeX document structure.

    Args:
        table_content (str): The LaTeX table content generated by df.to_latex()

    Returns:
        str: The complete LaTeX document including the table
    """
    latex_document = r"""
\documentclass[border=2pt]{standalone}
\usepackage{booktabs}
\renewcommand{\familydefault}{\sfdefault}

\begin{document}

%s
\end{document}
"""
    return latex_document % table_content


def generate_latex_table(df: pd.DataFrame, separate: bool = False) -> str:
    if separate:
        table_content = df.to_latex(index=False, column_format="l|cc")
    else:
        table_content = df.to_latex(index=False, column_format="l|cc|cc")
    # return wrap_latex_table(table_content)
    return table_content


def generate_html_table(df: pd.DataFrame, separate: bool = False) -> str:
    if separate:
        header = """
        <table border="1" class="dataframe">
          <thead>
            <tr style="text-align: right;">
              <th>Dataset</th>
              <th>Model 1</th>
              <th>Model 2</th>
            </tr>
          </thead>
          <tbody>
        """
    else:
        header = """
        <table border="1" class="dataframe">
          <thead>
            <tr>
              <th>Dataset</th>
              <th colspan="2">Model 1</th>
              <th colspan="2">Model 2</th>
            </tr>
            <tr>
              <th></th>
              <th>ELBO</th>
              <th>MAE</th>
              <th>ELBO</th>
              <th>MAE</th>
            </tr>
          </thead>
          <tbody>
        """

    body = generate_html_rows(df, separate)
    footer = """
          </tbody>
        </table>
        """
    return header + body + footer


def generate_markdown_table(df: pd.DataFrame, separate: bool = False) -> str:
    if separate:
        header = "| Dataset   | Model 1 | Model 2 |\n|-----------|---------|----------|\n"
    else:
        header = "| Dataset   | Model 1 ELBO | Model 1 MAE | Model 2 ELBO | Model 2 MAE |\n|-----------|--------------|-------------|--------------|-------------|\n"

    body = generate_markdown_rows(df, separate)
    return header + body


def generate_rich_table(
    df: pd.DataFrame, title: str = "Model Comparison", separate: bool = False
) -> Table:
    rich_table = Table(title=title)
    rich_table.add_column("Dataset", style="cyan", no_wrap=True)
    if separate:
        rich_table.add_column("Model 1", style="magenta")
        rich_table.add_column("Model 2", style="green")
        for _, row in df.iterrows():
            rich_table.add_row(
                row["Dataset"],
                format_value(row["Model 1"]),
                format_value(row["Model 2"]),
            )
    else:
        rich_table.add_column("Model 1 ELBO", style="magenta")
        rich_table.add_column("Model 1 MAE", style="magenta")
        rich_table.add_column("Model 2 ELBO", style="green")
        rich_table.add_column("Model 2 MAE", style="green")
        for _, row in df.iterrows():
            rich_table.add_row(
                row["Dataset"],
                format_value(row["Model 1 ELBO"]),
                format_value(row["Model 1 MAE"]),
                format_value(row["Model 2 ELBO"]),
                format_value(row["Model 2 MAE"]),
            )
    return rich_table


def generate_html_rows(df: pd.DataFrame, separate: bool = False) -> str:
    rows = ""
    for _, row in df.iterrows():
        if separate:
            rows += f"""
            <tr>
              <td>{row['Dataset']}</td>
              <td>{format_value(row['Model 1'])}</td>
              <td>{format_value(row['Model 2'])}</td>
            </tr>
            """
        else:
            rows += f"""
            <tr>
              <td>{row['Dataset']}</td>
              <td>{format_value(row['Model 1 ELBO'])}</td>
              <td>{format_value(row['Model 1 MAE'])}</td>
              <td>{format_value(row['Model 2 ELBO'])}</td>
              <td>{format_value(row['Model 2 MAE'])}</td>
            </tr>
            """
    return rows


def generate_markdown_rows(df: pd.DataFrame, separate: bool = False) -> str:
    rows = ""
    for _, row in df.iterrows():
        if separate:
            rows += f"| {row['Dataset']} | {format_value(row['Model 1'])} | {format_value(row['Model 2'])} |\n"
        else:
            rows += f"| {row['Dataset']} | {format_value(row['Model 1 ELBO'])} | {format_value(row['Model 1 MAE'])} | {format_value(row['Model 2 ELBO'])} | {format_value(row['Model 2 MAE'])} |\n"
    return rows


def format_value(value: Any) -> str:
    return f"{value:.4f}" if isinstance(value, float) else str(value)


@beartype
def generate_and_save_metric_tables(
    json_data: Dict[str, Any],
    output_dir: Path = Path("reports"),
) -> List[Path]:
    """
    Generate and save tables for ELBO, MAE, and combined metrics in LaTeX, HTML, and Markdown formats.

    This function uses the `generate_metric_tables` function to create both separate and combined tables
    for different metrics and formats, then saves them to files in the specified output directory.

    Args:
        json_data (Dict[str, Any]): The input JSON data containing metrics for different datasets and models.
        output_dir (Path): The directory where the output files will be saved.

    Returns:
        None

    Raises:
        IOError: If there's an error writing to the output files.

    Example:
        >>> json_data = {
        ...     "simulated_model1-123": {"run_name": "simulated_model1-123", "-ELBO": -7.339, "MAE": 1.094},
        ...     "simulated_model2-456": {"run_name": "simulated_model2-456", "-ELBO": -7.512, "MAE": 1.123},
        ...     "pancreas_model1-789": {"run_name": "pancreas_model1-789", "-ELBO": -8.234, "MAE": 0.987},
        ...     "pancreas_model2-012": {"run_name": "pancreas_model2-012", "-ELBO": -8.123, "MAE": 0.998}
        ... }
        >>> output_dir = Path("output")
        >>> generate_and_save_metric_tables(json_data, output_dir)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    json_file = output_dir / "input_metrics.json"
    with json_file.open("w") as f:
        json.dump(json_data, f, indent=2)

    separate_table_strings, _ = generate_metric_tables(json_data, separate=True)

    combined_table_strings, _ = generate_metric_tables(
        json_data, separate=False
    )

    files_to_save = {
        "combined_table.html": combined_table_strings["html"],
        "combined_table.tex": wrap_latex_table(
            combined_table_strings["latex"],
        ),
        "combined_table.md": combined_table_strings["markdown"],
        "elbo_table.html": separate_table_strings["html"]["ELBO"],
        "elbo_table.tex": wrap_latex_table(
            separate_table_strings["latex"]["ELBO"],
        ),
        "elbo_table.md": separate_table_strings["markdown"]["ELBO"],
        "mae_table.html": separate_table_strings["html"]["MAE"],
        "mae_table.tex": wrap_latex_table(
            separate_table_strings["latex"]["MAE"],
        ),
        "mae_table.md": separate_table_strings["markdown"]["MAE"],
    }

    output_file_paths = [json_file]
    for file_name, content in files_to_save.items():
        file_path = output_dir / file_name
        try:
            with file_path.open("w") as f:
                f.write(content)
            output_file_paths.append(file_path)
        except IOError as e:
            raise IOError(f"Error writing to file {file_path}: {e}")

    print(f"All tables have been generated and saved in {output_dir}")
    return output_file_paths
