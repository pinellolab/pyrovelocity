import shutil
import tempfile
from pathlib import Path
from typing import Tuple

import nbformat
import papermill as pm
import rich_click as click
from flytekit import task
from flytekit import workflow
from jupytext import read
from jupytext import write
from nbconvert import HTMLExporter
from nbconvert import PDFExporter
from nbconvert.writers import FilesWriter

from pyrovelocity.logging import configure_logging


__all__ = ["notebook_workflow"]

logger = configure_logging(__name__)

click.rich_click.SHOW_ARGUMENTS = True
click.rich_click.USE_MARKDOWN = True

CACHE_FLAG = True
CACHE_VERSION = "0.1.dev0"


@task(cache=CACHE_FLAG, cache_version=CACHE_VERSION)
def convert_py_to_ipynb(py_file_path: str) -> str:
    if CACHE_FLAG:
        logger.warning(
            f"\nCache flag enabled: {CACHE_FLAG}\n"
            f"Cache version: {CACHE_VERSION}\n"
        )

    logger.info(f"Converting {py_file_path} to ipynb format.")

    notebook = read(py_file_path)
    ipynb_file_path = py_file_path.replace(".py", ".ipynb")
    write(notebook, ipynb_file_path)

    if not Path(ipynb_file_path).exists():
        raise FileNotFoundError(
            "Conversion failed or output file does not exist."
        )
    else:
        logger.info(f"Conversion successful. Output: {ipynb_file_path}")

    return ipynb_file_path


@task(cache=CACHE_FLAG, cache_version=CACHE_VERSION)
def execute_notebook(ipynb_file_path: str, test_mode: bool) -> str:
    executed_ipynb_file_path = ipynb_file_path.replace(
        ".ipynb", "_executed.ipynb"
    )

    logger.info(
        f"Executing {ipynb_file_path} and saving to {executed_ipynb_file_path}."
    )
    if test_mode:
        logger.warning(f"Test mode enabled: {test_mode}")

    pm.execute_notebook(
        ipynb_file_path,
        executed_ipynb_file_path,
        parameters=dict(TEST_MODE=test_mode),
    )

    executed_ipynb_file = Path(executed_ipynb_file_path)
    if not executed_ipynb_file.exists():
        raise FileNotFoundError(
            "Execution failed or output file {executed_ipynb_file_path} does not exist."
        )
    else:
        logger.info(f"Execution successful. Output: {executed_ipynb_file_path}")

    return executed_ipynb_file_path


@task(cache=CACHE_FLAG, cache_version=CACHE_VERSION)
def convert_ipynb_to_html_and_pdf(
    executed_ipynb_file_path: str
) -> Tuple[str, str]:
    logger.info(
        f"Converting {executed_ipynb_file_path} to html and pdf formats."
    )
    notebook_node = nbformat.read(executed_ipynb_file_path, as_version=4)

    html_exporter = HTMLExporter()
    pdf_exporter = PDFExporter()

    (body_html, resources_html) = html_exporter.from_notebook_node(
        notebook_node
    )
    del resources_html
    html_file_path = executed_ipynb_file_path.replace(".ipynb", ".html")
    with open(html_file_path, "w") as html_file:
        html_file.write(body_html)

    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_exporter = PDFExporter()
        pdf_exporter.output_files_dir = tmpdir
        (body_pdf, resources_pdf) = pdf_exporter.from_notebook_node(
            notebook_node
        )

        pdf_file_name = Path(executed_ipynb_file_path).stem + ".pdf"
        temp_pdf_file_path = Path(tmpdir) / pdf_file_name

        pdf_writer = FilesWriter(build_directory=str(tmpdir))
        pdf_writer.write(
            body_pdf, resources_pdf, notebook_name=temp_pdf_file_path.stem
        )

        final_pdf_file_path = Path(
            executed_ipynb_file_path.replace(".ipynb", ".pdf")
        )
        shutil.copy(temp_pdf_file_path, final_pdf_file_path)

    if not Path(html_file_path).exists():
        raise FileNotFoundError(
            f"Conversion failed or output file {html_file_path} does not exist."
        )
    if not final_pdf_file_path.exists():
        raise FileNotFoundError(
            "Conversion failed or output file {final_pdf_file_path} does not exist."
        )
    logger.info(
        f"Conversion successful. Outputs:\n"
        f"HTML File: {html_file_path}\n"
        f"PDF File: {final_pdf_file_path}\n"
    )

    return html_file_path, str(final_pdf_file_path)


@workflow
def notebook_workflow(
    py_file_path: str = "notebook.py", test_mode: bool = True
) -> Tuple[str, str, str]:
    """
    This workflow converts a python percent-formatted notebook script into an
    executed ipython notebook and derived html and pdf formats. This can be
    tested locally with a command similar to:

    ```bash
    pyrovelocity \
        execution_context=local_shell \
        entity_config=notebooks_notebook_workflow \
        entity_config.inputs._args_.0.py_file_path=scripts/examples/literate_template.py \
        entity_config.inputs._args_.0.test_mode=true
    ```

    Alternatively, use the `execute` command in the CLI to run the workflow
    similar to:
    
    ```bash
    python -m pyrovelocity.workflows.notebooks execute \
        --py-file-path scripts/examples/literate_template.py \
        --test-mode
    ```

    See `python -m pyrovelocity.workflows.notebooks execute --help` for more
    details.

    Args:
        py_file_path (str, optional): Path to notebook script. Defaults to "notebook.py".
        test_mode (bool, optional): Flag indicating execution in test mode. Defaults to True.

    Returns:
        Tuple[str, str, str]: Paths to executed ipynb, html, and pdf files.
    """
    ipynb_file_path = convert_py_to_ipynb(py_file_path=py_file_path)
    executed_ipynb_file_path = execute_notebook(
        ipynb_file_path=ipynb_file_path, test_mode=test_mode
    )
    html_file_path, pdf_file_path = convert_ipynb_to_html_and_pdf(
        executed_ipynb_file_path=executed_ipynb_file_path
    )
    return executed_ipynb_file_path, html_file_path, pdf_file_path


@click.group(
    invoke_without_command=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.pass_context
def cli(ctx):
    """
    # notebooks workflow
    The _**notebooks workflow**_ converts a python
    [percent formatted](https://jupytext.readthedocs.io/en/latest/formats-scripts.html)
    notebook script into ipython notebook format, executes the notebook with
    papermill and converts it to html and pdf format.

    Pass -h or --help to each command group listed below for detailed help.

    [notebook percent format](https://jupytext.readthedocs.io/en/latest/formats-scripts.html)
    [papermill](https://github.com/nteract/papermill)
    [nbformat](https://github.com/jupyter/nbformat)
    [nbconvert](https://github.com/jupyter/nbconvert)
    """
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@cli.command("execute")
@click.option(
    "--py-file-path",
    type=str,
    help="Path to the percent-formatted .py file (e.g. notebook.py) to be processed",
    required=True,
)
@click.option(
    "--test-mode",
    is_flag=True,
    default=False,
    help="Flag to run the notebook in test mode.",
)
def cli_execute_notebook(py_file_path: str, test_mode: bool):
    """
    # execute
    Execute the notebooks workflow with the provided python file path.

    The python file should be in
    [percent format](https://jupytext.readthedocs.io/en/latest/formats-scripts.html)
    and contain markdown cells for literate programming.
    """
    result = notebook_workflow(py_file_path=py_file_path, test_mode=test_mode)
    click.echo(
        f"Workflow completed with outputs:\n\n"
        f"Executed Notebook: {result[0]}\n"
        f"HTML File: {result[1]}\n"
        f"PDF File: {result[2]}\n"
    )


if __name__ == "__main__":
    cli()
