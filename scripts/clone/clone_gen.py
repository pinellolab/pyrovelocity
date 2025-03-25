#!/usr/bin/env python
import logging
import os
from pathlib import Path

import numpy as np
import rich_click as click
from anndata import AnnData
from beartype import beartype
from beartype.typing import Dict, List, Optional
from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

from pyrovelocity.io.datasets import larry_mono, larry_neu
from pyrovelocity.plots._trajectory import get_clone_trajectory

click.rich_click.SHOW_ARGUMENTS = True
click.rich_click.USE_MARKDOWN = True


def configure_logging(logger_name: str = "clone_gen") -> logging.Logger:
    """Configure rich logging with custom theme."""
    console_theme = Theme(
        {
            "logging.level.info": "dim cyan",
            "logging.level.warning": "magenta",
            "logging.level.error": "bold red",
            "logging.level.debug": "green",
        }
    )
    console = Console(theme=console_theme)
    rich_handler = RichHandler(
        console=console,
        rich_tracebacks=True,
        show_time=True,
        show_level=True,
        show_path=False,
        markup=True,
        log_time_format="[%X]",
    )
    valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    if log_level not in valid_log_levels:
        log_level = "INFO"

    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[rich_handler],
    )
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)
    return logger


logger = configure_logging()


@beartype
def generate_clone_trajectory(
    adata: AnnData,
    average_start_point: bool = True,
    times: List[int] = [2, 4, 6],
    clone_num: Optional[int] = None,
    fix_nans: bool = True,
) -> AnnData:
    """Generate clone trajectory data from AnnData object.

    Args:
        adata: The input AnnData object
        average_start_point: Whether to average the start point
        times: List of time points to consider
        clone_num: Maximum number of clones to process
        fix_nans: Whether to replace NaN values with zeros

    Returns:
        AnnData object with clone trajectory information
    """
    logger.info(f"Generating clone trajectory for dataset with {adata.n_obs} cells")
    adata_clone = get_clone_trajectory(
        adata, average_start_point=average_start_point, 
        times=times, clone_num=clone_num
    )
    
    if fix_nans and "clone_vector_emb" in adata_clone.obsm:
        nan_count = np.isnan(adata_clone.obsm["clone_vector_emb"]).sum()
        if nan_count > 0:
            logger.info(f"Fixing {nan_count} NaN values in clone_vector_emb")
            adata_clone.obsm["clone_vector_emb"][
                np.isnan(adata_clone.obsm["clone_vector_emb"])
            ] = 0
    
    return adata_clone


@beartype
def generate_all_clone_trajectories(
    output_dir: Path,
    mono_path: Optional[str] = None,
    neu_path: Optional[str] = None,
    output_names: Dict[str, str] = None,
) -> Dict[str, Path]:
    """Pre-compute and cache clone trajectories for different lineage datasets.

    Args:
        output_dir: Directory to save generated trajectory files
        mono_path: Optional custom path for mono dataset
        neu_path: Optional custom path for neu dataset
        output_names: Optional custom output filenames

    Returns:
        Dictionary mapping dataset names to file paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if output_names is None:
        output_names = {
            "mono": "larry_mono_clone_trajectory.h5ad",
            "neu": "larry_neu_clone_trajectory.h5ad",
            "multilineage": "larry_multilineage_clone_trajectory.h5ad",
        }
    
    logger.info(f"Loading monocyte lineage data from {'custom path' if mono_path else 'default path'}")
    mono_adata = larry_mono(mono_path) if mono_path else larry_mono()
    mono_clone = generate_clone_trajectory(mono_adata)
    mono_clone_path = output_dir / output_names["mono"]
    logger.info(f"Writing monocyte clone trajectory to {mono_clone_path}")
    mono_clone.write_h5ad(mono_clone_path)
    
    logger.info(f"Loading neutrophil lineage data from {'custom path' if neu_path else 'default path'}")
    neu_adata = larry_neu(neu_path) if neu_path else larry_neu()
    neu_clone = generate_clone_trajectory(neu_adata)
    neu_clone_path = output_dir / output_names["neu"]
    logger.info(f"Writing neutrophil clone trajectory to {neu_clone_path}")
    neu_clone.write_h5ad(neu_clone_path)
    
    logger.info("Creating concatenated multilineage clone trajectory")
    multi_clone = mono_clone.concatenate(neu_clone)
    multi_clone_path = output_dir / output_names["multilineage"]
    logger.info(f"Writing multilineage clone trajectory to {multi_clone_path}")
    multi_clone.write_h5ad(multi_clone_path)
    
    logger.info("All clone trajectories generated successfully")
    
    return {
        "mono": mono_clone_path,
        "neu": neu_clone_path,
        "multilineage": multi_clone_path
    }


@click.group(
    invoke_without_command=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.pass_context
def cli(ctx):
    """
    # clone_gen
    _**clone_gen**_ generates pre-computed clone trajectory files for PyroVelocity.
    
    This tool downloads LARRY dataset samples and computes clone trajectories that 
    can be later used directly in the plot_lineage_fate_correlation function.
    
    Pass -h or --help to each command group listed below for detailed help.
    """
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@cli.command("generate")
@click.option(
    "-o",
    "--output-dir",
    "output_dir",
    default="data/external",
    help="Output directory for the generated trajectories.",
    show_default=True,
    type=click.Path(),
)
@click.option(
    "--mono-path",
    "mono_path",
    default=None,
    help="Optional custom path for larry_mono dataset.",
    type=click.Path(exists=False),
)
@click.option(
    "--neu-path",
    "neu_path",
    default=None,
    help="Optional custom path for larry_neu dataset.",
    type=click.Path(exists=False),
)
def generate_trajectories(output_dir, mono_path, neu_path):
    """
    # clone_gen generate
    
    Generate pre-computed clone trajectories for the LARRY datasets.
    
    This command:
    1. Downloads the larry_mono and larry_neu datasets if needed
    2. Computes clone trajectories using get_clone_trajectory
    3. Creates a concatenated multilineage trajectory
    4. Saves all trajectories to h5ad files
    
    These pre-computed trajectories can then be used with plot_lineage_fate_correlation
    to generate consistent visualizations without redundant computation.
    """
    output_dir_path = Path(output_dir)
    result_paths = generate_all_clone_trajectories(
        output_dir=output_dir_path,
        mono_path=mono_path,
        neu_path=neu_path,
    )
    
    logger.info("Clone trajectories generated and saved to:")
    for name, path in result_paths.items():
        logger.info(f"  - {name}: {path}")
    
    logger.info("\nYou can now create functions in pyrovelocity.io.datasets to load these files:")
    logger.info("""
@beartype
def larry_mono_clone_trajectory(
    file_path: str | Path = "data/external/larry_mono_clone_trajectory.h5ad",
) -> anndata._core.anndata.AnnData:
    \"\"\"
    Pre-computed clone trajectory data for the LARRY monocyte lineage.
    
    This contains the output of get_clone_trajectory applied to the larry_mono dataset.
    
    Returns:
        AnnData object with clone trajectory information
    \"\"\"
    url = "https://storage.googleapis.com/pyrovelocity/data/larry_mono_clone_trajectory.h5ad"
    adata = sc.read(file_path, backup_url=url, sparse=True, cache=True)
    return adata
    """)


@cli.command("examine")
@click.argument(
    "trajectory_path",
    type=click.Path(exists=True),
)
def examine_trajectory(trajectory_path):
    """
    # clone_gen examine
    
    Examine a generated clone trajectory file and print information about its contents.
    
    ## arguments
    - `TRAJECTORY_PATH`: Path to the clone trajectory file to examine
    """
    import scanpy as sc
    
    try:
        adata = sc.read(trajectory_path)
        logger.info(f"Successfully loaded file: {trajectory_path}")
        logger.info(f"AnnData object with n_obs × n_vars = {adata.n_obs} × {adata.n_vars}")
        
        if "state_info" in adata.obs:
            centroid_count = sum(adata.obs["state_info"] == "Centroid")
            logger.info(f"Contains {centroid_count} centroid cells")
        
        if "clone_vector_emb" in adata.obsm:
            logger.info("Contains clone_vector_emb in obsm")
            nan_count = np.isnan(adata.obsm["clone_vector_emb"]).sum()
            if nan_count > 0:
                logger.warning(f"Contains {nan_count} NaN values in clone_vector_emb")
            else:
                logger.info("No NaN values found in clone_vector_emb")
        else:
            logger.error("Missing clone_vector_emb in obsm")
        
        logger.info("\nAvailable keys:")
        logger.info(f"  obs keys: {list(adata.obs.keys())}")
        logger.info(f"  var keys: {list(adata.var.keys())}")
        logger.info(f"  obsm keys: {list(adata.obsm.keys())}")
        
    except Exception as e:
        logger.error(f"Error examining trajectory file: {e}")


if __name__ == "__main__":
    cli()
