import argparse
from pathlib import Path
from typing import Sequence
from typing import Union

import rich.console
import rich.pretty
import rich.syntax
import rich.tree
from hydra import compose
from hydra import initialize
from hydra.core.config_store import ConfigStore
from hydra.core.global_hydra import GlobalHydra
from hydra_zen import make_config
from hydra_zen import make_custom_builds_fn
from omegaconf import DictConfig
from omegaconf import ListConfig
from omegaconf import OmegaConf
from pytorch_lightning.utilities import rank_zero_only

from pyrovelocity.api import train_model
from pyrovelocity.utils import get_pylogger


def hydra_zen_configure():
    """
    Use hydra-zen to generate a configuration for hydra's config store.
    Generates a hydra-zen configuration for the pyrovelocity pipeline,
    which includes dataset, model training, and reporting configurations.

    This function defines helper functions for creating dataset, model,
    and reports configurations, and uses these functions to generate the
    hydra-zen configuration.

    Returns:
        Config : Type[DataClass]
        See the documentation for hydra_zen.make_config for more information.
    """
    # define helper functions
    pbuilds = make_custom_builds_fn(zen_partial=True, populate_full_signature=True)

    def create_dataset_config(
        name, dl_root, data_file, rel_path, url, process_method, process_args
    ):
        return dict(
            data_file=data_file,
            dl_root=dl_root,
            dl_path="${.dl_root}/${.data_file}",
            rel_path=rel_path,
            url=url,
            derived=dict(
                process_method=process_method,
                process_args=process_args,
                rel_path="${data_external.processed_path}/" + f"{name}_processed.h5ad",
            ),
        )

    def create_model_config(
        source, name, model_suffix, vector_field_basis, **custom_training_parameters
    ):
        return dict(
            path="${paths.models}/" + f"{name}_model{model_suffix}",
            model_path="${.path}/model",
            input_data_path="${data_external."
            + f"{source}.{name}"
            + ".derived.rel_path}",
            trained_data_path="${.path}/trained.h5ad",
            pyrovelocity_data_path="${.path}/pyrovelocity.pkl",
            metrics_path="${.path}/metrics.json",
            run_info_path="${.path}/run_info.json",
            vector_field_parameters=dict(basis=vector_field_basis),
            training_parameters=pbuilds(
                train_model,
                loss_plot_path="${..path}/loss_plot.png",
                patient_improve=0.0001,
                patient_init=45,
                **custom_training_parameters,
            ),
        )

    def create_reports_config(model_name: str, model_number: int):
        path = f"{paths['reports']}/{model_name}_model{model_number}"
        return dict(
            path=path,
            shared_time_plot=f"{path}/shared_time.pdf",
            volcano_plot=f"{path}/volcano.pdf",
            rainbow_plot=f"{path}/rainbow.pdf",
            uncertainty_param_plot=f"{path}/param_uncertainties.pdf",
            vector_field_plot=f"{path}/vector_field.pdf",
            biomarker_selection_plot=f"{path}/markers_selection_scatterplot.tif",
            biomarker_phaseportrait_plot=f"{path}/markers_phaseportrait.pdf",
        )

    # define configuration
    base = dict(log_level="INFO")

    paths = dict(
        data="data",
        models="models",
        reports="reports",
    )

    config = make_config(
        base=base,
        paths=paths,
        data_external=dict(
            root_path="${paths.data}/external",
            processed_path="${paths.data}/processed",
            sources=["simulate", "velocyto", "scvelo", "pyrovelocity"],
            simulate=dict(
                download=["medium"],
                process=["medium"],
                sources=dict(
                    gcs_root_url="https://storage.googleapis.com/pyrovelocity/data"
                ),
                medium=create_dataset_config(
                    "simulated_medium",
                    dl_root="${data_external.root_path}",
                    data_file="simulated_medium.h5ad",
                    rel_path="${data_external.root_path}/simulated_medium.h5ad",
                    url="${data_external.simulate.sources.gcs_root_url}/simulated_medium.h5ad",
                    process_method="load_data",
                    process_args=dict(),
                ),
            ),
            velocyto=dict(
                download=["pons"],
                process=["pons"],
                sources=dict(
                    gcs_root_url="https://storage.googleapis.com/pyrovelocity/data"
                ),
                pons=create_dataset_config(
                    "pons",
                    dl_root="${data_external.root_path}",
                    data_file="oligo_lite.h5ad",
                    rel_path="${data_external.root_path}/oligo_lite.h5ad",
                    url="${data_external.velocyto.sources.gcs_root_url}/oligo_lite.h5ad",
                    process_method="load_data",
                    process_args=dict(),
                ),
            ),
            scvelo=dict(
                download=["pancreas", "pbmc68k"],
                process=["pancreas", "pbmc68k"],
                sources=dict(
                    figshare_root_url="https://ndownloader.figshare.com/files",
                    scvelo_root_url="https://github.com/theislab/scvelo_notebooks/raw/master",
                ),
                pancreas=create_dataset_config(
                    "pancreas",
                    dl_root="data/Pancreas",
                    data_file="endocrinogenesis_day15.h5ad",
                    rel_path="${data_external.root_path}/endocrinogenesis_day15.h5ad",
                    url="${data_external.scvelo.sources.scvelo_root_url}/data/Pancreas/endocrinogenesis_day15.h5ad",
                    process_method="load_data",
                    process_args=dict(process_cytotrace=True),
                ),
                pbmc68k=create_dataset_config(
                    "pbmc68k",
                    dl_root="data/PBMC",
                    data_file="pbmc68k.h5ad",
                    rel_path="${data_external.root_path}/pbmc68k.h5ad",
                    url="${data_external.scvelo.sources.figshare_root_url}/27686886",
                    process_method="load_data",
                    process_args=dict(),
                ),
            ),
            pyrovelocity=dict(
                download=["larry"],
                process=[],
                sources=dict(
                    figshare_root_url="https://ndownloader.figshare.com/files"
                ),
                larry=create_dataset_config(
                    "larry",
                    dl_root="${data_external.root_path}",
                    data_file="larry.h5ad",
                    rel_path="${data_external.root_path}/larry.h5ad",
                    url="${data_external.pyrovelocity.sources.figshare_root_url}/37028569",
                    process_method="",
                    process_args=dict(),
                ),
            ),
        ),
        model_training=dict(
            train=[
                "simulate_model2",
                "pancreas_model1",
                "pancreas_model2",
                "pbmc68k_model1",
                "pbmc68k_model2",
                "pons_model1",
                "pons_model2",
            ],
            simulate_model2=create_model_config(
                "simulate", "medium", 2, "umap", max_epochs=4000, offset=True
            ),
            pancreas_model1=create_model_config(
                "scvelo",
                "pancreas",
                1,
                "umap",
                guide_type="auto_t0_constraint",
                max_epochs=2000,
            ),
            pancreas_model2=create_model_config(
                "scvelo",
                "pancreas",
                2,
                "umap",
                offset=True,
                max_epochs=2000,
            ),
            pbmc68k_model1=create_model_config(
                "scvelo",
                "pbmc68k",
                1,
                "tsne",
                guide_type="auto_t0_constraint",
                cell_state="celltype",
                max_epochs=2000,
            ),
            pbmc68k_model2=create_model_config(
                "scvelo",
                "pbmc68k",
                2,
                "tsne",
                cell_state="celltype",
                offset=True,
                max_epochs=2000,
            ),
            pons_model1=create_model_config(
                "velocyto",
                "pons",
                1,
                "umap",
                guide_type="auto_t0_constraint",
                cell_state="celltype",
                max_epochs=2000,
            ),
            pons_model2=create_model_config(
                "velocyto",
                "pons",
                2,
                "umap",
                cell_state="celltype",
                offset=True,
                max_epochs=2000,
            ),
        ),
        reports=dict(
            model_summary=dict(
                summarize=[
                    "pancreas_model1",
                    "pancreas_model2",
                    "pbmc68k_model1",
                    "pbmc68k_model2",
                    "pons_model1",
                    "pons_model2",
                ],
                pancreas_model1=create_reports_config("pancreas", 1),
                pancreas_model2=create_reports_config("pancreas", 2),
                pbmc68k_model1=create_reports_config("pbmc68k", 1),
                pbmc68k_model2=create_reports_config("pbmc68k", 2),
                pons_model1=create_reports_config("pons", 1),
                pons_model2=create_reports_config("pons", 2),
            ),
            figure2=dict(
                tag="fig2",
                path="${paths.reports}/${.tag}",
                tif_path="${.path}/${.tag}_raw_gene_selection_model1.tif",
                svg_path="${.path}/${.tag}_raw_gene_selection_model1.svg",
                biomarker_selection_path="${.path}/${.tag}_markers_selection_scatterplot.tif",
                biomarker_phaseportrait_path="${.path}/${.tag}_markers_phaseportrait.pdf",
                uncertainty_param_plot_path="${.path}/${.tag}_param_uncertainties.pdf",
                uncertainty_magnitude_plot_path="${.path}/${.tag}_magnitude_uncertainties.pdf",
                pancreas_model1=dict(
                    shared_time_plot="${..path}/${..tag}_pancreas_shared_time.pdf",
                    volcano_plot="${..path}/${..tag}_pancreas_volcano.pdf",
                    rainbow_plot="${..path}/${..tag}_pancreas_rainbow.pdf",
                    vector_field_plot="${..path}/${..tag}_pancreas_vector_field.pdf",
                ),
            ),
            figureS3=dict(
                tag="figS3",
                path="${paths.reports}/${.tag}",
                tif_path="${.path}/${.tag}_raw_gene_selection_model2.tif",
                svg_path="${.path}/${.tag}_raw_gene_selection_model2.svg",
            ),
        ),
    )
    return config


def initialize_hydra_config() -> DictConfig:
    """Initialize Hydra configuration for PyroVelocity pipeline.

    Returns:
        DictConfig: A Hydra configuration object containing dataset, model training,
        and reporting configurations for the pyrovelocity pipeline.

    Example:
        >>> cfg = initialize_hydra_config()
    """
    GlobalHydra.instance().clear()
    config = hydra_zen_configure()
    initialize(version_base="1.2", config_path=None)
    cs = ConfigStore.instance()
    cs.store(name="config", node=config)
    return compose(config_name="config")


def config_setup(config_path: str) -> Union[DictConfig, ListConfig]:
    """Convert template into concrete configuration file.
    Args:
        config_path {Text}: path to config
    """

    logger = get_pylogger(name="CONF")

    template_config_path = config_path.replace("config.yaml", "template-config.yaml")
    conf = OmegaConf.load(template_config_path)
    with open(config_path, "w") as conf_file:
        OmegaConf.save(config=conf, f=conf_file, resolve=True)

    conf = OmegaConf.load(config_path)
    print_config_tree(conf, logger, ())
    return conf


@rank_zero_only
def print_config_tree(
    cfg: DictConfig,
    logger,
    print_order: Sequence[str] = (
        "datamodule",
        "model",
        "callbacks",
        "logger",
        "trainer",
        "paths",
        "extras",
    ),
    resolve: bool = False,
    save_to_file: bool = False,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.
        print_order (Sequence[str], optional): Determines in what order config components are printed.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
        save_to_file (bool, optional): Whether to export config to the hydra output folder.
    """

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    queue = []

    # add fields from `print_order` to queue
    for field in print_order:
        queue.append(field) if field in cfg else logger.warning(
            f"Field '{field}' not found in config. Skipping '{field}' config printing..."
        )

    # add all the other fields to queue (not specified in `print_order`)
    for field in cfg:
        if field not in queue:
            queue.append(field)

    # generate config tree from queue
    for field in queue:
        branch = tree.add(field, style=style, guide_style=style)

        config_group = cfg[field]
        if isinstance(config_group, DictConfig):
            branch_content = OmegaConf.to_yaml(config_group, resolve=resolve)
        else:
            branch_content = str(config_group)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    # print config tree
    rich.print(tree)

    # save config tree to file
    if save_to_file:
        with open(Path(cfg.paths.output, "config_tree.log"), "w") as file:
            rich.print(tree, file=file)


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args = args_parser.parse_args()

    config_setup(config_path=args.config)
