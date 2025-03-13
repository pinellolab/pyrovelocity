import os
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import scvelo as scv
from anndata import AnnData
from beartype import beartype
from beartype.typing import List
from pandas import DataFrame

from pyrovelocity.analysis.analyze import top_mae_genes
from pyrovelocity.io import CompressedPickle
from pyrovelocity.logging import configure_logging
from pyrovelocity.plots import (
    cluster_violin_plots,
    plot_gene_ranking,
    plot_gene_selection_summary,
    plot_parameter_posterior_distributions,
    plot_report,
    plot_shared_time_uncertainty,
    plot_vector_field_summary,
    posterior_curve,
    save_subfigures,
)
from pyrovelocity.plots._rainbow import rainbowplot_module as rainbowplot
from pyrovelocity.styles.colors import LARRY_CELL_TYPE_COLORS
from pyrovelocity.utils import (
    save_anndata_counts_to_dataframe,
    save_parameter_posterior_mean_dataframe,
)

__all__ = ["summarize_dataset"]

logger = configure_logging(__name__)


@beartype
def summarize_dataset(
    data_model: str,
    data_model_path: str | Path,
    model_path: str | Path,
    pyrovelocity_data_path: str | Path,
    postprocessed_data_path: str | Path,
    cell_state: str,
    vector_field_basis: str,
    reports_path: str | Path = "reports",
    enable_experimental_plots: bool = False,
    selected_genes: list[str] = [""],
) -> Tuple[Path, Path]:
    """
    Construct summary plots for each data set and model.

    Args:
        data_model (str): string containing the data set and model identifier,
            e.g. simulated_model1
        data_model_path (str | Path):
            path to a model trained on a particualar data set,
        model_path (str | Path): path to the model,
            e.g. models/simulated_model1/model
        pyrovelocity_data_path (str | Path): path to the pyrovelocity data,
            e.g. models/simulated_model1/pyrovelocity.pkl.zst
        postprocessed_data_path (str | Path): path to the trained data,
            e.g. models/simulated_model1/postprocessed.h5ad
        cell_state (str): string containing the cell state identifier,
            e.g. cell_type
        vector_field_basis (str): string containing the vector field basis
            identifier, e.g. umap
        reports_path (str | Path): path to the reports, e.g. reports
        enable_experimental_plots (bool): flag to enable experimental plots

    Returns:
        Path: Top-level path to reports outputs for the data model combination,
            e.g. reports/simulated_model1

    Examples:
        >>> # xdoctest: +SKIP
        >>> from pyrovelocity.tasks.summarize import summarize_dataset
        >>> tmp = getfixture("tmp_path")
        >>> summarize_dataset(
        ...     data_model="simulated_model1",
        ...     data_model_path="models/simulated_model1",
        ...     model_path="models/simulated_model1/model",
        ...     pyrovelocity_data_path="models/simulated_model1/pyrovelocity.pkl.zst",
        ...     postprocessed_data_path="models/simulated_model1/postprocessed.h5ad",
        ...     cell_state="leiden",
        ...     vector_field_basis="umap",
        ... )
    """

    logger.info(f"\n\nPlotting summary figure(s) in: {data_model}\n\n")

    data_model_reports_path = Path(reports_path) / Path(data_model)
    logger.info(
        f"\n\nVerifying existence of path for:\n\n"
        f"  reports: {data_model_reports_path}\n"
    )

    Path(data_model_path).mkdir(parents=True, exist_ok=True)
    Path(data_model_reports_path).mkdir(parents=True, exist_ok=True)
    posterior_phase_portraits_path = Path(data_model_reports_path) / Path(
        "posterior_phase_portraits"
    )
    Path(posterior_phase_portraits_path).mkdir(parents=True, exist_ok=True)

    dataframe_path = Path(data_model_path) / f"{data_model}_dataframe.pkl.zst"
    volcano_plot = data_model_reports_path / "volcano.pdf"
    gene_selection_rainbow_plot = (
        data_model_reports_path / "gene_selection_rainbow_plot.pdf"
    )
    vector_field_plot = data_model_reports_path / "vector_field.pdf"
    shared_time_plot = data_model_reports_path / "shared_time.pdf"
    vector_field_summary_plot = (
        data_model_reports_path / "vector_field_summary_plot.pdf"
    )
    gene_selection_summary_plot = (
        data_model_reports_path / "gene_selection_summary_plot.pdf"
    )
    parameter_uncertainty_plot = (
        data_model_reports_path / "parameter_uncertainties.pdf"
    )

    # experimental plots
    t0_selection_plot = data_model_reports_path / "t0_selection.png"
    violin_clusters_lin = data_model_reports_path / "violin_clusters_lin.pdf"
    violin_clusters_log = data_model_reports_path / "violin_clusters_log.pdf"

    output_filenames = [
        dataframe_path,
        volcano_plot,
        gene_selection_rainbow_plot,
        vector_field_plot,
        shared_time_plot,
        vector_field_summary_plot,
        gene_selection_summary_plot,
        parameter_uncertainty_plot,
    ]

    experimental_filenames = [
        t0_selection_plot,
        violin_clusters_lin,
        violin_clusters_log,
    ]

    if enable_experimental_plots:
        output_filenames.extend(experimental_filenames)

    if os.path.exists(posterior_phase_portraits_path):
        phase_portraits_exist = (
            len(os.listdir(posterior_phase_portraits_path)) > 0
        )
    if all(os.path.isfile(f) for f in output_filenames):
        logger.info(
            "\n\t"
            + "\n\t".join(str(f) for f in output_filenames)
            + "\nAll output files exist"
        )
        logger.warn("Remove output files if you want to regenerate them.")
        return (data_model_reports_path, dataframe_path)

    logger.info(f"Loading trained data: {postprocessed_data_path}")
    adata = scv.read(postprocessed_data_path)
    # gene_mapping = {"1100001G20Rik": "Wfdc21"}
    # adata = rename_anndata_genes(adata, gene_mapping)

    logger.info(f"Loading pyrovelocity data: {pyrovelocity_data_path}")
    posterior_samples = CompressedPickle.load(pyrovelocity_data_path)

    if enable_experimental_plots:
        # t0 selection plot
        if os.path.isfile(t0_selection_plot):
            logger.info(
                f"\nt0_selection plot exists: {t0_selection_plot}\n"
                f"Remove output file if you want to regenerate it.\n\n"
            )
        else:
            logger.info("Constructing t0 selection plot")
            from pyrovelocity.plots import plot_t0_selection

            plot_t0_selection(
                posterior_samples=posterior_samples,
                t0_selection_plot=t0_selection_plot,
            )

        # cluster-specific uncertainty metric violin plots
        if os.path.isfile(violin_clusters_log):
            logger.info(f"{violin_clusters_log} exists")
        else:
            logger.info(f"Generating figure: {violin_clusters_log}")
            for fig_name in [violin_clusters_lin, violin_clusters_log]:
                cluster_violin_plots(
                    data_model=data_model,
                    adata=adata,
                    posterior_samples=posterior_samples,
                    cluster_key=cell_state,
                    violin_flag=True,
                    pairs=None,
                    show_outlier=False,
                    fig_name=fig_name,
                )

        # phase portraint predictive plots
        if phase_portraits_exist:
            logger.info(
                f"\nFiles exist in posterior phase portraits path:\n"
                f"{posterior_phase_portraits_path}\n"
                f"Remove this directory or all its files if you want to regenerate them.\n\n"
            )
        else:
            logger.info("Generating posterior predictive phase portrait plots")
            posterior_curve(
                adata=adata,
                posterior_samples=posterior_samples,
                gene_set=putative_marker_genes,
                data_model=data_model,
                model_path=model_path,
                output_directory=posterior_phase_portraits_path,
            )

    # ##################
    # save dataframes
    # ##################

    save_anndata_counts_to_dataframe(adata, dataframe_path)
    gene_parameter_posteriors_path: Path = (
        data_model_reports_path / "gene_parameter_posteriors.csv"
    )
    save_parameter_posterior_mean_dataframe(
        adata,
        posterior_samples,
        gene_parameter_posteriors_path,
    )

    gene_ranking_path: Path = data_model_reports_path / "gene_ranking.csv"
    volcano_data: DataFrame = posterior_samples["gene_ranking"]
    volcano_data.to_csv(gene_ranking_path)

    ##################
    # generate figures
    ##################

    # vector field summary plot
    if os.path.isfile(vector_field_summary_plot):
        logger.info(f"{vector_field_summary_plot} exists")
    else:
        logger.info(f"Generating figure: {vector_field_summary_plot}")
        plot_vector_field_summary(
            adata=adata,
            posterior_samples=posterior_samples,
            vector_field_basis=vector_field_basis,
            plot_name=vector_field_summary_plot,
            cell_state=cell_state,
            state_color_dict=LARRY_CELL_TYPE_COLORS
            if "larry" in data_model
            else None,
            save_fig=False,
        )

    # shared time plot
    if os.path.isfile(shared_time_plot):
        logger.info(f"{shared_time_plot} exists")
    else:
        logger.info(f"Generating figure: {shared_time_plot}")

        plot_shared_time_uncertainty(
            adata=adata,
            posterior_samples=posterior_samples,
            vector_field_basis=vector_field_basis,
            shared_time_plot=shared_time_plot,
        )

    # extract putative marker genes
    logger.info(f"Searching for marker genes")
    putative_marker_genes = top_mae_genes(
        volcano_data=volcano_data,
        mae_top_percentile=100 * 24 / len(volcano_data),
        min_genes_per_bin=3,
        max_genes_per_bin=4,
    )

    # volcano plot
    if os.path.isfile(volcano_plot):
        logger.info(f"{volcano_plot} exists")
    else:
        logger.info(f"Generating figure: {volcano_plot}")

        plot_gene_ranking(
            posterior_samples=posterior_samples,
            adata=adata,
            putative_marker_genes=putative_marker_genes,
            selected_genes=selected_genes,
            time_correlation_with="st",
            show_marginal_histograms=True,
            save_volcano_plot=True,
            volcano_plot_path=volcano_plot,
            show_xy_labels=True,
        )

    # parameter uncertainty plot
    if os.path.isfile(parameter_uncertainty_plot):
        logger.info(f"{parameter_uncertainty_plot} exists")
    else:
        logger.info(f"Generating figure: {parameter_uncertainty_plot}")
        plot_parameter_posterior_distributions(
            posterior_samples=posterior_samples,
            adata=adata,
            geneset=putative_marker_genes,
            save_plot=True,
            parameter_uncertainty_plot=parameter_uncertainty_plot,
        )

    # rainbow plot for gene selection review
    if os.path.isfile(gene_selection_rainbow_plot):
        logger.info(f"{gene_selection_rainbow_plot} exists")
    else:
        logger.info(f"Generating figure: {gene_selection_rainbow_plot}")
        fig = rainbowplot(
            volcano_data=volcano_data,
            adata=adata,
            posterior_samples=posterior_samples,
            genes=putative_marker_genes,
            data=["st", "ut"],
            basis=vector_field_basis,
            cell_state=cell_state,
            save_plot=True,
            rainbow_plot_path=gene_selection_rainbow_plot,
        )

    # gene selection summary plot
    if os.path.isfile(gene_selection_summary_plot):
        logger.info(f"{gene_selection_summary_plot} exists")
    else:
        logger.info(f"Generating figure: {gene_selection_summary_plot}")

        plot_report(
            adata=adata,
            posterior_samples=posterior_samples,
            volcano_data=volcano_data,
            putative_marker_genes=putative_marker_genes,
            selected_genes=selected_genes,
            vector_field_basis=vector_field_basis,
            cell_state=cell_state,
            state_color_dict=LARRY_CELL_TYPE_COLORS
            if "larry" in data_model
            else None,
            report_file_path=gene_selection_summary_plot,
            figure_file_path=f"{gene_selection_summary_plot}.dill.zst",
        )

    # mean vector field plot
    if os.path.isfile(vector_field_plot):
        logger.info(f"{vector_field_plot} exists")
    else:
        logger.info(f"Generating figure: {vector_field_plot}")
        fig, ax = plt.subplots()
        ax.axis("off")

        scv.pl.velocity_embedding_grid(
            adata,
            basis=vector_field_basis,
            color=cell_state,
            title="",
            vkey="velocity_pyro",
            s=1,
            linewidth=0.5,
            ax=ax,
            show=False,
            legend_loc="right margin",
            density=0.4,
            scale=0.2,
            arrow_size=2,
            arrow_length=2,
            arrow_color="black",
        )
        for ext in ["", ".png"]:
            fig.savefig(
                f"{vector_field_plot}{ext}",
                facecolor=fig.get_facecolor(),
                bbox_inches="tight",
                edgecolor="none",
                dpi=300,
            )
        plt.close(fig)

    return (data_model_reports_path, dataframe_path)
