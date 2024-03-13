import os
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import scvelo as scv
import seaborn as sns
from beartype import beartype

from pyrovelocity.analyze import pareto_frontier_genes
from pyrovelocity.io import CompressedPickle
from pyrovelocity.logging import configure_logging
from pyrovelocity.plots import cluster_violin_plots
from pyrovelocity.plots import extrapolate_prediction_sample_predictive
from pyrovelocity.plots import plot_gene_ranking
from pyrovelocity.plots import plot_parameter_posterior_distributions
from pyrovelocity.plots import posterior_curve
from pyrovelocity.plots import rainbowplot
from pyrovelocity.plots import summarize_fig2_part1
from pyrovelocity.plots import summarize_fig2_part2
from pyrovelocity.utils import save_anndata_counts_to_dataframe


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
) -> Tuple[Path, Path]:
    """
    Construct summary plots for each data set and model.

    Args:
        data_model (str): string containing the data set and model identifier,
            e.g. simulated_model1
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

    Returns:
        Path: Top-level path to reports outputs for the data model combination,
            e.g. reports/simulated_model1

    Examples:
        >>> from pyrovelocity.summarize import summarize_dataset # xdoctest: +SKIP
        >>> tmp = getfixture("tmp_path") # xdoctest: +SKIP
        >>> summarize_dataset(
        ...     "simulated_model1",
        ...     "models/simulated_model1",
        ...     "models/simulated_model1/model",
        ...     "models/simulated_model1/pyrovelocity.pkl.zst",
        ...     "models/simulated_model1/postprocessed.h5ad",
        ...     "leiden",
        ...     "umap",
        ... ) # xdoctest: +SKIP
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
    rainbow_plot = data_model_reports_path / "rainbow.pdf"
    vector_field_plot = data_model_reports_path / "vector_field.pdf"
    shared_time_plot = data_model_reports_path / "shared_time.pdf"
    fig2_part1_plot = data_model_reports_path / "fig2_part1_plot.pdf"
    fig2_part2_plot = data_model_reports_path / "fig2_part2_plot.pdf"
    violin_clusters_lin = data_model_reports_path / "violin_clusters_lin.pdf"
    violin_clusters_log = data_model_reports_path / "violin_clusters_log.pdf"
    parameter_uncertainty_plot_path = (
        data_model_reports_path / "parameter_uncertainties.pdf"
    )
    t0_selection_plot = data_model_reports_path / "t0_selection.png"

    output_filenames = [
        dataframe_path,
        volcano_plot,
        rainbow_plot,
        vector_field_plot,
        shared_time_plot,
        parameter_uncertainty_plot_path,
        t0_selection_plot,
    ]
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
    # print(posterior_samples.keys())

    logger.info("Constructing t0 selection plot")
    fig, ax = plt.subplots(5, 6)
    fig.set_size_inches(26, 24)
    ax = ax.flatten()

    posterior_samples_keys_to_check = ["t0", "switching", "cell_time"]

    num_samples_list = [
        posterior_samples[key].shape[0]
        for key in posterior_samples_keys_to_check
    ]
    assert (
        len(set(num_samples_list)) == 1
    ), f"The number of samples is not equal across keys: {posterior_samples_keys_to_check}"

    num_samples = num_samples_list[0]

    for sample in range(num_samples):
        t0_sample = posterior_samples["t0"][sample]
        switching_sample = posterior_samples["switching"][sample]
        cell_time_sample = posterior_samples["cell_time"][sample]
        ax[sample].scatter(
            t0_sample.flatten(),
            2 * np.ones(t0_sample.shape[-1]),
            s=1,
            c="red",
            label="t0",
        )
        ax[sample].scatter(
            switching_sample.flatten(),
            3 * np.ones(t0_sample.shape[-1]),
            s=1,
            c="purple",
            label="switching",
        )
        ax[sample].scatter(
            cell_time_sample.flatten(),
            np.ones(cell_time_sample.shape[0]),
            s=1,
            c="blue",
            label="shared time",
        )
        ax[sample].set_ylim(-0.5, 4)
        if sample == 28:
            ax[sample].legend(loc="best", bbox_to_anchor=(0.5, 0.0, 0.5, 0.5))
        # print((t0_sample.flatten() > cell_time_sample.flatten().max()).sum())
        # print((t0_sample.flatten() < switching_sample.flatten().max()).sum())
        # print((t0_sample.flatten() > switching_sample.flatten().max()).sum())
        # for gene in adata.var_names[
        #     t0_sample.flatten() > cell_time_sample.flatten().max()
        # ]:
        #     print(gene)
    ax[-1].hist(t0_sample.flatten(), bins=200, color="red", alpha=0.3)
    ax[-1].hist(cell_time_sample.flatten(), bins=500, color="blue", alpha=0.3)

    fig.savefig(
        t0_selection_plot,
        facecolor=fig.get_facecolor(),
        bbox_inches="tight",
        edgecolor="none",
        dpi=300,
    )

    logger.info(
        "Extrapolating prediction samples for predictive posterior plots"
    )
    (
        grid_time_samples_ut,
        grid_time_samples_st,
        grid_time_samples_u0,
        grid_time_samples_s0,
        grid_time_samples_uinf,
        grid_time_samples_sinf,
        grid_time_samples_uscale,
        grid_time_samples_state,
        grid_time_samples_t0,
        grid_time_samples_dt_switching,
    ) = extrapolate_prediction_sample_predictive(
        posterior_samples["cell_time"],
        model_path,
        adata,
        grid_time_points=500,
    )
    # extrapolate_prediction_trace(data_model_conf, adata, grid_time_points=5)

    # ##################
    # # save dataframe
    # ##################

    save_anndata_counts_to_dataframe(adata, dataframe_path)

    ##################
    # generate figures
    ##################

    # vector fields
    cell_type = cell_state

    if os.path.isfile(fig2_part1_plot):
        logger.info(f"{fig2_part1_plot} exists")
    else:
        logger.info(f"Generating figure: {fig2_part1_plot}")
        summarize_fig2_part1(
            adata,
            posterior_samples["vector_field_posterior_samples"],
            posterior_samples["cell_time"],
            posterior_samples["original_spaces_embeds_magnitude"],
            posterior_samples["pca_embeds_angle"],
            posterior_samples["embeds_angle"],
            vector_field_basis,
            posterior_samples["vector_field_posterior_mean"],
            cell_type,
            fig2_part1_plot,
        )

    # gene selection
    if os.path.isfile(fig2_part2_plot):
        logger.info(f"{fig2_part2_plot} exists")
    else:
        logger.info(f"Generating figure: {fig2_part2_plot}")
        summarize_fig2_part2(
            adata,
            posterior_samples,
            basis=vector_field_basis,
            cell_state=cell_type,
            plot_name=fig2_part2_plot,
            fig=None,
        )

    # cluster violin plots
    if os.path.isfile(violin_clusters_log):
        logger.info(f"{violin_clusters_log} exists")
    else:
        logger.info(f"Generating figure: {violin_clusters_log}")
        for fig_name in [violin_clusters_lin, violin_clusters_log]:
            cluster_violin_plots(
                data_model,
                adata=adata,
                posterior_samples=posterior_samples,
                cluster_key=cell_state,
                violin_flag=True,
                pairs=None,
                show_outlier=False,
                fig_name=fig_name,
            )

    # shared time plot
    cell_time_mean = posterior_samples["cell_time"].mean(0).flatten()
    cell_time_std = posterior_samples["cell_time"].std(0).flatten()
    adata.obs["shared_time_uncertain"] = cell_time_std
    adata.obs["shared_time_mean"] = cell_time_mean

    if os.path.isfile(shared_time_plot):
        logger.info(f"{shared_time_plot} exists")
    else:
        logger.info(f"Generating figure: {shared_time_plot}")
        fig, ax = plt.subplots(1, 3)
        fig.set_size_inches(9.2, 2.6)
        ax = ax.flatten()
        ax_cb = scv.pl.scatter(
            adata,
            c="shared_time_mean",
            ax=ax[0],
            show=False,
            cmap="inferno",
            fontsize=7,
            colorbar=True,
        )
        ax_cb = scv.pl.scatter(
            adata,
            c="shared_time_uncertain",
            ax=ax[1],
            show=False,
            cmap="inferno",
            fontsize=7,
            colorbar=True,
        )
        select = adata.obs["shared_time_uncertain"] > np.quantile(
            adata.obs["shared_time_uncertain"], 0.9
        )
        sns.kdeplot(
            x=adata.obsm[f"X_{vector_field_basis}"][:, 0][select],
            y=adata.obsm[f"X_{vector_field_basis}"][:, 1][select],
            ax=ax[1],
            levels=3,
            fill=False,
        )
        ax[2].hist(cell_time_std / cell_time_mean, bins=100)
        fig.savefig(
            shared_time_plot,
            facecolor=fig.get_facecolor(),
            bbox_inches="tight",
            edgecolor="none",
            dpi=300,
        )

    volcano_data = posterior_samples["gene_ranking"]
    number_of_marker_genes = min(
        max(int(len(volcano_data) * 0.1), 4), 20, len(volcano_data)
    )
    logger.info(f"Searching for {number_of_marker_genes} marker genes")
    geneset = pareto_frontier_genes(volcano_data, number_of_marker_genes)

    logger.info("Generating posterior phase portraits")
    posterior_curve(
        adata,
        posterior_samples,
        grid_time_samples_ut,
        grid_time_samples_st,
        grid_time_samples_u0,
        grid_time_samples_s0,
        grid_time_samples_uinf,
        grid_time_samples_sinf,
        grid_time_samples_uscale,
        grid_time_samples_state,
        grid_time_samples_t0,
        grid_time_samples_dt_switching,
        geneset,
        data_model,
        posterior_phase_portraits_path,
    )

    # volcano plot
    if os.path.isfile(volcano_plot):
        logger.info(f"{volcano_plot} exists")
    else:
        logger.info(f"Generating figure: {volcano_plot}")

        volcano_data, fig = plot_gene_ranking(
            [posterior_samples],
            [adata],
            selected_genes=geneset,
            time_correlation_with="st",
            show_marginal_histograms=True,
        )

        fig.subplots_adjust(wspace=0.1, hspace=0.1)
        for ext in ["", ".png"]:
            fig.savefig(
                f"{volcano_plot}{ext}",
                facecolor=fig.get_facecolor(),
                bbox_inches="tight",
                edgecolor="none",
                dpi=300,
            )

    # parameter uncertainty
    if os.path.isfile(parameter_uncertainty_plot_path):
        logger.info(f"{parameter_uncertainty_plot_path} exists")
    else:
        logger.info(f"Generating figure: {parameter_uncertainty_plot_path}")
        plot_parameter_posterior_distributions(
            posterior_samples=posterior_samples,
            adata=adata,
            geneset=geneset,
            parameter_uncertainty_plot_path=parameter_uncertainty_plot_path,
        )

    # rainbow plot

    if os.path.isfile(rainbow_plot):
        logger.info(f"{rainbow_plot} exists")
    else:
        logger.info(f"Generating figure: {rainbow_plot}")
        fig = rainbowplot(
            volcano_data=volcano_data,
            adata=adata,
            posterior_samples=posterior_samples,
            genes=geneset,
            data=["st", "ut"],
            basis=vector_field_basis,
            cell_state=cell_state,
        )

        for ext in ["", ".png"]:
            fig.savefig(
                f"{rainbow_plot}{ext}",
                facecolor=fig.get_facecolor(),
                bbox_inches="tight",
                edgecolor="none",
                dpi=300,
            )

    # mean vector field plot
    if os.path.isfile(vector_field_plot):
        logger.info(f"{vector_field_plot} exists")
    else:
        logger.info(f"Generating figure: {vector_field_plot}")
        fig, ax = plt.subplots()

        scv.pl.velocity_embedding_grid(
            adata,
            basis=vector_field_basis,
            color=cell_state,
            title="",
            vkey="velocity_pyro",
            linewidth=1,
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

    return (data_model_reports_path, dataframe_path)
