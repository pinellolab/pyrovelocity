import errno
import os
import pickle
from logging import Logger
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scvelo as scv
import seaborn as sns
from astropy import units as u
from astropy.stats import circstd

# from scipy.stats import circvar, circstd, circmean
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib_venn import venn2
from omegaconf import DictConfig
from statannotations.Annotator import Annotator

from pyrovelocity.config import print_config_tree
from pyrovelocity.io.compressedpickle import CompressedPickle
from pyrovelocity.plot import get_posterior_sample_angle_uncertainty
from pyrovelocity.plot import plot_arrow_examples
from pyrovelocity.plot import plot_gene_ranking
from pyrovelocity.plot import plot_posterior_time
from pyrovelocity.plot import plot_vector_field_uncertain
from pyrovelocity.plot import rainbowplot
from pyrovelocity.utils import get_pylogger


def plots(
    conf: DictConfig,
    logger: Logger,
    order: list,
    log_flag: bool,
    violin_flag: bool,
    pairs: list,
    show_outlier: bool,
    fig_name: str = None,
) -> None:
    """Construct summary plots for each data set and model.

    Args:
        conf (DictConfig): OmegaConf configuration object
        logger (Logger): Python logger

    Examples:
        plots(conf, logger)
    """

    time_cov_list = []
    mag_cov_list = []
    umap_mag_cov_list = []
    umap_angle_std_list = []
    pca_angle_std_list = []
    pca_mag_cov_list = []
    pca_angle_uncertain_list = []
    umap_angle_uncertain_list = []
    names = []
    logger.info(
        f"\n\nVerifying existence of paths for:\n\n"
        f"  reports: {conf.reports.figureS3.path}\n"
    )
    Path(conf.reports.figureS3.path).mkdir(parents=True, exist_ok=True)
    print_config_tree(conf.reports.figureS3_extras, logger, ())

    for data_model in conf.train_models:
        ##################
        # load data
        ##################
        print(data_model)
        data_model_conf = conf.model_training[data_model]
        pyrovelocity_data_path = data_model_conf.pyrovelocity_data_path

        posterior_samples = CompressedPickle.load(pyrovelocity_data_path)

        print_config_tree(data_model_conf, logger, ())

        umap_cell_angles = posterior_samples["embeds_angle"] / np.pi * 180
        # umap_cell_cirsvar = circvar(umap_cell_angles, axis=0)
        umap_angle_std = circstd(umap_cell_angles * u.deg, method="angular", axis=0)
        umap_angle_std_list.append(umap_angle_std)
        umap_angle_uncertain = get_posterior_sample_angle_uncertainty(umap_cell_angles)
        umap_angle_uncertain_list.append(umap_angle_uncertain)

        pca_cell_vector = posterior_samples["pca_vector_field_posterior_samples"]
        pca_cell_magnitudes = np.sqrt((pca_cell_vector**2).sum(axis=-1))
        pca_cell_magnitudes_mean = pca_cell_magnitudes.mean(axis=-2)
        pca_cell_magnitudes_std = pca_cell_magnitudes.std(axis=-2)
        pca_cell_magnitudes_cov = pca_cell_magnitudes_std / pca_cell_magnitudes_mean
        pca_mag_cov_list.append(pca_cell_magnitudes_cov)

        pca_cell_angles = posterior_samples["pca_embeds_angle"] / np.pi * 180
        # pca_cell_cirsvar = circvar(pca_cell_angles, axis=0)
        pca_cell_cirsstd = circstd(pca_cell_angles * u.deg, method="angular", axis=0)
        pca_angle_std_list.append(pca_cell_cirsstd)
        pca_angle_uncertain = get_posterior_sample_angle_uncertainty(pca_cell_angles)
        pca_angle_uncertain_list.append(pca_angle_uncertain)

        umap_cell_magnitudes = np.sqrt(
            (posterior_samples["vector_field_posterior_samples"] ** 2).sum(axis=-1)
        )
        umap_cell_magnitudes_mean = umap_cell_magnitudes.mean(axis=-2)
        umap_cell_magnitudes_std = umap_cell_magnitudes.std(axis=-2)
        umap_cell_magnitudes_cov = umap_cell_magnitudes_std / umap_cell_magnitudes_mean

        print(posterior_samples.keys())
        cell_magnitudes = posterior_samples["original_spaces_embeds_magnitude"]
        cell_magnitudes_mean = cell_magnitudes.mean(axis=-2)
        cell_magnitudes_std = cell_magnitudes.std(axis=-2)
        cell_magnitudes_cov = cell_magnitudes_std / cell_magnitudes_mean

        cell_time_mean = posterior_samples["cell_time"].mean(0).flatten()
        cell_time_std = posterior_samples["cell_time"].std(0).flatten()
        cell_time_cov = cell_time_std / cell_time_mean
        time_cov_list.append(cell_time_cov)
        mag_cov_list.append(cell_magnitudes_cov)
        umap_mag_cov_list.append(umap_cell_magnitudes_cov)
        name = data_model.replace("_model2", "").replace("larry_", "")
        names += [name] * len(cell_time_cov)

    print(posterior_samples["pca_vector_field_posterior_samples"].shape)
    print(posterior_samples["embeds_angle"].shape)
    time_cov_list = np.hstack(time_cov_list)
    mag_cov_list = np.hstack(mag_cov_list)
    pca_mag_cov_list = np.hstack(pca_mag_cov_list)
    pca_angle_std_list = np.hstack(pca_angle_std_list)
    umap_mag_cov_list = np.hstack(umap_mag_cov_list)
    umap_angle_std_list = np.hstack(umap_angle_std_list)

    metrics_df = pd.DataFrame(
        {
            r"$CoV({\mathrm{time}})$": time_cov_list,
            r"$CoV({\mathrm{magnitude}})$": mag_cov_list,
            r"$Std({\mathrm{angle}}_{pca})$": pca_angle_std_list,
            r"$CoV({\mathrm{magnitude}}_{pca})$": pca_mag_cov_list,
            r"$Std({\mathrm{angle}}_{umap})$": umap_angle_std_list,
            r"$CoV({\mathrm{magnitude}}_{umap})$": umap_mag_cov_list,
            "dataset": names,
        }
    )

    max_values, min_values = {}, {}
    for key in metrics_df.keys()[0:6]:
        key_data = metrics_df[key]
        q1, q3 = np.percentile(key_data, (25, 75))
        max_values[key] = q3 + (q3 - q1) * 1.5
        if key_data.min() >= 0:
            min_values[key] = 0
        else:
            min_values[key] = q1 - (q3 - q1) * 1.5
    print(max_values)
    print(min_values)

    if log_flag:
        log_time_cov_list = np.log(time_cov_list)
        log_mag_cov_list = np.log(mag_cov_list)
        log_umap_mag_cov_list = np.log(umap_mag_cov_list)
        pca_angle_uncertain_list = np.hstack(pca_angle_uncertain_list)
        log_pca_mag_cov_list = np.log(pca_mag_cov_list)
        umap_angle_uncertain_list = np.hstack(umap_angle_uncertain_list)
        metrics_df = pd.DataFrame(
            {
                r"$\log(CoV({\mathrm{time}}))$": log_time_cov_list,
                r"$\log(CoV({\mathrm{magnitude}}))$": log_mag_cov_list,
                r"$CircStd({\mathrm{angle}}_{pca})$": pca_angle_uncertain_list,
                r"$\log(CoV({\mathrm{magnitude}}_{pca}))$": log_pca_mag_cov_list,
                r"$CircStd({\mathrm{angle}}_{umap})$": umap_angle_uncertain_list,
                r"$\log(CoV({\mathrm{magnitude}}_{umap}))$": log_umap_mag_cov_list,
                "dataset": names,
            }
        )

    logger.info(metrics_df.head())
    parameters = {"axes.labelsize": 25, "axes.titlesize": 35}
    plt.rcParams.update(parameters)
    fig, ax = plt.subplots(3, 2)
    ax = ax.flatten()
    fig.set_size_inches(20, 20)

    if violin_flag:
        for i in range(6):
            sns.violinplot(
                x="dataset",
                y=metrics_df.keys()[i],
                data=metrics_df,
                ax=ax[i],
                order=order,
                showfliers=show_outlier,
            )
    else:
        for i in range(6):
            sns.boxplot(
                x="dataset",
                y=metrics_df.keys()[i],
                data=metrics_df,
                ax=ax[i],
                order=order,
                showfliers=show_outlier,
            )

    if not pairs is None:
        for i in range(6):
            annotator = Annotator(
                ax[i],
                pairs,
                data=metrics_df,
                x="dataset",
                y=metrics_df.keys()[i],
                order=order,
            )
            annotator.configure(test="Mann-Whitney", text_format="star", loc="inside")
            annotator.apply_and_annotate()

    for axi in ax:
        axi.tick_params(axis="both", labelsize=20)

    fig.savefig(
        fig_name,
        facecolor=fig.get_facecolor(),
        bbox_inches="tight",
        edgecolor="none",
        dpi=300,
    )


@hydra.main(version_base="1.2", config_path="..", config_name="config.yaml")
def main(conf: DictConfig) -> None:
    """Plot results
    Args:
        config {DictConfig}: hydra configuration
    """

    logger = get_pylogger(name="PLOT", log_level=conf.base.log_level)
    print_config_tree(conf, logger, ())

    logger.info(
        f"\n\nVerifying existence of paths for:\n\n"
        f"  reports: {conf.reports.figureS3.path}\n"
    )
    Path(conf.reports.figureS3.path).mkdir(parents=True, exist_ok=True)
    confS3 = conf.reports.figureS3_extras
    print(confS3.violin_plots_pbmc_lin)
    if os.path.isfile(confS3.violin_plots_pbmc_lin):
        logger.info(
            f"\n\nFigure S3 outputs already exist:\n\n"
            f"  see contents of: {conf.reports.figureS3.path}\n"
        )
    else:
        logger.info(f"\n\nPlotting figure S3\n\n")
        all_ex = ["larry_linear", "larry_log", "pbmc_linear", "pbmc_log"]
        for ex in all_ex:  # ['larry_log']:
            if ex == "larry_linear":
                order = ["larry", "neu", "mono", "multilineage", "tips"]
                pairs = None
                log_flag = False
                fig_name = confS3.violin_plots_larry_lin
            elif ex == "larry_log":
                order = ["larry", "neu", "mono", "multilineage", "tips"]
                pairs = None
                log_flag = True
                fig_name = confS3.violin_plots_larry_log
            elif ex == "pbmc_linear":
                order = ["pancreas", "pbmc10k", "pbmc68k", "pons"]
                pairs = None  # [("pancreas", "pbmc10k")]
                log_flag = False
                fig_name = confS3.violin_plots_pbmc_lin
            elif ex == "pbmc_log":
                order = ["pancreas", "pbmc10k", "pbmc68k", "pons"]
                pairs = None  # [("pancreas", "pbmc10k")]
                log_flag = True
                fig_name = confS3.violin_plots_pbmc_log
            plots(
                conf,
                logger,
                order=order,
                log_flag=log_flag,
                violin_flag=True,
                pairs=pairs,
                show_outlier=False,
                fig_name=fig_name,
            )


if __name__ == "__main__":
    main()
