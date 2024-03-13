from pathlib import Path

import anndata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scvelo as scv
import seaborn as sns
from astropy import units as u
from astropy.stats import circstd as acircstd

# from statannotations.Annotator import Annotator
from pyrovelocity.logging import configure_logging


__all__ = [
    "cluster_violin_plots",
    "get_posterior_sample_angle_uncertainty",
    "plot_state_uncertainty",
]

logger = configure_logging(__name__)


def cluster_violin_plots(
    data_model: str,
    adata: anndata.AnnData,
    posterior_samples,
    cluster_key: str,
    violin_flag: bool,
    pairs: list,
    show_outlier: bool,
    fig_name: str | Path = None,
) -> None:
    """Construct violin plots for each cluster."""

    time_cov_list = []
    mag_cov_list = []
    umap_mag_cov_list = []
    umap_angle_std_list = []
    pca_angle_std_list = []
    pca_mag_cov_list = []
    pca_angle_uncertain_list = []
    umap_angle_uncertain_list = []
    names = []
    # print(adata)
    # print(adata.obs[cluster_key])
    fig_name = str(fig_name)

    # get cluster order
    cluster_time_list = []
    clusters = adata.obs[cluster_key].values.categories
    for cluster in clusters:
        adata_cluster = adata[adata.obs[cluster_key] == cluster]
        cluster_time = adata_cluster.obs["velocity_pseudotime"].mean()
        cluster_time_list.append(cluster_time)
    # print(cluster_time_list)
    sorted_cluster_id = sorted(
        range(len(cluster_time_list)),
        key=lambda k: cluster_time_list[k],
        reverse=False,
    )
    order = [str(category) for category in clusters[sorted_cluster_id]]

    umap_cell_angles = posterior_samples["embeds_angle"] / np.pi * 180
    # umap_cell_cirsvar = circvar(umap_cell_angles, axis=0)
    umap_angle_std = acircstd(
        umap_cell_angles * u.deg, method="angular", axis=0
    )
    umap_angle_std_list.append(umap_angle_std)
    umap_angle_uncertain = get_posterior_sample_angle_uncertainty(
        umap_cell_angles
    )
    umap_angle_uncertain_list.append(umap_angle_uncertain)

    pca_cell_vector = posterior_samples["pca_vector_field_posterior_samples"]
    pca_cell_magnitudes = np.sqrt((pca_cell_vector**2).sum(axis=-1))
    pca_cell_magnitudes_mean = pca_cell_magnitudes.mean(axis=-2)
    pca_cell_magnitudes_std = pca_cell_magnitudes.std(axis=-2)
    pca_cell_magnitudes_cov = pca_cell_magnitudes_std / pca_cell_magnitudes_mean
    pca_mag_cov_list.append(pca_cell_magnitudes_cov)

    pca_cell_angles = posterior_samples["pca_embeds_angle"] / np.pi * 180
    # pca_cell_cirsvar = circvar(pca_cell_angles, axis=0)
    pca_cell_cirsstd = acircstd(
        pca_cell_angles * u.deg, method="angular", axis=0
    )
    pca_angle_std_list.append(pca_cell_cirsstd)
    pca_angle_uncertain = get_posterior_sample_angle_uncertainty(
        pca_cell_angles
    )
    pca_angle_uncertain_list.append(pca_angle_uncertain)

    umap_cell_magnitudes = np.sqrt(
        (posterior_samples["vector_field_posterior_samples"] ** 2).sum(axis=-1)
    )
    umap_cell_magnitudes_mean = umap_cell_magnitudes.mean(axis=-2)
    umap_cell_magnitudes_std = umap_cell_magnitudes.std(axis=-2)
    umap_cell_magnitudes_cov = (
        umap_cell_magnitudes_std / umap_cell_magnitudes_mean
    )

    # print(posterior_samples.keys())
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
    name = list(adata.obs[cluster_key])
    names += name

    can_cast_all = all(name.isdigit() for name in names)

    if can_cast_all:
        processed_names = [f"c_{int(name)}" for name in names]
        order = [f"c_{int_name}" for int_name in order]
        logger.warning(
            "Converted integer index to string 'c_int' for cluster names"
        )
    else:
        processed_names = names

    # print(posterior_samples["pca_vector_field_posterior_samples"].shape)
    # print(posterior_samples["embeds_angle"].shape)
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
            "dataset": processed_names,
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
    # print(max_values)
    # print(min_values)

    if "log" in fig_name:
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
                "dataset": processed_names,
            }
        )

    logger.debug(metrics_df.head())
    metrics_df["dataset"] = metrics_df["dataset"].astype("category")
    parameters = {"axes.labelsize": 25, "axes.titlesize": 35}
    plt.rcParams.update(parameters)
    fig, ax = plt.subplots(6, 1)
    ax = ax.flatten()
    fig.set_size_inches(20, 60)
    ax[0].set_title(f"{data_model.split('_')[0]}")
    # ax[0].set_ylim(-5.5, -2)

    if violin_flag:
        for i in range(6):
            sns.violinplot(
                x="dataset",
                y=metrics_df.keys()[i],
                data=metrics_df,
                ax=ax[i],
                order=order,
                # showfliers=show_outlier,
            )
    else:
        for i in range(6):
            sns.boxenplot(
                x="dataset",
                y=metrics_df.keys()[i],
                data=metrics_df,
                ax=ax[i],
                order=order,
                showfliers=show_outlier,
            )

    # if not pairs is None:
    #     for i in range(6):
    #         annotator = Annotator(
    #             ax[i],
    #             pairs,
    #             data=metrics_df,
    #             x="dataset",
    #             y=metrics_df.keys()[i],
    #             order=order,
    #         )
    #         annotator.configure(
    #             test="Mann-Whitney", text_format="star", loc="inside"
    #         )
    #         annotator.apply_and_annotate()

    for axi in ax:
        axi.tick_params(axis="both", labelsize=20)
        axi.set_xlabel("")

    for ext in ["", ".png"]:
        fig.savefig(
            f"{fig_name}{ext}",
            facecolor=fig.get_facecolor(),
            bbox_inches="tight",
            edgecolor="none",
            dpi=300,
        )


def get_posterior_sample_angle_uncertainty(posterior_angles):
    from astropy import units as u
    from astropy.stats import circstd

    x_values = np.arange(360)
    y_values = []
    n_samples = 100
    method = "angular"
    for i in x_values:
        datac = np.linspace(0, i + 1, n_samples) * u.deg
        y_values.append(circstd(datac, method=method))
    y_values = np.array(y_values)
    angle_std = circstd(posterior_angles * u.deg, method="angular", axis=0)
    y_values = y_values.reshape(-1, 1)
    return x_values[np.argmin(np.abs(y_values - angle_std), axis=0)]


def plot_state_uncertainty(
    posterior_samples,
    adata,
    kde=True,
    data="denoised",
    top_percentile=0.9,
    ax=None,
    basis="umap",
):
    if data == "denoised":
        adata.obs["state_uncertain"] = np.sqrt(
            (
                (posterior_samples["st"] - posterior_samples["st"].mean(0)) ** 2
                + (posterior_samples["ut"] - posterior_samples["ut"].mean(0))
                ** 2
            ).sum(-1)
        ).mean(0)
    else:
        adata.obs["state_uncertain"] = np.sqrt(
            (
                (posterior_samples["s"] - posterior_samples["s"].mean(0)) ** 2
                + (posterior_samples["u"] - posterior_samples["u"].mean(0)) ** 2
            ).sum(-1)
        ).mean(0)

    ax = scv.pl.scatter(
        adata,
        basis=basis,
        color="state_uncertain",
        cmap="RdBu_r",
        ax=ax,
        show=False,
        colorbar=True,
        fontsize=7,
    )

    if kde:
        select = adata.obs["state_uncertain"] > np.quantile(
            adata.obs["state_uncertain"], top_percentile
        )
        sns.kdeplot(
            adata.obsm[f"X_{basis}"][:, 0][select],
            adata.obsm[f"X_{basis}"][:, 1][select],
            ax=ax,
            levels=3,
            fill=False,
        )
    else:
        select = None
    return select, ax
