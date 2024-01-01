import os
from logging import Logger
from pathlib import Path

import anndata
import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyro
import scvelo as scv
import seaborn as sns
import torch
from astropy import units as u
from astropy.stats import circstd as acircstd
from omegaconf import DictConfig
from pyro.infer import Predictive, infer_discrete
from pyrovelocity._velocity import PyroVelocity
from pyrovelocity.io.compressedpickle import CompressedPickle
from pyrovelocity.plot import (
    get_posterior_sample_angle_uncertainty,
    plot_gene_ranking,
    plot_posterior_time,
    plot_vector_field_uncertain,
    rainbowplot,
)
from pyrovelocity.plots.rainbow import pareto_frontier_genes
from pyrovelocity.utils import (
    anndata_counts_to_df,
    get_pylogger,
)

# from scipy.stats import circmean
# from scipy.stats import circstd
from statannotations.Annotator import Annotator

"""Loads model-trained data and generates figures.

for data_model in
    [
        pancreas_model1,
        pancreas_model2,
        pbmc68k_model1,
        pbmc68k_model2,
        pons_model1,
        pons_model2,
    ]

Inputs:
  data:
    "models/{data_model}/trained.h5ad"
    "models/{data_model}/pyrovelocity.pkl"

Outputs:
  figures:
    shared_time_plot: reports/{data_model}/shared_time.pdf
    volcano_plot: reports/{data_model}/volcano.pdf
    rainbow_plot: reports/{data_model}/rainbow.pdf
    uncertainty_param_plot: reports/{data_model}/param_uncertainties.pdf
    vector_field_plot: reports/{data_model}/vector_field.pdf
    biomarker_selection_plot: reports/{data_model}/markers_selection_scatterplot.tif
    biomarker_phaseportrait_plot: reports/{data_model}/markers_phaseportrait.pdf
"""


def summarize_fig2_part1(
    adata,
    posterior_vector_field,
    posterior_time,
    cell_magnitudes,
    pca_embeds_angle,
    embed_radians,
    embedding,
    embed_mean,
    cluster="cell_type",
    plot_name="test",
):
    dot_size = 3.5
    font_size = 6.5
    scale = 0.35
    scale_high = 7.8
    scale_low = 7.8

    arrow = 3.6
    density = 0.4
    ress = pd.DataFrame(
        {
            "cell_type": adata.obs[cluster].values,
            "X1": adata.obsm[f"X_{embedding}"][:, 0],
            "X2": adata.obsm[f"X_{embedding}"][:, 1],
        }
    )
    fig = plt.figure(figsize=(9.6, 2), constrained_layout=False)
    fig.subplots_adjust(
        hspace=0.2, wspace=0.1, left=0.01, right=0.99, top=0.99, bottom=0.45
    )
    ax = fig.subplots(1, 6)
    pos = ax[0].get_position()

    sns.scatterplot(
        x="X1",
        y="X2",
        data=ress,
        alpha=0.9,
        s=dot_size,
        linewidth=0,
        edgecolor="none",
        hue="cell_type",
        ax=ax[0],
        legend="brief",
    )
    ax[0].axis("off")
    ax[0].set_title("Cell types\n", fontsize=font_size)
    ax[0].legend(
        loc="lower left",
        bbox_to_anchor=(0.5, -0.48),
        ncol=5,
        fancybox=True,
        prop={"size": font_size},
        fontsize=font_size,
        frameon=False,
    )
    kwargs = dict(
        color="gray",
        s=dot_size,
        show=False,
        alpha=0.25,
        min_mass=3.5,
        scale=scale,
        frameon=False,
        density=density,
        arrow_size=3,
        linewidth=1,
    )
    scv.pl.velocity_embedding_grid(
        adata, basis=embedding, fontsize=font_size, ax=ax[1], title="", **kwargs
    )
    ax[1].set_title("Scvelo\n", fontsize=7)
    scv.pl.velocity_embedding_grid(
        adata,
        fontsize=font_size,
        basis=embedding,
        title="",
        ax=ax[2],
        vkey="velocity_pyro",
        **kwargs,
    )
    ax[2].set_title("Pyro-Velocity\n", fontsize=7)

    pca_cell_angles = pca_embeds_angle / np.pi * 180  # degree
    pca_angles_std = get_posterior_sample_angle_uncertainty(pca_cell_angles)

    cell_time_mean = posterior_time.mean(0).flatten()
    cell_time_std = posterior_time.std(0).flatten()
    cell_time_cov = cell_time_std / cell_time_mean

    plot_vector_field_uncertain(
        adata,
        embed_mean,
        cell_time_std,
        ax=ax[3],
        cbar=True,
        fig=fig,
        basis=embedding,
        scale=scale,
        arrow_size=arrow,
        p_mass_min=1,
        autoscale=True,
        density=density,
        only_grid=False,
        uncertain_measure="shared time",
        cmap="winter",
        cmax=None,
    )

    cell_magnitudes_mean = cell_magnitudes.mean(axis=-2)
    cell_magnitudes_std = cell_magnitudes.std(axis=-2)
    cell_magnitudes_cov = cell_magnitudes_std / cell_magnitudes_mean
    plot_vector_field_uncertain(
        adata,
        embed_mean,
        cell_magnitudes_cov,
        ax=ax[4],
        cbar=True,
        fig=fig,
        basis=embedding,
        scale=scale,
        arrow_size=arrow,
        p_mass_min=1,
        autoscale=True,
        density=density,
        only_grid=False,
        uncertain_measure="base magnitude",
        cmap="summer",
        cmax=None,
    )

    plot_vector_field_uncertain(
        adata,
        embed_mean,
        pca_angles_std,
        ax=ax[5],
        cbar=True,
        fig=fig,
        basis=embedding,
        scale=scale,
        arrow_size=arrow,
        p_mass_min=1,
        autoscale=True,
        density=density,
        only_grid=False,
        uncertain_measure="PCA angle",
        cmap="inferno",
        cmax=None,
    )
    for ext in ["", ".png"]:
        fig.savefig(
            f"{plot_name}{ext}",
            facecolor=fig.get_facecolor(),
            bbox_inches="tight",
            edgecolor="none",
            dpi=300,
        )


def summarize_fig2_part2(
    adata, posterior_samples, plot_name="", basis="", cell_state="", fig=None
):
    if fig is None:
        fig = plt.figure(figsize=(9.5, 5))
        subfigs = fig.subfigures(
            1, 2, wspace=0.0, hspace=0, width_ratios=[1.8, 4]
        )
        ax = subfigs[0].subplots(2, 1)
        plot_posterior_time(
            posterior_samples,
            adata,
            ax=ax[0],
            fig=subfigs[0],
            addition=False,
            basis=basis,
        )
        volcano_data, _ = plot_gene_ranking(
            [posterior_samples], [adata], ax=ax[1], time_correlation_with="st"
        )
        _ = rainbowplot(
            volcano_data,
            adata,
            posterior_samples,
            subfigs[1],
            data=["st", "ut"],
            basis=basis,
            cell_state=cell_state,
            num_genes=4,
        )
        for ext in ["", ".png"]:
            fig.savefig(
                f"{plot_name}{ext}",
                facecolor=fig.get_facecolor(),
                bbox_inches="tight",
                edgecolor="none",
                dpi=300,
            )


def cluster_violin_plots(
    logger: Logger,
    data_model: str,
    adata: anndata.AnnData,
    posterior_samples,
    cluster_key: str,
    violin_flag: bool,
    pairs: list,
    show_outlier: bool,
    fig_name: str = None,
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
    print(adata)
    print(adata.obs[cluster_key])

    # get cluster order
    cluster_time_list = []
    clusters = adata.obs[cluster_key].values.categories
    for cluster in clusters:
        adata_cluster = adata[adata.obs[cluster_key] == cluster]
        cluster_time = adata_cluster.obs["velocity_pseudotime"].mean()
        cluster_time_list.append(cluster_time)
    print(cluster_time_list)
    sorted_cluster_id = sorted(
        range(len(cluster_time_list)),
        key=lambda k: cluster_time_list[k],
        reverse=False,
    )
    order = clusters[sorted_cluster_id]

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
    name = list(adata.obs[cluster_key])
    names += name

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
                "dataset": names,
            }
        )

    logger.info(metrics_df.head())
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
            annotator.configure(
                test="Mann-Whitney", text_format="star", loc="inside"
            )
            annotator.apply_and_annotate()

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


def plot_parameter_posterior_distributions(
    posterior_samples,
    adata: anndata.AnnData,
    geneset,
    parameter_uncertainty_plot_path: str,
):
    fig, ax = plt.subplots(3, 1)
    fig.set_size_inches(18, 12)
    for index, kinetics in enumerate(["alpha", "beta", "gamma"]):
        print(posterior_samples[kinetics].squeeze().shape)
        print(np.isin(adata.var_names, list(geneset)).sum())
        df = pd.DataFrame(
            np.log(
                posterior_samples[kinetics].squeeze()[
                    :, np.isin(adata.var_names, list(geneset))
                ],
            ),
            columns=adata.var_names[np.isin(adata.var_names, list(geneset))],
        )
        df = df.apply(lambda x: x - x.mean())
        df_long = df.melt(var_name="index", value_name="value")
        print(df_long.head())
        df_long["index"] = pd.Categorical(
            df_long["index"], categories=geneset, ordered=True
        )
        ax1 = sns.violinplot(
            x="index", y="value", data=df_long, ax=ax[index], showfliers=False
        )
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=30, ha="right")
        ax1.set_ylabel(kinetics)
        ax1.set_xlabel("")
    fig.subplots_adjust(
        hspace=0.4, wspace=0.45, left=0.08, right=0.95, top=0.9, bottom=0.15
    )
    for ext in ["", ".png"]:
        fig.savefig(
            f"{parameter_uncertainty_plot_path}{ext}",
            facecolor=fig.get_facecolor(),
            bbox_inches="tight",
            edgecolor="none",
            dpi=300,
        )


def extrapolate_prediction_sample_predictive(
    posterior_time, data_model_conf, adata, grid_time_points=1000
):
    pyrovelocity_model_path = data_model_conf.model_path
    PyroVelocity.setup_anndata(adata)
    model = PyroVelocity(
        adata, add_offset=False, guide_type="auto_t0_constraint"
    )
    model = model.load_model(pyrovelocity_model_path, adata, use_gpu=0)
    print(pyrovelocity_model_path)

    scdl = model._make_data_loader(adata=adata, indices=None, batch_size=1000)

    posterior_samples_list = []
    for tensor_dict in scdl:
        print("--------------------")
        u_obs = tensor_dict["U"]
        s_obs = tensor_dict["X"]
        u_log_library = tensor_dict["u_lib_size"]
        s_log_library = tensor_dict["s_lib_size"]
        u_log_library_mean = tensor_dict["u_lib_size_mean"]
        s_log_library_mean = tensor_dict["s_lib_size_mean"]
        u_log_library_scale = tensor_dict["u_lib_size_scale"]
        s_log_library_scale = tensor_dict["s_lib_size_scale"]
        ind_x = tensor_dict["ind_x"].long().squeeze()
        dummy_obs = (
            torch.tensor(u_obs).to("cuda:0"),
            torch.tensor(s_obs).to("cuda:0"),
            torch.tensor(u_log_library).to("cuda:0"),
            torch.tensor(s_log_library).to("cuda:0"),
            torch.tensor(u_log_library_mean).to("cuda:0"),
            torch.tensor(s_log_library_mean).to("cuda:0"),
            torch.tensor(u_log_library_scale).to("cuda:0"),
            torch.tensor(s_log_library_scale).to("cuda:0"),
            torch.tensor(ind_x).to("cuda:0"),
            None,
            None,
        )

        posterior_samples = {}
        posterior_samples_batch_sample = []
        for sample in range(5):
            guide_trace = pyro.poutine.trace(model.module.guide).get_trace(
                *dummy_obs
            )
            trained_model = pyro.poutine.replay(
                model.module.model, trace=guide_trace
            )
            model_discrete = infer_discrete(
                trained_model, temperature=0, first_available_dim=-3
            )
            trace = pyro.poutine.trace(model_discrete).get_trace(*dummy_obs)
            map_estimate_cell_gene_state = trace.nodes["cell_gene_state"][
                "value"
            ]
            alpha = trace.nodes["alpha"]["value"]
            beta = trace.nodes["beta"]["value"]
            gamma = trace.nodes["gamma"]["value"]
            t0 = trace.nodes["t0"]["value"]
            dt_switching = trace.nodes["dt_switching"]["value"]
            cell_time = trace.nodes["cell_time"]["value"]

            if "u_offset" in trace.nodes:
                u_offset = trace.nodes["u_offset"]["value"]
                s_offset = trace.nodes["s_offset"]["value"]
                u_scale = trace.nodes["u_scale"]["value"]
            else:
                u_offset = alpha.new_zeros(alpha.shape)
                s_offset = alpha.new_zeros(alpha.shape)
                u_scale = alpha.new_ones(alpha.shape)
            posterior_samples_batch_sample.append(
                {
                    "cell_gene_state": map_estimate_cell_gene_state.unsqueeze(
                        -3
                    ),
                    "alpha": alpha.unsqueeze(-2).unsqueeze(-3),
                    "beta": beta.unsqueeze(-2).unsqueeze(-3),
                    "gamma": gamma.unsqueeze(-2).unsqueeze(-3),
                    "u_offset": u_offset.unsqueeze(-2).unsqueeze(-3),
                    "s_offset": s_offset.unsqueeze(-2).unsqueeze(-3),
                    "u_scale": u_scale.unsqueeze(-2).unsqueeze(-3),
                    "dt_switching": dt_switching.unsqueeze(-2).unsqueeze(-3),
                    "cell_time": cell_time.unsqueeze(-3),
                    "t0": t0.unsqueeze(-2).unsqueeze(-3),
                }
            )

        for key in posterior_samples_batch_sample[0].keys():
            posterior_samples[key] = torch.tensor(
                np.concatenate(
                    [
                        posterior_samples_batch_sample[j][key]
                        .cpu()
                        .detach()
                        .numpy()
                        for j in range(len(posterior_samples_batch_sample))
                    ],
                    axis=-3,
                )
            ).to("cuda:0")

        posterior_samples_new_tmp = Predictive(
            pyro.poutine.uncondition(
                model.module.model,
            ),
            posterior_samples,
        )(*dummy_obs)
        for key in posterior_samples:
            posterior_samples_new_tmp[key] = posterior_samples[key]
        posterior_samples_list.append(posterior_samples_new_tmp)

    print(len(posterior_samples_list))
    posterior_samples_new = {}
    for key in posterior_samples_list[0].keys():
        if posterior_samples_list[0][key].shape[-2] == 1:
            posterior_samples_new[key] = posterior_samples_list[0][key]
        else:
            posterior_samples_new[key] = torch.concat(
                [element[key] for element in posterior_samples_list], axis=-2
            )
    # posterior_samples_new = model.generate_posterior_samples(
    #    adata=adata, batch_size=512, num_samples=8
    # )

    for key in posterior_samples_new.keys():
        print(posterior_samples_new[key].shape)

    grid_time_samples_ut = posterior_samples_new["ut"]
    grid_time_samples_st = posterior_samples_new["st"]
    grid_time_samples_uinf = posterior_samples_new["u_inf"]
    grid_time_samples_sinf = posterior_samples_new["s_inf"]
    if "u_offset" in posterior_samples_new:
        grid_time_samples_u0 = posterior_samples_new["u_offset"]
        grid_time_samples_s0 = posterior_samples_new["s_offset"]
    else:
        grid_time_samples_u0 = np.zeros(grid_time_samples_uinf.shape)
        grid_time_samples_s0 = np.zeros(grid_time_samples_sinf.shape)

    grid_time_samples_t0 = posterior_samples_new["t0"]
    grid_time_samples_dt_switching = posterior_samples_new["dt_switching"]
    if "u_offset" in posterior_samples_new:
        grid_time_samples_uscale = posterior_samples_new["u_scale"]
    else:
        grid_time_samples_uscale = np.ones(grid_time_samples_uinf.shape)

    grid_time_samples_state = posterior_samples_new["cell_gene_state"]
    print(grid_time_samples_state.shape)
    print(grid_time_samples_uscale.shape)
    print(grid_time_samples_ut.shape)
    print(grid_time_samples_st.shape)
    if isinstance(grid_time_samples_state, np.ndarray):
        return (
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
        )
    else:
        return (
            grid_time_samples_ut.cpu().detach().numpy(),
            grid_time_samples_st.cpu().detach().numpy(),
            grid_time_samples_u0.cpu().detach().numpy(),
            grid_time_samples_s0.cpu().detach().numpy(),
            grid_time_samples_uinf.cpu().detach().numpy(),
            grid_time_samples_sinf.cpu().detach().numpy(),
            grid_time_samples_uscale.cpu().detach().numpy(),
            grid_time_samples_state.cpu().detach().numpy(),
            grid_time_samples_t0.cpu().detach().numpy(),
            grid_time_samples_dt_switching.cpu().detach().numpy(),
        )


def posterior_curve(
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
    gene_set,
    dataset,
    directory,
):
    # grid_cell_time = np.linspace(-50, 50, 500)
    grid_cell_time = posterior_samples["cell_time"]

    for figi, gene in enumerate(gene_set):
        (index,) = np.where(adata.var_names == gene)
        print(adata.shape, index, posterior_samples["st_mean"].shape)

        fig, ax = plt.subplots(3, 4)
        fig.set_size_inches(15, 10)
        ax = ax.flatten()
        for sample in range(4):
            t0_sample = posterior_samples["t0"][sample][:, index[0]].flatten()
            cell_time_sample = posterior_samples["cell_time"][sample].flatten()
            cell_time_sample_max = cell_time_sample.max()
            cell_time_sample_min = cell_time_sample.min()

            colors = np.array(["gray", "blue"])
            mask_t0_sample = (cell_time_sample >= t0_sample).astype(int)
            cell_colors = colors[mask_t0_sample]

            colors = np.array(["gray", "blue", "red"])
            grid_mask_t0_sample = (
                grid_cell_time.mean(0).flatten() >= t0_sample
            ).astype("float32")

            cell_gene_state_grid = grid_time_samples_state[0][
                :, index[0]
            ].astype("float32")

            grid_mask_t0_sample = grid_mask_t0_sample + cell_gene_state_grid
            grid_mask_t0_sample = grid_mask_t0_sample.astype(int)
            grid_mask_t0_sample[
                grid_cell_time.mean(0).flatten() < t0_sample
            ] = 0
            grid_cell_colors = colors[grid_mask_t0_sample]
            print(grid_time_samples_st.shape)

            im = ax[sample].scatter(
                posterior_samples["st_mean"][:, index[0]],
                posterior_samples["ut_mean"][:, index[0]],
                s=3,
                linewidth=0,
                # color=cell_colors,
                color=grid_cell_colors,
                alpha=0.6,
            )

            im = ax[sample].scatter(
                grid_time_samples_st[sample][:, index[0]],
                grid_time_samples_ut[sample][:, index[0]],
                s=15,
                marker="*",
                linewidth=0,
                alpha=0.2,
                c=grid_cell_colors,
            )
            ax[sample + 4].scatter(
                grid_cell_time.mean(0).flatten(),
                posterior_samples["ut_mean"][:, index[0]],
                s=3,
                linewidth=0,
                marker=".",
                color=grid_cell_colors,
                alpha=0.3,
            )
            ax[sample + 4].scatter(
                grid_cell_time[sample].flatten(),
                grid_time_samples_ut[sample][:, index[0]],
                s=15,
                marker=">",
                linewidth=0,
                color=grid_cell_colors,
                alpha=0.5,
            )
            ax[sample + 4].set_title("Unspliced", fontsize=7)
            ax[sample + 4].set_ylabel("Unspliced (Ut)", fontsize=7)
            ax[sample + 8].scatter(
                grid_cell_time.mean(0).flatten(),
                posterior_samples["st_mean"][:, index[0]],
                s=3,
                marker="*",
                linewidth=0,
                color=grid_cell_colors,
                alpha=0.3,
            )
            ax[sample + 8].scatter(
                grid_cell_time[sample].flatten(),
                grid_time_samples_st[sample][:, index[0]],
                s=15,
                marker="<",
                linewidth=0,
                color=grid_cell_colors,
                alpha=0.3,
            )
            ax[sample + 8].set_title("Spliced", fontsize=7)
            ax[sample + 8].set_ylabel("Spliced (St)", fontsize=7)

            u0 = grid_time_samples_u0[sample][:, index[0]].flatten()
            uscale = grid_time_samples_uscale[sample][:, index[0]].flatten()
            s0 = grid_time_samples_s0[sample][:, index[0]].flatten()
            u_inf = grid_time_samples_uinf[sample][:, index[0]].flatten()
            s_inf = grid_time_samples_sinf[sample][:, index[0]].flatten()

            t0_sample = grid_time_samples_t0[sample][:, index[0]].flatten()
            dt_switching_sample = grid_time_samples_dt_switching[sample][
                :, index[0]
            ].flatten()

            ##u0 = posterior_samples['u_offset'][sample][:, index[0]].flatten()
            ##s0 = posterior_samples['s_offset'][sample][:, index[0]].flatten()
            ##u_inf = posterior_samples['u_inf'][sample][:, index[0]].flatten()
            ##s_inf = posterior_samples['s_inf'][sample][:, index[0]].flatten()
            ##switching = posterior_samples['switching'][sample][:, index[0]].flatten()
            ##dt_switching = posterior_samples['dt_switching'][sample][:, index[0]].flatten()

            ax[sample + 4].scatter(
                t0_sample,
                u0 * uscale,
                s=80,
                marker="p",
                linewidth=0.5,
                c="purple",
                alpha=0.8,
            )
            ax[sample + 4].scatter(
                t0_sample + dt_switching_sample,
                u_inf * uscale,
                s=80,
                marker="*",
                linewidth=0.5,
                c="black",
                alpha=0.8,
            )

            ax[sample + 8].scatter(
                t0_sample,
                s0,
                s=80,
                marker="p",
                linewidth=0.5,
                c="purple",
                alpha=0.8,
            )
            ax[sample + 8].scatter(
                t0_sample + dt_switching_sample,
                s_inf,
                s=80,
                marker="p",
                linewidth=0.5,
                c="black",
                alpha=0.8,
            )

            ax[sample].scatter(
                s0, u0 * uscale, s=60, marker="p", linewidth=0.5, c="purple"
            )
            ax[sample].scatter(
                s_inf,
                u_inf * uscale,
                s=60,
                marker="p",
                linewidth=0.5,
                c="black",
            )
            # ax[sample].plot(grid_time_samples_st[sample][:, index[0]],
            #                grid_time_samples_ut[sample][:, index[0]],
            #                linestyle="--", linewidth=3, color='g')
            if sample == 0:
                print(gene, u0 * uscale, s0)
                print(gene, u_inf * uscale, s_inf)
                print(
                    t0_sample,
                    dt_switching_sample,
                    cell_time_sample_min,
                    cell_time_sample_max,
                    (cell_time_sample <= t0_sample).sum(),
                )
                print(cell_time_sample.shape)

            switching = t0_sample + dt_switching_sample
            state0 = (cell_gene_state_grid == 0) & (
                cell_time_sample <= switching
            )
            state0_false = (cell_gene_state_grid == 0) & (
                cell_time_sample > switching
            )
            state1 = (cell_gene_state_grid == 1) & (
                cell_time_sample >= switching
            )
            state1_false = (cell_gene_state_grid == 1) & (
                cell_time_sample < switching
            )

            ax[sample].set_title(
                f"{gene} model 2 sample {sample}\nt0>celltime:{(t0_sample>cell_time_sample_max)} {(t0_sample>cell_time_sample).sum()}\nstate0: {state0.sum()} {state0_false.sum()} state1: {state1.sum()} {state1_false.sum()}",
                fontsize=6.5,
            )
            ax[sample].set_xlim(
                0,
                max(
                    [
                        np.max(posterior_samples["st_mean"][:, index[0]]) * 1.1,
                        np.max(grid_time_samples_st[sample][:, index[0]]),
                        s0 * 1.1,
                        s_inf * 1.1,
                    ]
                ),
            )
            ax[sample].set_ylim(
                0,
                max(
                    [
                        np.max(posterior_samples["ut_mean"][:, index[0]]) * 1.1,
                        np.max(grid_time_samples_ut[sample][:, index[0]]),
                        u0 * uscale * 1.1,
                        u_inf * uscale * 1.05,
                    ]
                ),
            )
            fig.colorbar(im, ax=ax[sample])
        fig.tight_layout()
        fig.savefig(
            f"{directory}/fig{dataset}_{gene}_test.png",
            facecolor=fig.get_facecolor(),
            bbox_inches="tight",
            edgecolor="none",
            dpi=300,
        )


def plots(conf: DictConfig, logger: Logger) -> None:
    """Construct summary plots for each data set and model.

    Args:
        conf (DictConfig): OmegaConf configuration object
        logger (Logger): Python logger

    Examples:
        plots(conf, logger)
    """
    for data_model in conf.train_models:
        ##################
        # load data
        ##################

        data_model_conf = conf.model_training[data_model]
        cell_state = data_model_conf.training_parameters.cell_state
        trained_data_path = data_model_conf.trained_data_path
        pyrovelocity_data_path = data_model_conf.pyrovelocity_data_path
        reports_data_model_conf = conf.reports.model_summary[data_model]
        trained_data_path = reports_data_model_conf.trained_data_path
        pyrovelocity_data_path = reports_data_model_conf.pyrovelocity_data_path

        # print_config_tree(reports_data_model_conf, logger, ())

        logger.info(f"\n\nPlotting summary figure(s) in: {data_model}\n\n")

        logger.info(
            f"\n\nVerifying existence of path for:\n\n"
            f"  reports: {reports_data_model_conf.path}\n"
        )
        Path(reports_data_model_conf.path).mkdir(parents=True, exist_ok=True)
        Path(reports_data_model_conf.posterior_phase_portraits).mkdir(
            parents=True, exist_ok=True
        )

        dataframe_path = reports_data_model_conf.dataframe_path
        volcano_plot = reports_data_model_conf.volcano_plot
        rainbow_plot = reports_data_model_conf.rainbow_plot
        vector_field_plot = reports_data_model_conf.vector_field_plot
        shared_time_plot = reports_data_model_conf.shared_time_plot
        fig2_part1_plot = reports_data_model_conf.fig2_part1_plot
        fig2_part2_plot = reports_data_model_conf.fig2_part2_plot
        violin_clusters_lin = reports_data_model_conf.violin_clusters_lin
        violin_clusters_log = reports_data_model_conf.violin_clusters_log
        parameter_uncertainty_plot_path = (
            reports_data_model_conf.uncertainty_param_plot
        )

        output_filenames = [
            dataframe_path,
            volcano_plot,
            rainbow_plot,
            vector_field_plot,
            shared_time_plot,
            parameter_uncertainty_plot_path,
        ]
        if all(os.path.isfile(f) for f in output_filenames):
            logger.info(
                "\n\t"
                + "\n\t".join(output_filenames)
                + "\nAll output files exist"
            )
            return logger.warn(
                "Remove output files if you want to regenerate them."
            )

        logger.info(f"Loading trained data: {trained_data_path}")
        adata = scv.read(trained_data_path)
        # gene_mapping = {"1100001G20Rik": "Wfdc21"}
        # adata = rename_anndata_genes(adata, gene_mapping)

        logger.info(f"Loading pyrovelocity data: {pyrovelocity_data_path}")
        posterior_samples = CompressedPickle.load(pyrovelocity_data_path)
        print(posterior_samples.keys())

        fig, ax = plt.subplots(5, 6)
        fig.set_size_inches(26, 24)
        ax = ax.flatten()
        for sample in range(29):
            t0_sample = posterior_samples["t0"][sample]
            switching_sample = posterior_samples["switching"][sample]
            cell_time_sample = posterior_samples["cell_time"][sample]
            print(t0_sample.shape)
            print(cell_time_sample)
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
                ax[sample].legend(
                    loc="best", bbox_to_anchor=(0.5, 0.0, 0.5, 0.5)
                )
            print(
                (t0_sample.flatten() > cell_time_sample.flatten().max()).sum()
            )
            print(
                (t0_sample.flatten() < switching_sample.flatten().max()).sum()
            )
            print(
                (t0_sample.flatten() > switching_sample.flatten().max()).sum()
            )
            for gene in adata.var_names[
                t0_sample.flatten() > cell_time_sample.flatten().max()
            ]:
                print(gene)
        ax[-1].hist(t0_sample.flatten(), bins=200, color="red", alpha=0.3)
        ax[-1].hist(
            cell_time_sample.flatten(), bins=500, color="blue", alpha=0.3
        )

        fig.savefig(
            reports_data_model_conf.t0_selection,
            facecolor=fig.get_facecolor(),
            bbox_inches="tight",
            edgecolor="none",
            dpi=300,
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
            data_model_conf,
            adata,
            grid_time_points=500,
        )
        # extrapolate_prediction_trace(data_model_conf, adata, grid_time_points=5)

        ##################
        # save dataframe
        ##################

        logger.info(f"Saving AnnData object to dataframe: {dataframe_path}")
        (
            df,
            total_obs,
            total_var,
            max_spliced,
            max_unspliced,
        ) = anndata_counts_to_df(adata)

        CompressedPickle.save(
            dataframe_path,
            (
                df,
                total_obs,
                total_var,
                max_spliced,
                max_unspliced,
            ),
        )
        print(posterior_samples.keys())

        vector_field_basis = data_model_conf.vector_field_parameters.basis

        cell_type = data_model_conf.training_parameters.cell_state

        ##################
        # generate figures
        ##################

        # vector fields
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
                    logger,
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
                adata.obsm[f"X_{vector_field_basis}"][:, 0][select],
                adata.obsm[f"X_{vector_field_basis}"][:, 1][select],
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
        print(f"Searching for {number_of_marker_genes} marker genes")
        geneset = pareto_frontier_genes(volcano_data, number_of_marker_genes)

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
            reports_data_model_conf.posterior_phase_portraits,
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
        from pyrovelocity.plots.rainbow import rainbowplot

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


def rename_anndata_genes(adata, gene_mapping):
    """
    Renames genes in an AnnData object according to a dictionary mapping.

    Parameters:
    -----------
    adata: anndata.AnnData
        The AnnData object.
    gene_mapping: dict
        A dictionary containing mappings of old gene names (keys) to new gene names (values).

    Returns:
    -------
    adata: anndata.AnnData
        The AnnData object with renamed genes.
    """

    var_names = adata.var_names.tolist()
    var_names = [
        gene_mapping[name] if name in gene_mapping else name
        for name in var_names
    ]
    adata.var_names = var_names

    return adata


@hydra.main(version_base="1.2", config_path=".", config_name="config.yaml")
def main(conf: DictConfig) -> None:
    """Plots figures for model summary.
    Args:
        config {DictConfig}: hydra configuration
    """

    logger = get_pylogger(name="PLOT", log_level=conf.base.log_level)
    plots(conf, logger)


if __name__ == "__main__":
    main()
