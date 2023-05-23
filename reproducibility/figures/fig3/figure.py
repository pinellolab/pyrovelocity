import errno
import os
import pickle
from logging import Logger
from pathlib import Path
import cospar as cs

import hydra
import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy.stats import spearmanr
import numpy as np
import pandas as pd
import scvelo as scv
import seaborn as sns
from omegaconf import DictConfig

from pyrovelocity.config import print_config_tree
from pyrovelocity.io.compressedpickle import CompressedPickle
from pyrovelocity.plot import plot_arrow_examples
from pyrovelocity.plot import plot_gene_ranking
from pyrovelocity.plot import plot_posterior_time
from pyrovelocity.plot import plot_vector_field_uncertain
from pyrovelocity.plot import rainbowplot
from pyrovelocity.utils import get_pylogger
from pyrovelocity.data import load_larry

from scvelo.plotting.velocity_embedding_grid import default_arrow
from pyrovelocity.plot import align_trajectory_diff
from pyrovelocity.plot import get_clone_trajectory
from pyrovelocity.plot import plot_posterior_time
from pyrovelocity.plot import plot_vector_field_uncertain


"""Loads trained figure S3 data and produces figure S3.

Inputs:
  data:
    "models/pancreas_model2/trained.h5ad"
    "models/pbmc68k_model2/trained.h5ad"
  models:
    "models/pancreas_model2/pyrovelocity.pkl"
    "models/pbmc68k_model2/pyrovelocity.pkl"

Outputs:
  figures:
    "reports/figS3/figS3_raw_gene_selection_model1.{tif,svg}"
"""


def plots(conf: DictConfig, logger: Logger) -> None:
    ##################
    # load data
    ##################
    cs.logging.print_version()
    cs.settings.verbosity = 2
    cs.settings.data_path = "LARRY_data"  # A relative path to save data.
    cs.settings.figure_path = "LARRY_figure"  # A relative path to save figures.
    Path("LARRY_figure").mkdir(parents=True, exist_ok=True)
    Path("LARRY_data").mkdir(parents=True, exist_ok=True)
    cs.settings.set_figure_params(
        format="png", figsize=[4, 3.5], dpi=75, fontsize=14, pointsize=2
    )

    larry_cospar_data_path = conf.data_sets.larry_cospar.derived.rel_path
    larry_cytotrace_data_path = conf.data_sets.larry_cytotrace.derived.rel_path
    larry_dynamical_data_path = conf.data_sets.larry_dynamical.derived.rel_path

    figure3_plot_name = conf.reports.figure3.figure3
    print(figure3_plot_name)

    pyrovelocity_larry_data_path = (
        conf.model_training.larry_model2.pyrovelocity_data_path
    )
    trained_larry_data_path = conf.model_training.larry_model2.trained_data_path

    mono_data_path = conf.model_training.larry_mono_model2.pyrovelocity_data_path
    trained_mono_data_path = conf.model_training.larry_mono_model2.trained_data_path

    neu_data_path = conf.model_training.larry_neu_model2.pyrovelocity_data_path
    trained_neu_data_path = conf.model_training.larry_neu_model2.trained_data_path

    multilineage_data_path = conf.model_training.larry_multilineage_model2.pyrovelocity_data_path
    trained_neu_data_path = conf.model_training.larry_multilineage_model2.trained_data_path

    adata = load_larry()
    adata_input_vel = scv.read(larry_dynamical_data_path)
    adata_input_all = scv.read(trained_larry_data_path)

    adata_cospar = scv.read(larry_cospar_data_path)
    adata_cytotrace = scv.read(larry_cytotrace_data_path)  # skip=False

    cs.pl.fate_potency(
        adata_cospar,
        used_Tmap="transition_map",
        map_backward=True,
        method="norm-sum",
        color_bar=True,
        fate_count=True,
    )
    scv.tl.velocity_embedding(adata_input_vel, basis="emb")
    scv.pl.scatter(
        adata_input_all,
        basis="emb",
        fontsize=7,
        legend_loc="on data",
        legend_fontsize=7,
        color="state_info",
        show=False,
    )
        
    print_config_tree(conf.reports.figure3, logger, ())

    logger.info(
        f"\n\nVerifying existence of paths for:\n\n"
        f"  reports: {conf.reports.figureS3.path}\n"
    )
    Path(conf.reports.figure3.path).mkdir(parents=True, exist_ok=True)

    posterior_samples = CompressedPickle.load(pyrovelocity_larry_data_path)
    
    v_map_all = posterior_samples["vector_field_posterior_samples"]
    embeds_radian_all = posterior_samples["embeds_angle"]
    fdri = posterior_samples["fdri"]
    embed_mean_all = posterior_samples["vector_field_posterior_mean"]
    
    if os.path.exists("global_gold_standard2.h5ad"):
        adata_input_all_clone = scv.read("global_gold_standard2.h5ad")
    else:
        adata_reduced_gene_for_clone_vec = adata[:, adata_input_vel.var_names].copy()
        print(adata_reduced_gene_for_clone_vec.shape)
        adata_input_all_clone = get_clone_trajectory(adata_reduced_gene_for_clone_vec)
        adata_input_all_clone.write("global_gold_standard2.h5ad")
    
    adata_input_all_clone.obsm["clone_vector_emb"][
        np.isnan(adata_input_all_clone.obsm["clone_vector_emb"])
    ] = 0
    
    cutoff = 10
    density = 0.35
    diff_all = align_trajectory_diff(
        [adata_input_all_clone, adata_input_vel, adata_input_all],
        [
            adata_input_all_clone.obsm["clone_vector_emb"],
            adata_input_vel.obsm["velocity_emb"],
            embed_mean_all,
        ],
        embed="emb",
        density=density,
    )
    scvelo_all_cos = pd.DataFrame(diff_all).apply(
        lambda x: 1 - distance.cosine(x[2:4], x[4:6]), axis=1
    )
    pyro_all_cos = pd.DataFrame(diff_all).apply(
        lambda x: 1 - distance.cosine(x[2:4], x[6:8]), axis=1
    )
    scvelo_all_cos_mean = scvelo_all_cos.mean()
    pyro_all_cos_mean = pyro_all_cos.mean()
    print(scvelo_all_cos_mean, pyro_all_cos_mean)

    dotsize = 3
    scale = 0.35
    arrow = 3.5
    fig, ax = plt.subplots(4, 8)
    fig.set_size_inches(14, 8)
    ax2 = ax[3]

    res = pd.DataFrame(
        {
            "X": adata_input_all.obsm["X_emb"][:, 0],
            "Y": adata_input_all.obsm["X_emb"][:, 1],
            "celltype": adata_input_all.obs.state_info,
        }
    )
    sns.scatterplot(
        data=res,
        x="X",
        y="Y",
        hue="celltype",
        palette=dict(
            zip(
                adata_input_all.obs.state_info.cat.categories,
                adata_input_all.uns["state_info_colors"],
            )
        ),
        s=dotsize,
        alpha=0.9,
        linewidth=0,
        legend="brief",
        ax=ax2[0],
    )
    ax2[0].set_title("Cell types", fontsize=7)
    scv.pl.velocity_embedding_grid(
        adata_input_all_clone,
        # scale=0.25,
        scale=scale,
        autoscale=True,
        show=False,
        s=dotsize,
        density=density,
        arrow_size=arrow,
        linewidth=1,
        vkey="clone_vector",
        basis="emb",
        ax=ax2[1],
        title="Clonal progression",
        color="gray",
        arrow_color="black",
        fontsize=7,
    )
    
    scv.pl.velocity_embedding_grid(
        # adata_input_all,
        adata_input_vel,
        show=False,
        s=dotsize,
        density=density,
        # scale=None,
        scale=scale,
        arrow_size=arrow,
        linewidth=1,
        basis="emb",
        ax=ax2[2],
        title="Scvelo",
        fontsize=7,
        color="gray",
        arrow_color="black",
        autoscale=True,
    )
    ax2[2].set_title("scVelo cosine similarity: %.2f" % scvelo_all_cos_mean, fontsize=7)
    plot_vector_field_uncertain(
        adata_input_all,
        embed_mean_all,
        embeds_radian_all,
        ax=ax2[3],
        cbar=True,
        fig=fig,
        # basis='emb', scale=0.0003, p_mass_min=0.1,
        # basis='emb', scale=0.001, p_mass_min=20,
        # basis='emb', scale=None, p_mass_min=1,
        basis="emb",
        scale=scale,
        p_mass_min=1,
        density=density,
        arrow_size=arrow,
        #cbar_pos=[0.46, 0.28, 0.1, 0.012],
        only_grid=False,
        autoscale=True,
    )
    ax2[4].set_title(
        "Pyro-Velocity cosine similarity: %.2f" % pyro_all_cos_mean, fontsize=7
    )
    scv.pl.scatter(
        adata_cospar,
        basis="emb",
        fontsize=7,
        color="fate_potency",
        cmap="inferno_r",
        show=False,
        ax=ax2[5],
        s=dotsize,
    )
    ax2[5].set_title("Clonal fate potency", fontsize=7)
    gold = adata_cospar[adata_input_all.obs_names.str.replace("-0", ""), :].obs.fate_potency
    select = ~np.isnan(gold)
    scv.pl.scatter(
        adata_input_all,
        c="latent_time",
        basis="emb",
        s=dotsize,
        cmap="inferno",
        ax=ax2[6],
        show=False,
        fontsize=7,
    )
    ax2[6].set_title(
        "Scvelo latent time\ncorrelation: %.2f"
        % spearmanr(-gold[select], adata_input_all.obs.latent_time.values[select])[0],
        fontsize=7,
    )
    adata_input_all.obs.cytotrace = adata_cytotrace.obs.cytotrace
    plot_posterior_time(
        posterior_samples,
        adata_input_all,
        ax=ax2[7],
        basis="emb",
        fig=fig,
        addition=False,
        position="right",
    )
    ax2[7].set_title(
        "Pyro-Velocity shared time\ncorrelation: %.2f"
        % spearmanr(
            -gold[select], posterior_samples["cell_time"].mean(0).flatten()[select]
        )[0],
        fontsize=7,
    )
    fig.savefig(
        figure3_plot_name,
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

    if os.path.isfile(conf.reports.figureS3.tif_path) and os.path.isfile(
        conf.reports.figureS3.svg_path
    ):
        logger.info(
            f"\n\nFigure 2 outputs already exist:\n\n"
            f"  see contents of: {conf.reports.figureS3.path}\n"
        )
    else:
        logger.info(f"\n\nPlotting figure S3\n\n")
        plots(conf, logger)


if __name__ == "__main__":
    main()
