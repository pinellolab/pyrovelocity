import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from adjustText import adjust_text
from matplotlib import gridspec

from pyrovelocity.analyze import compute_volcano_data
from pyrovelocity.logging import configure_logging


__all__ = ["plot_gene_ranking"]

logger = configure_logging(__name__)


def plot_gene_ranking(
    posterior_samples,
    adata,
    ax=None,
    time_correlation_with="s",
    selected_genes=None,
    assemble=False,
    data="correlation",
    negative=False,
    adjust_text_bool=False,
    show_marginal_histograms=False,
) -> None:
    if selected_genes is not None:
        assert isinstance(selected_genes, (tuple, list))
        assert isinstance(selected_genes[0], str)
        volcano_data = posterior_samples[0]["gene_ranking"]
        genes = selected_genes
    elif "u" in posterior_samples[0]:
        volcano_data, genes = compute_volcano_data(
            posterior_samples,
            adata,
            time_correlation_with,
            selected_genes,
            negative,
        )
    else:
        volcano_data = posterior_samples[0]["gene_ranking"]
        genes = posterior_samples[0]["genes"]

    adjust_text_compatible = is_adjust_text_compatible()
    fig = None

    if data == "correlation":
        defaultfontsize = 7
        defaultdotsize = 3
        plot_title = "Pyro-Velocity genes"

        if show_marginal_histograms:
            time_corr_hist, time_corr_bins = np.histogram(
                volcano_data["time_correlation"], bins="auto", density=False
            )
            mean_mae_hist, mean_mae_bins = np.histogram(
                volcano_data["mean_mae"], bins="auto", density=False
            )

            fig = plt.figure(figsize=(10, 10))
            # ax_scatter = plt.subplot2grid((3, 3), (1, 0), colspan=2, rowspan=2)
            # ax_hist_x = plt.subplot2grid((3, 3), (0, 0), colspan=2)
            # ax_hist_y = plt.subplot2grid((3, 3), (1, 2), rowspan=2)
            gs = gridspec.GridSpec(
                3, 3, width_ratios=[2, 2, 1], height_ratios=[1, 2, 2]
            )
            ax_scatter = plt.subplot(gs[1:, :2])
            ax_hist_x = plt.subplot(gs[0, :2])
            ax_hist_y = plt.subplot(gs[1:, 2])

            # time histogram
            ax_hist_x.bar(
                time_corr_bins[:-1],
                time_corr_hist,
                width=np.diff(time_corr_bins),
                align="edge",
            )

            # MAE histogram
            ax_hist_y.barh(
                mean_mae_bins[:-1],
                mean_mae_hist,
                height=np.diff(mean_mae_bins),
                align="edge",
            )
            ax_hist_x.tick_params(axis="x", labelbottom=False)
            ax_hist_y.tick_params(axis="y", labelleft=False)

            defaultfontsize = 14
            defaultdotsize = 12
            plot_title = ""
            ax = ax_scatter

        sns.scatterplot(
            x="time_correlation",
            y="mean_mae",
            hue="selected genes",
            data=volcano_data,
            s=defaultdotsize,
            linewidth=0,
            ax=ax,
            legend=False,
            alpha=0.3,
        )
        ax.set_title(plot_title, fontsize=defaultfontsize)
        ax.set_xlabel(
            "shared time correlation\nwith spliced expression",
            fontsize=defaultfontsize,
        )
        ax.set_ylabel("negative mean\nabsolute error", fontsize=defaultfontsize)
        sns.despine()
        ax.tick_params(labelsize=defaultfontsize - 1)

        texts = []
        for i, g in enumerate(genes):
            ax.scatter(
                volcano_data.loc[g, :].time_correlation,
                volcano_data.loc[g, :].mean_mae,
                s=15,
                color="red",
                marker="*",
            )
            texts.append(
                ax.text(
                    volcano_data.loc[g, :].time_correlation,
                    volcano_data.loc[g, :].mean_mae,
                    g,
                    fontsize=defaultfontsize - 2,
                    color="black",
                    ha="center",
                    va="center",
                )
            )
            if not assemble and adjust_text_compatible:
                # TODO: remove unused code in plot_gene_ranking
                # if i % 2 == 0:
                #     offset = 10 + i * 5
                # else:
                #     offset = -10 - i * 5
                # if i % 2 == 0:
                #     offset_y = -10 + i * 5
                # else:
                #     offset_y = -10 + i * 5
                if not adjust_text_bool:
                    adjust_text(
                        texts,
                        arrowprops=dict(arrowstyle="-", color="red", alpha=0.5),
                        ha="center",
                        va="bottom",
                        ax=ax,
                    )
                else:
                    adjust_text(
                        texts,
                        precision=0.001,
                        expand_text=(1.01, 1.05),
                        expand_points=(1.01, 1.05),
                        force_text=(0.01, 0.25),
                        force_points=(0.01, 0.25),
                        arrowprops=dict(
                            arrowstyle="-", color="blue", alpha=0.6
                        ),
                        ax=ax,
                    )
    else:
        sns.scatterplot(
            x="switching",
            y="mean_mae",
            hue="selected genes",
            data=volcano_data,
            s=3,
            linewidth=0,
            ax=ax,
            legend=False,
            alpha=0.3,
        )
        ax.set_title("Pyro-Velocity genes", fontsize=7)
        ax.set_xlabel("gene switching time", fontsize=7)
        ax.set_ylabel("negative mean\nabsolute error", fontsize=7)
        sns.despine()

    return volcano_data, fig


def is_adjust_text_compatible():
    """Check if the current backend supports adjust_text."""
    try:
        plt.figure()
        test_text = plt.text(0.5, 0.5, "test")
        plt.close()
        adjust_text([test_text], autoalign="y")
        return True
    except Exception as e:
        logger.warning(
            f"adjust_text may not be compatible with the current backend: {e}"
        )
        return False
