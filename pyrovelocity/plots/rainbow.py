import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scvelo as scv
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable


def rainbowplot(
    volcano_data,
    adata,
    posterior_samples,
    fig=None,
    genes=None,
    data=["st", "ut"],
    cell_state="clusters",
    basis="umap",
    num_genes=5,
    add_line=True,
    negative=False,
    scvelo_colors=False,
) -> None:
    set_font_size(7)

    if genes is None:
        genes = get_genes(volcano_data, num_genes, negative)

    if fig is None:
        fig = create_figure()

    if scvelo_colors:
        colors = setup_scvelo_colors(adata, cell_state)
    else:
        colors = setup_colors(adata, cell_state)

    st, ut = get_posterior_samples(data, posterior_samples)

    number_of_genes = len(genes)

    subfigs = fig.subfigures(1, 2, wspace=0.0, width_ratios=[3, 1.5])

    ax = subfigs[0].subplots(number_of_genes, 2)
    ax_fig2 = subfigs[1].subplots(number_of_genes, 1)

    for n, gene in enumerate(genes):
        ress = get_data(gene, st, ut, adata, cell_state, posterior_samples)
        ax1 = ax[n, 1]
        ax2 = ax[n, 0]
        ax3 = ax_fig2[n]
        if n == 0:
            ax1.set_title("Rainbow plot", fontsize=7)
            ax2.set_title("Phase portrait", fontsize=7)
        plot_gene(ax1, ress, colors, add_line)
        scatterplot(ax2, ress, colors)
        (index,) = np.where(adata.var_names == gene)
        im = ax3.scatter(
            adata.obsm[f"X_{basis}"][:, 0],
            adata.obsm[f"X_{basis}"][:, 1],
            s=3,
            c=st[:, index].flatten(),
            cmap="RdBu_r",
        )
        set_colorbar(im, ax3, labelsize=5, fig=subfigs[1], rainbow=True)
        ax3.axis("off")
        set_labels(ax1, ax2, ax3, gene, number_of_genes, ress, n)

    sns.despine()
    subfigs[0].subplots_adjust(
        hspace=0.8, wspace=1.4, left=0.32, right=0.94, top=0.92, bottom=0.12
    )
    subfigs[1].subplots_adjust(
        hspace=0.8, wspace=0.4, left=0.2, right=0.7, top=0.92, bottom=0.08
    )
    subfigs[0].text(
        -0.025, 0.58, "unspliced expression", size=7, rotation="vertical", va="center"
    )
    subfigs[0].text(
        0.552, 0.58, "spliced expression", size=7, rotation="vertical", va="center"
    )
    return fig


def set_subfigure_titles(ax, n):
    if n == 0:
        ax[0].set_title("Rainbow plot", fontsize=7)
        ax[1].set_title("Phase portrait", fontsize=7)


def plot_gene_data(ax, ress, colors, add_line):
    plot_gene(ax[1], ress, colors, add_line)
    scatterplot(ax[0], ress, colors)


def scatter_gene(ax, adata, st, gene, basis):
    (index,) = np.where(adata.var_names == gene)
    return ax.scatter(
        adata.obsm[f"X_{basis}"][:, 0],
        adata.obsm[f"X_{basis}"][:, 1],
        s=3,
        c=st[:, index].flatten(),
        cmap="RdBu_r",
    )


def adjust_subfigure(subfig):
    subfig[0].subplots_adjust(
        hspace=0.8, wspace=1.4, left=0.32, right=0.94, top=0.92, bottom=0.12
    )
    subfig[1].subplots_adjust(
        hspace=0.8, wspace=0.4, left=0.2, right=0.7, top=0.92, bottom=0.08
    )

    subfig[0].text(
        -0.025, 0.58, "unspliced expression", size=7, rotation="vertical", va="center"
    )
    subfig[0].text(
        0.552, 0.58, "spliced expression", size=7, rotation="vertical", va="center"
    )

    sns.despine()


def pareto_frontier_genes(volcano_data, num_genes):
    sorted_data = volcano_data.sort_values(
        by=["mean_mae", "time_correlation"], ascending=[False, False]
    )

    pareto_frontier = sorted_data.iloc[:1]
    for i in range(1, len(sorted_data)):
        if (
            sorted_data["time_correlation"].iloc[i]
            >= pareto_frontier["time_correlation"].iloc[-1]
        ):
            pareto_frontier = pareto_frontier.append(sorted_data.iloc[i])

    if len(pareto_frontier) < num_genes:
        additional_genes = sorted_data.loc[
            ~sorted_data.index.isin(pareto_frontier.index)
        ].head(num_genes - len(pareto_frontier))
        pareto_frontier = pareto_frontier.append(additional_genes)

    pareto_frontier = pareto_frontier.sort_values(
        by="time_correlation", ascending=False
    )

    return pareto_frontier.index.tolist()


def set_font_size(size: int):
    matplotlib.rcParams.update({"font.size": size})


def create_figure():
    return plt.figure(figsize=(5.5, 4.5))


def get_genes(volcano_data, num_genes, negative):
    return (
        volcano_data.sort_values("mean_mae", ascending=False)
        .head(300)
        .sort_values("time_correlation", ascending=negative)
        .head(num_genes)
        .index
    )


def setup_scvelo_colors(adata, cell_state, basis):
    scv.pl.scatter(
        adata,
        basis=basis,
        fontsize=7,
        legend_loc="on data",
        legend_fontsize=7,
        color=cell_state,
        show=False,
    )
    return dict(
        zip(adata.obs.state_info.cat.categories, adata.uns["state_info_colors"])
    )


def setup_colors(adata, cell_state):
    clusters = adata.obs.loc[:, cell_state]
    return dict(
        zip(
            clusters.cat.categories,
            sns.color_palette("deep", clusters.cat.categories.shape[0]),
        )
    )


def get_posterior_samples(data, posterior_samples):
    if (data[0] in posterior_samples) and (data[1] in posterior_samples):
        st = posterior_samples[data[0]].mean(0).squeeze()
        ut = posterior_samples[data[1]].mean(0).squeeze()
    else:
        st = posterior_samples["st_mean"]
        ut = posterior_samples["ut_mean"]
    return st, ut


def get_data(gene, st, ut, adata, cell_state, posterior_samples):
    (index,) = np.where(adata.var_names == gene)
    pos_mean_time = posterior_samples["cell_time"].mean(0).flatten()
    ress = pd.DataFrame(
        {
            "cell_time": pos_mean_time / pos_mean_time.max(),
            "cell_type": adata.obs[cell_state].values,
            "spliced": st[:, index].flatten(),
            "unspliced": ut[:, index].flatten(),
        }
    )
    return ress.sort_values("cell_time")


def plot_gene(ax1, ress, colors, add_line):
    sns.scatterplot(
        x="cell_time",
        y="spliced",
        data=ress,
        alpha=0.1,
        linewidth=0,
        edgecolor="none",
        hue="cell_type",
        palette=colors,
        ax=ax1,
        marker="o",
        legend=False,
        s=5,
    )
    if add_line:
        for row in range(ress.shape[0]):
            ax1.vlines(
                x=ress.cell_time[row],
                ymin=0,
                ymax=ress.spliced[row],
                colors=colors[ress.cell_type[row]],
                alpha=0.1,
            )


def set_labels(ax1, ax2, ax3, gene, ngenes, ress, n):
    if n == 0:
        ax3.set_title("Denoised spliced", fontsize=7)
    if n == ngenes - 1:
        ax1.set_xlabel("shared time", fontsize=7)
        ax2.set_xlabel("spliced", fontsize=7)
    else:
        ax1.set_xlabel("")
        ax2.set_xlabel("")
    ax1.set_ylabel("")
    ax2.set_ylabel(gene, fontsize=7, rotation=0, labelpad=23)
    t = [0, round(ress["cell_time"].max(), 5)]
    t_label = ["0", "%.1E" % ress["cell_time"].max()]
    ax1.set_xticks(t, t_label, fontsize=7)
    t = [0, round(ress["spliced"].max(), 5)]
    t_label = ["0", "%.1E" % ress["spliced"].max()]
    ax1.set_yticks(t, t_label, fontsize=7)
    t = [0, round(ress["unspliced"].max(), 5)]
    t_label = ["0", "%.1E" % ress["unspliced"].max()]
    ax2.set_yticks(t, t_label, fontsize=7)
    t = [0, round(ress["spliced"].max(), 5)]
    t_label = ["0", "%.1E" % ress["spliced"].max()]
    ax2.set_xticks(t, t_label, fontsize=7)


def scatterplot(ax2, ress, colors):
    sns.scatterplot(
        x="spliced",
        y="unspliced",
        data=ress,
        alpha=0.4,
        linewidth=0,
        edgecolor="none",
        hue="cell_type",
        palette=colors,
        ax=ax2,
        marker="o",
        legend=False,
        s=3,
    )


def set_colorbar(
    smp,
    ax,
    orientation="vertical",
    labelsize=None,
    fig=None,
    position="right",
    rainbow=False,
):
    from matplotlib.ticker import MaxNLocator
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    if position == "right" and (not rainbow):
        cax = inset_axes(ax, width="2%", height="30%", loc=4, borderpad=0)
        cb = fig.colorbar(smp, orientation=orientation, cax=cax)
    else:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes(position, size="8%", pad=0.08)
        cb = fig.colorbar(smp, cax=cax, orientation=orientation, shrink=0.4)

    cb.ax.tick_params(labelsize=labelsize)
    cb.set_alpha(1)
    cb.draw_all()
    cb.locator = MaxNLocator(nbins=2, integer=True)

    if position == "left":
        cb.ax.yaxis.set_ticks_position("left")
    cb.update_ticks()
