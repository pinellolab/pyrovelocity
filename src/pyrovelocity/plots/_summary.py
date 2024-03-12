import matplotlib.pyplot as plt

from pyrovelocity.logging import configure_logging
from pyrovelocity.plots._genes import plot_gene_ranking
from pyrovelocity.plots._rainbow import rainbowplot
from pyrovelocity.plots._time import plot_posterior_time


logger = configure_logging(__name__)


__all__ = ["summarize_fig2_part2"]


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
