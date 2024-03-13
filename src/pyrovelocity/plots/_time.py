import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

from pyrovelocity.plots._common import set_colorbar


__all__ = ["plot_posterior_time"]


def plot_posterior_time(
    posterior_samples,
    adata,
    ax=None,
    fig=None,
    basis="umap",
    addition=True,
    position="left",
    s=3,
):
    if addition:
        sns.set_style("white")
        sns.set_context("paper", font_scale=1)
        matplotlib.rcParams.update({"font.size": 7})
        plt.figure()
        plt.hist(posterior_samples["cell_time"].mean(0), bins=100, label="test")
        plt.xlabel("mean of cell time")
        plt.ylabel("frequency")
        plt.title("Histogram of cell time posterior samples")
        plt.legend()
    pos_mean_time = posterior_samples["cell_time"].mean(0)
    adata.obs["cell_time"] = pos_mean_time / pos_mean_time.max()

    if ax is None:
        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(2.36, 2)
    im = ax.scatter(
        adata.obsm[f"X_{basis}"][:, 0],
        adata.obsm[f"X_{basis}"][:, 1],
        s=s,
        alpha=0.4,
        c=adata.obs["cell_time"],
        cmap="inferno",
        linewidth=0,
    )
    set_colorbar(im, ax, labelsize=5, fig=fig, position=position)
    ax.axis("off")
    if "cytotrace" in adata.obs.columns:
        ax.set_title(
            "Pyro-Velocity shared time\ncorrelation with Cytotrace: %.2f"
            % (
                spearmanr(
                    adata.obs["cell_time"].values,
                    1 - adata.obs.cytotrace.values,
                )[0]
            ),
            fontsize=7,
        )
    else:
        ax.set_title("Pyro-Velocity shared time\n", fontsize=7)
