from os import PathLike
from typing import Dict, List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from anndata import AnnData
from beartype import beartype
from beartype.typing import Optional
from matplotlib.figure import FigureBase
from matplotlib.gridspec import SubplotSpec

from pyrovelocity.logging import configure_logging

__all__ = ["plot_parameter_posterior_distributions"]

logger = configure_logging(__name__)


@beartype
def tex_or_plain(
    tex_str: str,
    plain_str: str,
) -> str:
    """
    Returns the TeX-formatted string if text.usetex is True,
    otherwise returns the plain string.
    """
    return tex_str if matplotlib.rcParams["text.usetex"] else plain_str


DEFAULT_PARAMETER_LABEL_MAPPINGS = {
    "alpha": tex_or_plain(r"$\alpha$", "α"),
    "beta": tex_or_plain(r"$\beta$", "β"),
    "gamma": tex_or_plain(r"$\gamma$", "γ"),
    "u_offset": tex_or_plain(r"$u_0$", "u₀"),
    "s_offset": tex_or_plain(r"$s_0$", "s₀"),
    "t0": tex_or_plain(r"$t_0$", "t₀"),
}


@beartype
def plot_parameter_posterior_distributions(
    posterior_samples: Dict[str, np.ndarray],
    adata: AnnData,
    geneset: List[str],
    parameter_names: List[str]
    | Dict[str, str] = DEFAULT_PARAMETER_LABEL_MAPPINGS,
    fig: Optional[FigureBase] = None,
    gs: Optional[SubplotSpec] = None,
    save_plot: bool = False,
    parameter_uncertainty_plot: PathLike | str = "parameter_uncertainty.pdf",
    default_fontsize: int = 7,
) -> Optional[FigureBase]:
    if isinstance(parameter_names, list):
        parameter_names = {param: param for param in parameter_names}

    parameters = [
        parameter
        for parameter in parameter_names.keys()
        if parameter in posterior_samples.keys()
    ]
    main_title = r"Density estimates from $\log$-posterior samples"
    if len(parameters) > 3:
        nrows = (len(parameters) + 1) // 2
        ncols = 2
    else:
        nrows = len(parameters)
        ncols = 1
    if gs is None:
        fig, ax = plt.subplots(len(parameters), 1)
        fig.set_size_inches(18, len(parameters) * 4)
    else:
        sgs = gs.subgridspec(
            nrows=nrows + 1,
            ncols=ncols,
            height_ratios=[0.1] + [1] * nrows,
        )
        title_ax = fig.add_subplot(sgs[0, :])
        title_ax.axis("off")
        title_ax.set_label("parameter_posteriors")
        title_ax.text(
            0.5,
            0.5,
            main_title,
            ha="center",
            va="center",
            fontsize=default_fontsize + 1,
            fontweight="bold",
            transform=title_ax.transAxes,
        )

    for index, parameter in enumerate(parameters):
        if gs is not None:
            col = index // nrows
            row = index % nrows + 1  # +1 because the first row is for the title
            ax1 = fig.add_subplot(sgs[row, col])
        else:
            row = index
            ax1 = ax[index]
        ax1.set_label("parameter_posteriors")
        df = pd.DataFrame(
            np.log(
                posterior_samples[parameter].squeeze()[
                    :, np.isin(adata.var_names, list(geneset))
                ],
            ),
            columns=adata.var_names[np.isin(adata.var_names, list(geneset))],
        )
        df_long = df.melt(var_name="index", value_name="value")
        logger.debug(df_long.head())
        df_long["index"] = pd.Categorical(
            df_long["index"], categories=geneset, ordered=True
        )

        can_cast_all = (
            df_long["index"].astype(str).apply(lambda x: x.isdigit()).all()
        )
        if can_cast_all:
            df_long["index"] = (
                df_long["index"].astype(str).apply(lambda x: f"g_{int(x)}")
            )
            logger.warning(
                f"Converted integer index to string 'g_int' for {parameter}"
            )
        else:
            pass

        dark_orange = "#ff6a14"
        sns.violinplot(
            x="index",
            y="value",
            color=dark_orange,
            linewidth=0,
            data=df_long,
            ax=ax1,
            inner="box",
            inner_kws=dict(
                box_width=1.5,
                whis_width=0.75,
                color=".8",
            ),
        )

        ax1.tick_params(axis="both", which="major", labelsize=default_fontsize)
        ax1.tick_params(axis="x", which="minor", bottom=False, top=False)

        if row < nrows:
            ax1.set_xticklabels([])
            ax1.set_xlabel("")
        else:
            ax1.set_xlabel("")
            truncated_labels = [
                label.get_text()[:7] for label in ax1.get_xticklabels()
            ]
            ax1.set_xticklabels(
                labels=truncated_labels,
                rotation=0,
                ha="center",
                fontdict={"fontsize": 5},
            )

        ax1.set_ylabel(parameter_names[parameter], fontsize=default_fontsize)

    if save_plot and fig is not None:
        fig.tight_layout()
        for ext in ["", ".png"]:
            fig.savefig(
                f"{parameter_uncertainty_plot}{ext}",
                facecolor=fig.get_facecolor(),
                bbox_inches="tight",
                edgecolor="none",
                dpi=300,
            )
        plt.close(fig)
    return fig
