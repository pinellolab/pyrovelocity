import anndata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pyrovelocity.logging import configure_logging


__all__ = ["plot_parameter_posterior_distributions"]

logger = configure_logging(__name__)


def plot_parameter_posterior_distributions(
    posterior_samples,
    adata: anndata.AnnData,
    geneset,
    parameter_uncertainty_plot_path: str,
):
    fig, ax = plt.subplots(3, 1)
    fig.set_size_inches(18, 12)
    for index, kinetics in enumerate(["alpha", "beta", "gamma"]):
        # print(posterior_samples[kinetics].squeeze().shape)
        # print(np.isin(adata.var_names, list(geneset)).sum())
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
        # print(df_long.head())
        df_long["index"] = pd.Categorical(
            df_long["index"], categories=geneset, ordered=True
        )
        ax1 = sns.violinplot(
            x="index",
            y="value",
            data=df_long,
            ax=ax[index],
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
