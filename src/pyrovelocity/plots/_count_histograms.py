import altair as alt
import numpy as np
import polars as pl
from altair.vegalite.v5.api import Chart as AltairChart
from anndata import AnnData
from beartype import beartype
from scipy.sparse import csr_matrix

__all__ = ["plot_spliced_unspliced_histogram"]


alt.data_transformers.enable("vegafusion")


@beartype
def plot_spliced_unspliced_histogram(
    adata: AnnData,
    spliced_layer: str = "spliced",
    unspliced_layer: str = "unspliced",
    min_count: int = 3,
    max_count: int = 200,
    title: str = "",
    default_font_size: int = 24,
    plot_width: int = 700,
    plot_height: int = 700,
) -> AltairChart:
    spliced_matrix = csr_matrix(adata.layers[spliced_layer])
    unspliced_matrix = csr_matrix(adata.layers[unspliced_layer])

    non_zero = spliced_matrix.nonzero()

    spliced_values = spliced_matrix[non_zero].A1
    unspliced_values = unspliced_matrix[non_zero].A1

    df = pl.DataFrame(
        {"spliced": spliced_values, "unspliced": unspliced_values}
    )

    df = df.filter(
        (pl.col("spliced") >= min_count)
        & (pl.col("unspliced") >= min_count)
        & (pl.col("spliced") <= max_count)
        & (pl.col("unspliced") <= max_count)
    )

    number_of_histogram_bins = int(np.maximum(60, np.sqrt(len(df))))

    chart = (
        alt.Chart(df)
        .mark_rect()
        .encode(
            x=alt.X(
                "spliced:Q",
                title="Spliced Counts",
                bin=alt.Bin(maxbins=number_of_histogram_bins),
                axis=alt.Axis(tickMinStep=1),
            ),
            y=alt.Y(
                "unspliced:Q",
                title="Unspliced Counts",
                bin=alt.Bin(maxbins=number_of_histogram_bins),
                axis=alt.Axis(tickMinStep=1),
            ),
            color=alt.Color("count():Q", scale=alt.Scale(scheme="greenblue")),
            tooltip=["spliced", "unspliced", "count()"],
        )
        .properties(
            title=title,
            width=plot_width,
            height=plot_height,
        )
        .configure(
            countTitle="counts",
        )
        .configure_axis(
            labelFontSize=default_font_size,
            titleFontSize=default_font_size,
            titleFontWeight="normal",
        )
        .configure_title(fontSize=default_font_size)
        .configure_legend(
            orient="top-right",
            titleOrient="left",
            titleFontSize=default_font_size,
            titleFontWeight="normal",
            labelFontSize=default_font_size,
        )
        .interactive()
    )

    return chart
