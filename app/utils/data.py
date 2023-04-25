import pickle
from typing import Any

import altair as alt
import numpy as np
import pandas as pd
import scvelo as scv
import zstandard as zstd


def load_compressed_pickle(file_path: str) -> Any:
    with open(file_path, "rb") as f:
        decompression_context = zstd.ZstdDecompressor()
        with decompression_context.stream_reader(f) as decompressor:
            obj = pickle.load(decompressor)
    return obj


def generate_sample_data():
    return scv.datasets.simulation(
        random_seed=0,
        n_obs=100,
        n_vars=12,
        alpha=5,
        beta=0.5,
        gamma=0.3,
        alpha_=0,
        noise_model="gillespie",  # "normal" vs "gillespie"
    )


def ensure_numpy_array(obj):
    return obj.toarray() if hasattr(obj, "toarray") else obj


def anndata_counts_to_df(adata):
    spliced_df = pd.DataFrame(
        ensure_numpy_array(adata.layers["raw_spliced"]),
        index=adata.obs_names,
        columns=adata.var_names,
    )
    unspliced_df = pd.DataFrame(
        ensure_numpy_array(adata.layers["raw_unspliced"]),
        index=adata.obs_names,
        columns=adata.var_names,
    )

    spliced_melted = spliced_df.reset_index().melt(
        id_vars="index", var_name="var_name", value_name="spliced"
    )
    unspliced_melted = unspliced_df.reset_index().melt(
        id_vars="index", var_name="var_name", value_name="unspliced"
    )

    df = spliced_melted.merge(unspliced_melted, on=["index", "var_name"])

    df = df.rename(columns={"index": "obs_name"})

    total_obs = adata.n_obs
    total_var = adata.n_vars

    max_spliced = adata.layers["raw_spliced"].max()
    max_unspliced = adata.layers["raw_unspliced"].max()

    return (
        df,
        total_obs,
        total_var,
        # max_spliced,
        # max_unspliced,
    )


def filter_var_counts_by_thresholds(
    df, spliced_count_thresholds, unspliced_count_thresholds
):
    spliced_var_gt_threshold = (
        (
            (df["spliced"] >= spliced_count_thresholds[0])
            & (df["spliced"] <= spliced_count_thresholds[1])
        )
        .sum()
        .sum()
    )

    unspliced_var_gt_threshold = (
        (
            (df["unspliced"] >= unspliced_count_thresholds[0])
            & (df["unspliced"] <= unspliced_count_thresholds[1])
        )
        .sum()
        .sum()
    )

    df_filtered = df[
        (
            (df["spliced"] >= spliced_count_thresholds[0])
            & (df["unspliced"] >= unspliced_count_thresholds[0])
            & (df["spliced"] <= spliced_count_thresholds[1])
            & (df["unspliced"] <= unspliced_count_thresholds[1])
        )
    ]

    return (
        df_filtered,
        spliced_var_gt_threshold,
        unspliced_var_gt_threshold,
    )


def filter_var_counts_by_threshold(df, min_spliced_counts, min_unspliced_counts):
    spliced_var_gt_threshold = (df["spliced"] >= min_spliced_counts).sum().sum()
    unspliced_var_gt_threshold = (df["unspliced"] >= min_unspliced_counts).sum().sum()

    df_filtered = df[
        (df["spliced"] >= min_spliced_counts)
        & (df["unspliced"] >= min_unspliced_counts)
    ]

    return (
        df_filtered,
        spliced_var_gt_threshold,
        unspliced_var_gt_threshold,
    )


def interactive_spliced_unspliced_plot(
    df, title, selected_vars=None, selected_obs=None
):
    if selected_vars is None:
        selected_vars = []
    if selected_obs is None:
        selected_obs = []

    df["highlight"] = df["var_name"].isin(selected_vars) | df["obs_name"].isin(
        selected_obs
    )

    color_condition = alt.condition(
        alt.datum.highlight,
        alt.value("green"),
        alt.value("gray"),
    )

    chart = (
        alt.Chart(df)
        .mark_circle()
        .encode(
            x=alt.X("spliced:Q", title="Spliced Counts"),
            y=alt.Y("unspliced:Q", title="Unspliced Counts"),
            tooltip=["spliced:Q", "unspliced:Q", "var_name:N", "obs_name:N"],
            color=color_condition,
            size=alt.Size("highlight:N", scale=alt.Scale(range=[15, 100]), legend=None),
        )
        .properties(
            title=title,
        )
        .interactive()
    )

    return chart


def interactive_spliced_unspliced_histogram(
    df, title, selected_vars=None, selected_obs=None
):
    if selected_vars is None:
        selected_vars = []
    if selected_obs is None:
        selected_obs = []

    df["highlight"] = df["var_name"].isin(selected_vars) | df["obs_name"].isin(
        selected_obs
    )

    number_of_histogram_bins = np.maximum(60, np.sqrt(len(df)))

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
        )
        .properties(
            title=title,
        )
        .configure(
            countTitle="counts",
        )
        .configure_legend(orient="top-right", titleOrient="left")
        .interactive()
    )

    return chart
