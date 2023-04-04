import altair as alt
import pandas as pd
import scanpy as sc
import scvelo as scv
import streamlit as st
from google.cloud import storage


def generate_sample_data():

    adata = scv.datasets.simulation(
        random_seed=0,
        n_obs=100,
        n_vars=12,
        alpha=5,
        beta=0.5,
        gamma=0.3,
        alpha_=0,
        noise_model="normal",  # vs "gillespie" broken in 0.2.4
    )

    return adata


def filter_var_counts_to_df(adata, min_spliced_counts, min_unspliced_counts):
    # Create DataFrames for spliced and unspliced counts
    spliced_df = pd.DataFrame(
        adata.layers["spliced"], index=adata.obs_names, columns=adata.var_names
    )
    unspliced_df = pd.DataFrame(
        adata.layers["unspliced"], index=adata.obs_names, columns=adata.var_names
    )

    # Melt the DataFrames to long format
    spliced_melted = spliced_df.reset_index().melt(
        id_vars="index", var_name="var_name", value_name="spliced"
    )
    unspliced_melted = unspliced_df.reset_index().melt(
        id_vars="index", var_name="var_name", value_name="unspliced"
    )

    # Combine the DataFrames
    df = spliced_melted.merge(unspliced_melted, on=["index", "var_name"])

    # Rename the 'index' column to 'obs_name'
    df = df.rename(columns={"index": "obs_name"})

    spliced_var_gt_threshold = (spliced_df > min_spliced_counts).sum().sum()
    unspliced_var_gt_threshold = (unspliced_df > min_unspliced_counts).sum().sum()

    # Filter the DataFrame to include only rows where either the spliced or unspliced count is greater than 0
    df_filtered = df[
        (df["spliced"] > min_spliced_counts) | (df["unspliced"] > min_unspliced_counts)
    ]

    total_obs = adata.n_obs
    total_var = adata.n_vars

    return (
        df_filtered,
        total_obs,
        total_var,
        spliced_var_gt_threshold,
        unspliced_var_gt_threshold,
    )


def interactive_spliced_unspliced_plot(df, title):
    chart = (
        alt.Chart(df)
        .mark_circle()
        .encode(
            x=alt.X("spliced:Q", title="Spliced Counts"),
            y=alt.Y("unspliced:Q", title="Unspliced Counts"),
            tooltip=["spliced:Q", "unspliced:Q", "var_name:N", "obs_name:N"],
        )
        .properties(
            title=title,
        )
        .interactive()
    )

    return chart


def st_show():

    adata = generate_sample_data()
    spliced_threshold = 0
    unspliced_threshold = 0
    (
        df,
        total_obs,
        total_var,
        spliced_var_gt_threshold,
        unspliced_var_gt_threshold,
    ) = filter_var_counts_to_df(adata, spliced_threshold, unspliced_threshold)

    title = (
        f"Spliced vs unspliced counts (obs: {total_obs}, "
        + f"var: {total_var}, spliced > {spliced_threshold}: {spliced_var_gt_threshold}, "
        + f"unspliced > 0: {unspliced_var_gt_threshold})"
    )

    c = interactive_spliced_unspliced_plot(df, title)
    st.altair_chart(c, use_container_width=True)

    st.write(df)
    # scv.set_figure_params(vector_friendly=False, transparent=False, facecolor="white")
    # # Plot initial scatter plots
    # axs = scv.pl.scatter(
    #     adata,
    #     ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"],
    #     ncols=4,
    #     nrows=3,
    #     xlim=[-1, 20],
    #     ylim=[-1, 20],
    #     show=False,
    #     dpi=300,
    #     figsize=(7, 5),
    # )
    # st.pyplot(axs[0].get_figure(), format="png", dpi=300)

    # # Recover dynamics and plot
    # scv.tl.recover_dynamics(adata)

    # axs2 = scv.pl.scatter(
    #     adata,
    #     ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"],
    #     ncols=4,
    #     nrows=3,
    #     xlim=[-1, 20],
    #     ylim=[-1, 20],
    #     color=["true_t"],
    #     show=False,
    #     dpi=300,
    #     figsize=(7, 5),
    # )
    # st.pyplot(axs2[0].get_figure(), format="png", dpi=300)

    # c = (
    #     alt.Chart(iris)
    #     .mark_point()
    #     .encode(x="petalLength", y="petalWidth", color="species")
    # )
    # st.altair_chart(c, use_container_width=True)
