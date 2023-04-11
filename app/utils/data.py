import altair as alt
import pandas as pd
import scvelo as scv


def generate_sample_data():
    adata = scv.datasets.simulation(
        random_seed=0,
        n_obs=100,
        n_vars=12,
        alpha=5,
        beta=0.5,
        gamma=0.3,
        alpha_=0,
        noise_model="gillespie",  # "normal" vs "gillespie"
    )

    return adata


def filter_var_counts_to_df(adata, min_spliced_counts, min_unspliced_counts):
    # create dataframes for spliced and unspliced counts
    spliced_df = pd.DataFrame(
        adata.layers["spliced"], index=adata.obs_names, columns=adata.var_names
    )
    unspliced_df = pd.DataFrame(
        adata.layers["unspliced"], index=adata.obs_names, columns=adata.var_names
    )

    # melt the dataframes to long format
    spliced_melted = spliced_df.reset_index().melt(
        id_vars="index", var_name="var_name", value_name="spliced"
    )
    unspliced_melted = unspliced_df.reset_index().melt(
        id_vars="index", var_name="var_name", value_name="unspliced"
    )

    # combine the dataframes
    df = spliced_melted.merge(unspliced_melted, on=["index", "var_name"])

    # rename the 'index' column
    df = df.rename(columns={"index": "obs_name"})

    spliced_var_gt_threshold = (spliced_df > min_spliced_counts).sum().sum()
    unspliced_var_gt_threshold = (unspliced_df > min_unspliced_counts).sum().sum()

    # filter the dataframe to include rows where the spliced or unspliced counts are greater than 0
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
