import streamlit as st
from utils.data import filter_var_counts_to_df
from utils.data import generate_sample_data
from utils.data import interactive_spliced_unspliced_plot


adata = generate_sample_data()
adata.layers["raw_unspliced"] = adata.layers["unspliced"]
adata.layers["raw_spliced"] = adata.layers["spliced"]

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

st.dataframe(df)

# from google.cloud import storage
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
