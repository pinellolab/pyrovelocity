import streamlit as st
from utils.config import get_app_config
from utils.data import filter_var_counts_to_df
from utils.data import generate_sample_data
from utils.data import interactive_spliced_unspliced_plot


PATH_PREFIX = "reproducibility/figures/"
cfg = get_app_config()


@st.cache_data(show_spinner=False, persist=True)
def load_data(PATH_PREFIX=PATH_PREFIX, cfg=cfg):
    import scvelo as scv

    return scv.read(PATH_PREFIX + cfg.model_training.pancreas_model2.trained_data_path)


# adata = generate_sample_data()
with st.spinner(f"loading {cfg.model_training.pancreas_model2.trained_data_path} ..."):
    adata = load_data()

spliced_threshold = 0
unspliced_threshold = 0
(
    df,
    total_obs,
    total_var,
    spliced_var_gt_threshold,
    unspliced_var_gt_threshold,
) = filter_var_counts_to_df(adata, spliced_threshold, unspliced_threshold)

# title = (
#     f"Spliced vs unspliced counts (obs: {total_obs}, "
#     + f"var: {total_var}, spliced > {spliced_threshold}: {spliced_var_gt_threshold}, "
#     + f"unspliced > 0: {unspliced_var_gt_threshold})"
# )


col_1, _, col_3 = st.columns([7, 1, 5])

# with col_1:
#     col11, col12 = st.columns([1, 1])

#     with col11:
#         obs_values = sorted(df["obs_name"].unique())
#         selected_obs = st.multiselect("Select cell(s)", obs_values)

#     with col12:
#         var_values = sorted(df["var_name"].unique())
#         selected_var = st.multiselect("Select gene(s)", var_values)

#     c = interactive_spliced_unspliced_plot(df, title, selected_var, selected_obs)
#     st.altair_chart(c, use_container_width=True)

with col_3:
    # if selected_var and selected_obs:
    #     display_df = df[
    #         df["var_name"].isin(selected_var) & df["obs_name"].isin(selected_obs)
    #     ]
    # elif selected_var:
    #     display_df = df[df["var_name"].isin(selected_var)]
    # elif selected_obs:
    #     display_df = df[df["obs_name"].isin(selected_obs)]
    # else:
    #     display_df = df

    display_df = df
    st.dataframe(display_df)
