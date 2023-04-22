import streamlit as st
from utils.config import get_app_config


# from utils.data import filter_var_counts_to_df
# from utils.data import interactive_spliced_unspliced_plot


PATH_PREFIX = "reproducibility/figures/"
cfg = get_app_config()


@st.cache_data(show_spinner=False, persist=True)
def load_data(PATH_PREFIX=PATH_PREFIX, cfg=cfg):
    import scvelo as scv

    return scv.read(PATH_PREFIX + cfg.model_training.pancreas_model2.trained_data_path)


with st.spinner(f"loading {cfg.model_training.pancreas_model2.trained_data_path} ..."):
    adata = load_data()

spliced_threshold = 0
unspliced_threshold = 0


@st.cache_data(show_spinner=False, persist=True)
def extract_dataframe(
    _adata=adata,
    spliced_threshold=spliced_threshold,
    unspliced_threshold=unspliced_threshold,
):
    from utils.data import filter_var_counts_to_df

    return filter_var_counts_to_df(_adata, spliced_threshold, unspliced_threshold)


(
    df,
    total_obs,
    total_var,
    spliced_var_gt_threshold,
    unspliced_var_gt_threshold,
) = extract_dataframe(adata, spliced_threshold, unspliced_threshold)

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

    if "page" not in st.session_state:
        st.session_state.page = 0

    page_size = 20
    display_df = df
    last_page = len(display_df) // page_size

    def next_page(last_page=last_page):
        if st.session_state.page + 1 > last_page:
            st.session_state.page = 0
        else:
            st.session_state.page += 1

    def prev_page(last_page=last_page):
        if st.session_state.page < 1:
            st.session_state.page = last_page
        else:
            st.session_state.page -= 1

    prev, next, current, first, last, _ = st.columns([1, 1, 2.5, 1, 1, 3.5])

    if next.button("⏵"):
        next_page()

    if prev.button("⏴"):
        prev_page()

    if first.button("⏮"):
        st.session_state.page = 0

    if last.button("⏭"):
        st.session_state.page = last_page

    current.write(f"{st.session_state.page + 1} of {last_page}")

    page_start_index = st.session_state.page * page_size
    page_end_index = page_start_index + page_size

    page_df = display_df.iloc[page_start_index:page_end_index]
    st.dataframe(
        data=page_df,
        # width=450,
        height=738,
        use_container_width=True,
    )
