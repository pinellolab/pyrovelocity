import streamlit as st
from utils.data import filter_var_counts_to_df
from utils.data import generate_sample_data
from utils.data import interactive_spliced_unspliced_histogram


adata = generate_sample_data()
adata.layers["raw_unspliced"] = adata.layers["unspliced"]
adata.layers["raw_spliced"] = adata.layers["spliced"]


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

title = (
    f"Spliced vs unspliced counts (obs: {total_obs}, "
    + f"var: {total_var}, spliced > {spliced_threshold}: {spliced_var_gt_threshold}, "
    + f"unspliced > 0: {unspliced_var_gt_threshold})"
)


@st.cache_data(show_spinner=True, persist=True)
def generate_histogram(df, title, selected_var, selected_obs):
    from utils.data import interactive_spliced_unspliced_histogram
    from utils.data import interactive_spliced_unspliced_plot

    return interactive_spliced_unspliced_histogram(
        df, title, selected_var, selected_obs
    )


obs_values = sorted(df["obs_name"].unique())
var_values = sorted(df["var_name"].unique())


@st.cache_data(show_spinner=True, persist=True)
def filter_dataframe_obs_var(df, selected_var, selected_obs):
    if selected_var and selected_obs:
        filtered_df = df[
            df["var_name"].isin(selected_var) & df["obs_name"].isin(selected_obs)
        ]
    elif selected_var:
        filtered_df = df[df["var_name"].isin(selected_var)]
    elif selected_obs:
        filtered_df = df[df["obs_name"].isin(selected_obs)]
    else:
        filtered_df = df
    return filtered_df


col_1, _, col_3 = st.columns([6.25, 0.25, 3.5])

with col_1:
    col11, col12 = st.columns([1, 1])

    with col11:
        selected_obs = st.multiselect("Select cell(s)", obs_values)

    with col12:
        selected_var = st.multiselect("Select gene(s)", var_values)

    filtered_df = filter_dataframe_obs_var(df, selected_var, selected_obs)

    spliced_unspliced_histogram_chart = generate_histogram(
        filtered_df, title, selected_var, selected_obs
    )
    spliced_unspliced_histogram_chart.height = 738
    st.altair_chart(
        spliced_unspliced_histogram_chart, use_container_width=True, theme=None
    )

with col_3:
    paginate = st.checkbox("paginate", value=True)

    if paginate:
        page_size = 20
        last_page = len(filtered_df) // page_size

        def next_page(last_page=last_page):
            if st.session_state.page + 1 > last_page:
                st.session_state.page = 1
            else:
                st.session_state.page += 1

        def prev_page(last_page=last_page):
            if st.session_state.page <= 1:
                st.session_state.page = last_page
            else:
                st.session_state.page -= 1

        prev, next, current, first, last, _ = st.columns([1, 1, 2.2, 1, 1, 3.8])

        if next.button("⏵"):
            next_page()

        if prev.button("⏴"):
            prev_page()

        if first.button("⏮"):
            st.session_state.page = 1

        if last.button("⏭"):
            st.session_state.page = last_page

        with current:
            st.selectbox(
                "page",
                range(1, last_page + 1, 1),
                key="page",
                label_visibility="collapsed",
            )

        page_start_index = st.session_state.page * page_size
        page_end_index = page_start_index + page_size

        display_df = filtered_df.iloc[page_start_index:page_end_index]
    else:
        display_df = filtered_df

    with st.spinner("loading dataframe"):
        st.dataframe(
            data=display_df,
            height=738,
            use_container_width=True,
        )
        st.text(f"showing {len(display_df)} of {len(filtered_df)} rows")
