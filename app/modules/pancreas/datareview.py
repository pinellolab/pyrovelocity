import streamlit as st
from utils.config import get_app_config


PATH_PREFIX = "reproducibility/figures/"
cfg = get_app_config()


@st.cache_data(show_spinner="loading dataframe", persist=True)
def load_pancreas_df(PATH_PREFIX=PATH_PREFIX, cfg=cfg):
    from utils.data import load_compressed_pickle

    return load_compressed_pickle(
        PATH_PREFIX + cfg.reports.model_summary.pancreas_model2.dataframe_path
    )


(
    df,
    total_obs,
    total_var,
    max_spliced,
    max_unspliced,
) = load_pancreas_df()


if "pancreas_spliced_threshold" not in st.session_state:
    st.session_state.pancreas_spliced_threshold = (2, int(max_spliced * 0.5))

if "pancreas_unspliced_threshold" not in st.session_state:
    st.session_state.pancreas_unspliced_threshold = (2, int(max_unspliced * 0.5))


@st.cache_data(show_spinner="filtering dataframe by count thresholds", persist=True)
def filter_count_thresholds_from_pancreas_df(
    df, spliced_threshold, unspliced_threshold
):
    from utils.data import filter_var_counts_by_thresholds

    return filter_var_counts_by_thresholds(df, spliced_threshold, unspliced_threshold)


(
    df_thresholded,
    spliced_var_gt_threshold,
    unspliced_var_gt_threshold,
) = filter_count_thresholds_from_pancreas_df(
    df,
    st.session_state.pancreas_spliced_threshold,
    st.session_state.pancreas_unspliced_threshold,
)


@st.cache_data(show_spinner="generating histogram", persist=True)
def generate_pancreas_histogram(df, title, selected_var, selected_obs):
    from utils.data import interactive_spliced_unspliced_histogram

    chart = interactive_spliced_unspliced_histogram(
        df, title, selected_var, selected_obs
    )
    chart.height = 738
    return chart


obs_values = sorted(df["obs_name"].unique())
var_values = sorted(df["var_name"].unique())


@st.cache_data(show_spinner="filtering dataframe by selected cells/genes", persist=True)
def filter_obs_vars_from_pancreas_df(df, selected_var, selected_obs):
    if selected_var and selected_obs:
        return df[df["var_name"].isin(selected_var) & df["obs_name"].isin(selected_obs)]
    elif selected_var:
        return df[df["var_name"].isin(selected_var)]
    elif selected_obs:
        return df[df["obs_name"].isin(selected_obs)]
    else:
        return df


def next_page(last_page):
    if st.session_state.pancreas_page + 1 > last_page:
        st.session_state.pancreas_page = 1
    else:
        st.session_state.pancreas_page += 1


def prev_page(last_page):
    if st.session_state.pancreas_page <= 1:
        st.session_state.pancreas_page = last_page
    else:
        st.session_state.pancreas_page -= 1


col_1, _, col_3 = st.columns([6.25, 0.25, 3.5])

with col_1:
    col11, col12 = st.columns([1, 1])

    with col11:
        selected_obs = st.multiselect("Select cell(s)", obs_values)

    with col12:
        selected_var = st.multiselect("Select gene(s)", var_values)

    filtered_df = filter_obs_vars_from_pancreas_df(
        df_thresholded, selected_var, selected_obs
    )


page_size = 20
last_page = max(1, len(filtered_df) // page_size)

with col_3:
    if paginate := st.checkbox("paginate", value=True):
        prev, next, current, first, last, _ = st.columns([1, 1, 2.2, 1, 1, 3.8])

        if next.button("⏵"):
            next_page(last_page)

        if prev.button("⏴"):
            prev_page(last_page)

        if first.button("⏮"):
            st.session_state.pancreas_page = 1

        if last.button("⏭"):
            st.session_state.pancreas_page = last_page

        with current:
            with st.spinner("loading page number"):
                st.selectbox(
                    "page",
                    range(1, last_page + 1),
                    key="pancreas_page",
                    label_visibility="collapsed",
                )

        page_start_index = (st.session_state.pancreas_page - 1) * page_size
        page_end_index = page_start_index + page_size

        with st.spinner("computing display dataframe"):
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

with col_1:
    spliced_unspliced_histogram_chart = generate_pancreas_histogram(
        filtered_df, "Spliced vs unspliced counts", selected_var, selected_obs
    )

    with st.spinner("loading histogram"):
        st.altair_chart(
            spliced_unspliced_histogram_chart,
            use_container_width=True,
            theme="streamlit",
        )

    summary_text, unspliced_slider, _, spliced_slider = st.columns([2.7, 3.6, 0.1, 3.6])
    with summary_text:
        unspliced_lower_threshold = st.session_state.pancreas_unspliced_threshold[0]
        unspliced_upper_threshold = st.session_state.pancreas_unspliced_threshold[1]
        spliced_lower_threshold = st.session_state.pancreas_spliced_threshold[0]
        spliced_upper_threshold = st.session_state.pancreas_spliced_threshold[1]

        st.text(
            f"cells: {total_obs}, "
            + f"genes: {total_var}\n"
            + f"{unspliced_lower_threshold} ≤ U ≤ {unspliced_upper_threshold}:  {unspliced_var_gt_threshold}\n"
            + f"{spliced_lower_threshold} ≤ S ≤ {spliced_upper_threshold}:  {spliced_var_gt_threshold}\n"
            + f"{unspliced_lower_threshold, spliced_lower_threshold} ≤ U,S ≤ {unspliced_upper_threshold, spliced_upper_threshold}: "
            + f"{len(filtered_df)}"
        )

    with unspliced_slider:
        st.slider(
            "unspliced thresholds",
            0,
            int(max_unspliced),
            key="pancreas_unspliced_threshold",
        )

    with spliced_slider:
        st.slider(
            "spliced thresholds", 0, int(max_spliced), key="pancreas_spliced_threshold"
        )
