import pandas as pd
import streamlit as st


class DataReview:
    def __init__(
        self,
        dataset: str,
        fraction_max_counts: float,
        df: pd.DataFrame,
        total_obs: int,
        total_var: int,
        max_spliced: int,
        max_unspliced: int,
    ):
        self.dataset = dataset
        self.fraction_max_counts = fraction_max_counts
        self.df = df
        self.total_obs = total_obs
        self.total_var = total_var
        self.max_spliced = max_spliced
        self.max_unspliced = max_unspliced

    @st.cache_data(
        show_spinner="filtering dataframe by count thresholds", persist=True
    )
    def filter_count_thresholds_from_dataset_df(
        _self, df, spliced_threshold, unspliced_threshold
    ):
        from utils.data import filter_var_counts_by_thresholds

        return filter_var_counts_by_thresholds(
            df, spliced_threshold, unspliced_threshold
        )

    @st.cache_data(show_spinner="generating histogram", persist=True)
    def generate_dataset_histogram(
        _self, df, title, selected_var, selected_obs
    ):
        from utils.data import interactive_spliced_unspliced_histogram

        chart = interactive_spliced_unspliced_histogram(
            df, title, selected_var, selected_obs
        )
        chart.height = 738
        return chart

    @st.cache_data(
        show_spinner="filtering dataframe by selected cells/genes", persist=True
    )
    def filter_obs_vars_from_dataset_df(_self, df, selected_var, selected_obs):
        if selected_var and selected_obs:
            return df[
                df["var_name"].isin(selected_var)
                & df["obs_name"].isin(selected_obs)
            ]
        elif selected_var:
            return df[df["var_name"].isin(selected_var)]
        elif selected_obs:
            return df[df["obs_name"].isin(selected_obs)]
        else:
            return df

    def next_page(self, last_page):
        if st.session_state[f"{self.dataset}_page"] + 1 > last_page:
            st.session_state[f"{self.dataset}_page"] = 1
        else:
            st.session_state[f"{self.dataset}_page"] += 1

    def prev_page(self, last_page):
        if st.session_state[f"{self.dataset}_page"] <= 1:
            st.session_state[f"{self.dataset}_page"] = last_page
        else:
            st.session_state[f"{self.dataset}_page"] -= 1

    def generate_page(self):
        if f"{self.dataset}_spliced_threshold" not in st.session_state:
            st.session_state[f"{self.dataset}_spliced_threshold"] = (
                2,
                int(self.max_spliced * self.fraction_max_counts),
            )

        if f"{self.dataset}_unspliced_threshold" not in st.session_state:
            st.session_state[f"{self.dataset}_unspliced_threshold"] = (
                2,
                int(self.max_unspliced * self.fraction_max_counts),
            )

        (
            df_thresholded,
            spliced_var_gt_threshold,
            unspliced_var_gt_threshold,
        ) = self.filter_count_thresholds_from_dataset_df(
            self.df,
            st.session_state[f"{self.dataset}_spliced_threshold"],
            st.session_state[f"{self.dataset}_unspliced_threshold"],
        )

        obs_values = sorted(self.df["obs_name"].unique())
        var_values = sorted(self.df["var_name"].unique())

        col_1, _, col_3 = st.columns([6.25, 0.25, 3.5])

        with col_1:
            col11, col12 = st.columns([1, 1])

            with col11:
                selected_obs = st.multiselect("Select cell(s)", obs_values)

            with col12:
                selected_var = st.multiselect("Select gene(s)", var_values)

            filtered_df = self.filter_obs_vars_from_dataset_df(
                df_thresholded, selected_var, selected_obs
            )

        page_size = 20
        last_page = max(1, len(filtered_df) // page_size)

        with col_3:
            if st.checkbox("paginate", value=True):
                prev, next, current, first, last, _ = st.columns(
                    [1, 1, 2.2, 1, 1, 3.8]
                )

                if next.button("⏵"):
                    self.next_page(last_page)

                if prev.button("⏴"):
                    self.prev_page(last_page)

                if first.button("⏮"):
                    st.session_state[f"{self.dataset}_page"] = 1

                if last.button("⏭"):
                    st.session_state[f"{self.dataset}_page"] = last_page

                with current:
                    with st.spinner("loading page number"):
                        st.selectbox(
                            "page",
                            range(1, last_page + 1),
                            key=f"{self.dataset}_page",
                            label_visibility="collapsed",
                        )

                page_start_index = (
                    st.session_state[f"{self.dataset}_page"] - 1
                ) * page_size
                page_end_index = page_start_index + page_size

                with st.spinner("computing display dataframe"):
                    display_df = filtered_df.iloc[
                        page_start_index:page_end_index
                    ]
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
            spliced_unspliced_histogram_chart = self.generate_dataset_histogram(
                filtered_df,
                "Spliced vs unspliced counts",
                selected_var,
                selected_obs,
            )

            with st.spinner("loading histogram"):
                st.altair_chart(
                    spliced_unspliced_histogram_chart,
                    use_container_width=True,
                    theme="streamlit",
                )

            summary_text, unspliced_slider, _, spliced_slider = st.columns(
                [2.7, 3.6, 0.1, 3.6]
            )
            with summary_text:
                unspliced_lower_threshold = st.session_state[
                    f"{self.dataset}_unspliced_threshold"
                ][0]
                unspliced_upper_threshold = st.session_state[
                    f"{self.dataset}_unspliced_threshold"
                ][1]
                spliced_lower_threshold = st.session_state[
                    f"{self.dataset}_spliced_threshold"
                ][0]
                spliced_upper_threshold = st.session_state[
                    f"{self.dataset}_spliced_threshold"
                ][1]

                st.text(
                    f"cells: {self.total_obs}, "
                    + f"genes: {self.total_var}\n"
                    + f"{unspliced_lower_threshold} ≤ U ≤ {unspliced_upper_threshold}:  {unspliced_var_gt_threshold}\n"
                    + f"{spliced_lower_threshold} ≤ S ≤ {spliced_upper_threshold}:  {spliced_var_gt_threshold}\n"
                    + f"{unspliced_lower_threshold, spliced_lower_threshold} ≤ U,S ≤ {unspliced_upper_threshold, spliced_upper_threshold}: "
                    + f"{len(filtered_df)}"
                )

            with unspliced_slider:
                st.slider(
                    "unspliced thresholds",
                    0,
                    int(self.max_unspliced),
                    key=f"{self.dataset}_unspliced_threshold",
                )

            with spliced_slider:
                st.slider(
                    "spliced thresholds",
                    0,
                    int(self.max_spliced),
                    key=f"{self.dataset}_spliced_threshold",
                )
