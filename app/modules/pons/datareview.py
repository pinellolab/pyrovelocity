import streamlit as st
from pages.datareview import DataReview
from utils.config import get_app_config


PATH_PREFIX = "reproducibility/figures/"
cfg = get_app_config()


@st.cache_data(show_spinner="loading dataframe", persist=True)
def load_pons_df(PATH_PREFIX=PATH_PREFIX, cfg=cfg):
    from utils.data import load_compressed_pickle

    return load_compressed_pickle(
        PATH_PREFIX + cfg.reports.model_summary.pons_model2.dataframe_path
    )


(
    df,
    total_obs,
    total_var,
    max_spliced,
    max_unspliced,
) = load_pons_df()


DataReview(
    "pons", 0.5, df, total_obs, total_var, max_spliced, max_unspliced
).generate_page()
