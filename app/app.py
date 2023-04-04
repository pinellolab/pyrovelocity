from pathlib import Path

import altair as alt
import scvelo as scv
import streamlit as st
from modules import home
from modules import simulation
from utils.altair_theme import custom_theme
from utils.html_factory import CSSStyle
from utils.html_factory import make_div
from utils.html_factory import make_img
from utils.html_factory import st_write_bs4

from pyrovelocity.config import initialize_hydra_config


# setup
st.set_page_config(
    page_title="pyrovelocity report",
    page_icon="https://raw.githubusercontent.com/pinellolab/pyrovelocity/master/docs/_static/logo.png",
    layout="wide",
    initial_sidebar_state="auto",
)

# #MainMenu {visibility: hidden;}
# header {visibility: hidden;}
streamlit_style = """
            <style>
            @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@100&display=swap');
            @import url('https://fonts.googleapis.com/css2?family=Open+Sans:wght@300&display=swap');
            @import url('https://fonts.cdnfonts.com/css/latin-modern-sans');

			html, body, [class*="css"]  {
			font-family: 'LMSans10', 'Open Sans', 'Roboto', sans-serif;
			}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            #MainMenu {visibility: visible;}
            div.block-container{padding-top:2rem;}
            </style>
            """
st.markdown(streamlit_style, unsafe_allow_html=True)

# initialize hydra configuration
cfg = initialize_hydra_config()

# register and use the custom altair theme
alt.themes.register("custom_theme", custom_theme)
alt.themes.enable("custom_theme")

sidebar_header_logo = make_img(
    src=Path("./docs/_static/logo.png"),
    style=CSSStyle(width="128px", flex=1, margin_right="20px"),
)
sidebar_header_title = make_div(style=CSSStyle(text_align="left", flex=3))
sidebar_header_title.extend(
    [
        make_div(
            style=CSSStyle(font_size="18px", font_weight="bold"),
            text="pyrovelocity",
        ),
        make_div(
            style=CSSStyle(font_size="14px"),
            text="Probabilistic modeling",
        ),
        make_div(
            style=CSSStyle(font_size="14px", font_style="italic"),
            text="of RNA velocity",
        ),
    ]
)
sidebar_header = make_div(
    style=CSSStyle(margin_top="-80px", margin_bottom="20px", display="flex")
)
sidebar_header.extend([sidebar_header_logo, sidebar_header_title])

with st.sidebar:
    st_write_bs4(sidebar_header)

# Main Chapter selection
no_data = "🪹 home"
sidebar_label = "NAVIGATE"

selected_module = st.sidebar.selectbox(
    label=sidebar_label,
    options=[
        no_data,
        "🖥️ simulation",
        "⚪️ pancreas",
        "🟠 pons",
        "🔴 peripheral blood",
        "🔵 hematopoietic stem cells",
    ],
)

if selected_module == no_data:
    home.st_show()
elif selected_module == "🖥️ simulation":
    simulation.st_show()
else:
    st.error("Page does not yet exist.")
