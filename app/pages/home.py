from pathlib import Path
from utils.html_factory import CSSStyle, make_div, make_img, st_write_bs4
import streamlit as st

# Title
TITLE_P1 = ""
TITLE_P2 = "Probabilistic model of"
TITLE_P3 = "RNA velocity"

home_markdown_table = """
|         |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| ------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| CI/CD   | [![CI - Test](https://github.com/pinellolab/pyrovelocity/actions/workflows/tests.yml/badge.svg)](https://github.com/pinellolab/pyrovelocity/actions/workflows/tests.yml) [![CML](https://github.com/pinellolab/pyrovelocity/actions/workflows/cml.yml/badge.svg)](https://github.com/pinellolab/pyrovelocity/actions/workflows/cml.yml) [![pre-commit.ci status](https://results.pre-commit.ci/badge/github/pinellolab/pyrovelocity/master.svg)](https://results.pre-commit.ci/latest/github/pinellolab/pyrovelocity/master) |
| Docs    | [![Documentation Status](https://readthedocs.org/projects/pyrovelocity/badge/?version=latest)](https://pyrovelocity.readthedocs.io/en/latest/?badge=latest)                                                                                                                                                                                                                                                                                                                                                                  |
| Package | [![PyPI - Version](https://img.shields.io/pypi/v/pyrovelocity.svg?logo=pypi&label=PyPI&logoColor=gold)](https://pypi.org/project/pyrovelocity/) [![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pyrovelocity.svg?logo=python&label=Python&logoColor=gold)](https://pypi.org/project/pyrovelocity/)                                                                                                                                                                                                          |
| Meta    | [![code style - Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![License - MIT](https://img.shields.io/badge/license-AGPL%203-purple)](https://spdx.org/licenses/)                                                                                                                                                                                                                                                                                                       |
"""

BIG_TITLE_STYLE = CSSStyle(
    text_align="center",
    font_size="58px",
    line_height="64px",
    margin="40px 80px 60px 80px",
    font_weight=700,
)
BIG_TITLE_DIV = make_div(style=BIG_TITLE_STYLE)
BIG_TITLE_DIV.extend(
    [
        make_div(text=TITLE_P1),
        make_div(text=TITLE_P2),
        make_div(text=TITLE_P3),
    ]
)

AUTHOR_STYLE = CSSStyle(
    text_align="center",
    font_size="48px",
    line_height="48px",
    font_style="italic",
    margin_bottom="40px",
)
AUTHOR_DIV = make_div(style=AUTHOR_STYLE, text="Pinello lab")


BIG_LOGO_STYLE = CSSStyle(
    text_align="center",
    width="30%",
    margin_left="auto",
    margin_right="auto",
    display="block",
)

BIG_LOGO_IMG = make_img(
    style=BIG_LOGO_STYLE,
    src=Path("./docs/_static/logo.png"),
)

# main
def st_show():
    st_write_bs4(BIG_LOGO_IMG)
    st_write_bs4(BIG_TITLE_DIV)
    st_write_bs4(AUTHOR_DIV)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(' ')
    with col2:
        st.markdown(home_markdown_table, unsafe_allow_html=True)
    with col3:
        st.write(' ')
    