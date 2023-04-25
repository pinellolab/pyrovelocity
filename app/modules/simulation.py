import streamlit_book as stb


def st_show():
    stb.set_book_config(
        menu_title="",
        menu_icon="",
        options=[
            "data review",
            # "velocity estimation",
        ],
        paths=[
            "app/modules/simulation/datareview.py",
            # "app/modules/simulation/velocityestimation.py",
        ],
        icons=[""],
        save_answers=False,
        orientation="vertical",
        styles={"nav-link": {"font-family": "LMSans10"}},
    )
