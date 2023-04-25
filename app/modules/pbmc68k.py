import streamlit_book as stb


def st_show():
    stb.set_book_config(
        menu_title="",
        menu_icon="",
        options=[
            "data review",
        ],
        paths=[
            "app/modules/pbmc68k/datareview.py",
        ],
        icons=[""],
        save_answers=False,
        orientation="vertical",
        styles={"nav-link": {"font-family": "LMSans10"}},
    )
