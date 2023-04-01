import streamlit as st
import scvelo as scv

# setup
st.set_page_config(
        page_title="pyrovelocity report",
        page_icon="https://raw.githubusercontent.com/pinellolab/pyrovelocity/master/docs/_static/logo.png",
        layout="centered",
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
            div.block-container{padding-top:0rem;}
            </style>
            """
st.markdown(streamlit_style, unsafe_allow_html=True) 

st.title("pyrovelocity report")
