import altair as alt
import scvelo as scv
import streamlit as st
from google.cloud import storage


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

# https://bit.ly/42WbSYD
# http://storage.googleapis.com/BUCKET_NAME/OBJECT_NAME
@st.cache_data(show_spinner=False)
def load_gcs_data(bucket_name, source_path, destination_file_name):
    client = storage.Client.create_anonymous_client()
    bucket = client.bucket(bucket_name=bucket_name)
    blob = bucket.blob(source_path)
    blob.download_to_filename(destination_file_name)

    print(
        f"\nDownloaded blob {source_path}\n"
        f"from bucket {bucket.name}\n"
        f"to {destination_file_name}.\n"
    )


data_set_name = "pancreas"
model_information = dict(model_name="fixed time model", model_id="model1")
data_file_name = f"{data_set_name}_{model_information['model_id']}.h5ad"

with st.spinner(
    f"loading {data_set_name} data for {model_information['model_name']}..."
):
    load_gcs_data(
        "pyrovelocity",
        f"data/{data_set_name}/{model_information['model_id']}/trained.h5ad",
        data_file_name,
    )
