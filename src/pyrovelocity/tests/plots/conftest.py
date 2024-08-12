import pytest

from pyrovelocity.io.serialization import load_anndata_from_json


@pytest.fixture
def adata_preprocessed():
    return load_anndata_from_json(
        "src/pyrovelocity/tests/data/preprocessed_3_4.json"
    )
