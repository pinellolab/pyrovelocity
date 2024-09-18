from importlib.resources import files

import pytest

from pyrovelocity.io.serialization import load_anndata_from_json


@pytest.fixture
def adata_preprocessed_3_4():
    fixture_file_path = (
        files("pyrovelocity.tests.data") / "preprocessed_3_4.json"
    )
    return load_anndata_from_json(fixture_file_path)
