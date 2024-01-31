"""Tests for `pyrovelocity.data` module."""
from pathlib import Path

import pytest
import requests_mock

from pyrovelocity.data import (
    download_dataset,
    load_anndata_from_path,
    subset,
    validate_url_and_file,
)


def test_load_data_module():
    from pyrovelocity import data

    print(data.__file__)


@pytest.fixture
def temp_data_dir(tmp_path):
    data_dir = tmp_path / "data/external"
    data_dir.mkdir(parents=True, exist_ok=True)
    return str(data_dir)


def test_successful_download(temp_data_dir):
    """Test if dataset is successfully downloaded with valid parameters."""
    dataset_name = "pancreas"
    data_path = download_dataset(
        data_set_name=dataset_name, data_external_path=temp_data_dir
    )
    assert Path(data_path).exists()


def test_invalid_url(temp_data_dir):
    """Test handling of invalid URL."""
    with pytest.raises(ValueError):
        download_dataset(
            data_set_name="invalid_url_dataset",
            data_external_path=temp_data_dir,
            data_url="http://invalidurl.com",
        )


def test_invalid_dataset_name(temp_data_dir):
    """Test handling of invalid dataset name."""
    with pytest.raises(AttributeError):
        download_dataset(
            data_set_name="non_existent_dataset",
            data_external_path=temp_data_dir,
        )


def test_invalid_source(temp_data_dir):
    """Test handling of invalid source."""
    with pytest.raises(ValueError):
        download_dataset(
            data_set_name="pancreas",
            data_external_path=temp_data_dir,
            source="invalid_source",
        )


valid_h5ad_url = "https://example.com/valid_file.h5ad"
small_h5ad_url = "https://example.com/small_file.h5ad"
non_h5ad_url = "https://example.com/invalid_file.txt"
invalid_url = "http//invalid_url"


@pytest.fixture
def mock_requests():
    with requests_mock.Mocker() as m:
        m.head(
            valid_h5ad_url,
            headers={
                "Content-Disposition": 'attachment; filename="valid_file.h5ad"',
                "Content-Length": "2000000",  # 2 MB
            },
        )

        m.head(
            small_h5ad_url,
            headers={
                "Content-Disposition": 'attachment; filename="small_file.h5ad"',
                "Content-Length": "500000",  # 0.5 MB
            },
        )

        m.head(
            non_h5ad_url,
            headers={
                "Content-Disposition": 'attachment; filename="invalid_file.txt"',
                "Content-Length": "2000000",
            },
        )

        yield m


def test_validate_valid_url(mock_requests):
    """Test validation of a valid URL leading to a .h5ad file."""
    is_valid, message = validate_url_and_file(valid_h5ad_url)
    assert is_valid
    assert "URL validated and file is an .h5ad file" in message


def test_validate_small_file_url(mock_requests):
    """Test validation of a URL leading to a .h5ad file but smaller than required size."""
    is_valid, message = validate_url_and_file(small_h5ad_url)
    assert not is_valid
    assert "The file size is less than or equal to 1MB" in message


def test_validate_non_h5ad_url(mock_requests):
    """Test validation of a URL leading to a non-.h5ad file."""
    is_valid, message = validate_url_and_file(non_h5ad_url)
    assert not is_valid
    assert "The file does not have an .h5ad extension" in message


def test_validate_invalid_url_format():
    """Test validation of an invalid URL format."""
    is_valid, message = validate_url_and_file(invalid_url)
    assert not is_valid
    assert "Invalid URL format" in message


def test_subset_from_adata(default_sample_data):
    """Test deriving a data subset from an AnnData object."""
    n_obs = 50
    n_vars = 10
    subset_adata, _ = subset(
        adata=default_sample_data, n_obs=n_obs, n_vars=n_vars
    )
    assert subset_adata.n_obs == n_obs
    assert subset_adata.n_vars == n_vars


def test_subset_from_file(default_sample_data_file):
    """Test deriving a data subset from a file."""
    n_obs = 50
    subset_adata, _ = subset(file_path=default_sample_data_file, n_obs=n_obs)
    assert subset_adata.n_obs == n_obs


def test_invalid_input():
    """Test handling of invalid inputs."""
    with pytest.raises(ValueError):
        subset()  # No AnnData object or file path provided


def test_save_subset(default_sample_data, tmp_path):
    """Test saving the subset to a file."""
    n_obs = 50
    output_file = tmp_path / "subset_adata.h5ad"
    _, output_path = subset(
        adata=default_sample_data,
        n_obs=n_obs,
        save_subset=True,
        output_path=output_file,
    )
    assert output_path == output_file
    assert Path(output_file).exists()


def test_subset_specific_n_obs_vars(default_sample_data):
    """Test deriving a data subset with specific n_obs and n_vars."""
    n_obs = 50
    n_vars = 10
    subset_adata, _ = subset(
        adata=default_sample_data, n_obs=n_obs, n_vars=n_vars
    )
    assert subset_adata.n_obs == n_obs
    assert subset_adata.n_vars == n_vars


def test_load_valid_h5ad_file(default_sample_data_file):
    """Test loading from a valid .h5ad file."""
    adata = load_anndata_from_path(default_sample_data_file)
    assert adata is not None


def test_invalid_file_extension(tmp_path):
    """Test loading from a file with an invalid extension."""
    invalid_file = tmp_path / "invalid_file.txt"
    invalid_file.touch()
    with pytest.raises(ValueError):
        load_anndata_from_path(invalid_file)


def test_nonexistent_file():
    """Test loading from a non-existent file."""
    nonexistent_file = Path("nonexistent_file.h5ad")
    with pytest.raises(ValueError):
        load_anndata_from_path(nonexistent_file)
