import hashlib
from pathlib import Path

import pytest

from pyrovelocity.io.hash import hash_file


def create_file_with_content(path: Path, content: str):
    path.write_text(content)
    return path


def calculate_sha256(content: str):
    return hashlib.sha256(content.encode()).hexdigest()


@pytest.fixture
def sample_files(tmp_path):
    files = {
        "empty": create_file_with_content(tmp_path / "empty.txt", ""),
        "small": create_file_with_content(tmp_path / "small.txt", "test file"),
        "medium": create_file_with_content(
            tmp_path / "medium.txt", "A" * 10000
        ),
        "large": create_file_with_content(
            tmp_path / "large.txt", "B" * 1000000
        ),
    }
    return files


def test_hash_file(sample_files):
    for file_type, file_path in sample_files.items():
        content = file_path.read_text()
        expected_hash = calculate_sha256(content)
        assert (
            hash_file(file_path) == expected_hash
        ), f"Hash mismatch for {file_type} file"


def test_hash_file_nonexistent_file(tmp_path):
    non_existent_file = tmp_path / "nonexistent.txt"
    with pytest.raises(FileNotFoundError):
        hash_file(non_existent_file)


def test_hash_file_directory(tmp_path):
    with pytest.raises(IsADirectoryError):
        hash_file(tmp_path)
