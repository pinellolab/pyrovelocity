import hashlib
from pathlib import Path

from beartype import beartype


@beartype
def hash_file(
    file_path: str | Path,
    chunk_size: int = 8192,
):
    sha256_hash = hashlib.sha256()
    file_path = Path(file_path)

    with file_path.open("rb") as f:
        for byte_block in iter(lambda: f.read(chunk_size), b""):
            sha256_hash.update(byte_block)

    return sha256_hash.hexdigest()
