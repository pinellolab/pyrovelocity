import os
import tarfile
from pathlib import Path

from beartype import beartype
from fsspec import AbstractFileSystem
from fsspec.implementations.local import LocalFileSystem
from returns.result import Failure
from returns.result import Result
from returns.result import Success

from pyrovelocity.logging import configure_logging


__all__ = ["create_tarball_from_filtered_dir"]

logger = configure_logging(__name__)


@beartype
def create_tarball_from_filtered_dir(
    src_dir: Path | str,
    output_filename: Path | str,
    extensions: tuple[str, ...] = (".png", ".pdf"),
) -> Result[None, Exception]:
    """
    Creates a gzipped tarball of files with specific extensions under the given
    directory using fsspec.

    Args:
        src_dir: The top-level directory path to search for files.
        output_filename: The path to the output gzipped tarball file.
        extensions: A tuple of file extensions to include in the tarball. Defaults to ('.png', '.pdf').

    Returns:
        Result[None, Exception]: A result object encapsulating success or failure.

    Examples:
        >>> from pathlib import Path
        >>> tmp = getfixture('tmp_path')  # Get a temporary directory path
        >>> (tmp / 'document.pdf').write_text('PDF content')  # Create a PDF file
        >>> (tmp / 'image.png').write_text('PNG content')  # Create a PNG file
        >>> (tmp / 'exclude.txt').write_text('Should be excluded')  # Create a TXT file
        >>> output_tarball = tmp / 'output.tar.gz'  # Define the output tarball path
        >>> result = create_tarball_from_filtered_dir(tmp, output_tarball)  # Create tarball
        >>> print(f"Result: {result}") # Check the result
        Result: <Success: None>
        >>> import tarfile
        >>> with tarfile.open(output_tarball, 'r:gz') as tar:  # Open the created tarball
        ...     filenames = {member.name for member in tar.getmembers()}  # List contents
        >>> assert 'document.pdf' in filenames  # PDF should be included
        >>> assert 'image.png' in filenames  # PNG should be included
        >>> assert 'exclude.txt' not in filenames  # TXT should be excluded
    """
    try:
        logger.info(
            f"\nCreating gzipped tarball from {src_dir} for files with extensions:\n"
            f"\t{extensions}\n"
        )
        fs: AbstractFileSystem = LocalFileSystem()
        with tarfile.open(output_filename, "w:gz") as tar:
            for root, dirs, files in fs.walk(str(src_dir)):
                for file in files:
                    if file.endswith(extensions):
                        full_path = os.path.join(root, file)

                        # path is relative to the src_dir in the archive
                        relative_path = os.path.relpath(
                            full_path, start=str(src_dir)
                        )

                        with fs.open(full_path, "rb") as file_obj:
                            tarinfo = tarfile.TarInfo(name=relative_path)
                            tarinfo.size = file_obj.size
                            tar.addfile(tarinfo=tarinfo, fileobj=file_obj)
        return Success(None)

    except Exception as e:
        logger.error(
            f"Failed to create gzipped tarball from {src_dir} due to exception: {e}"
        )
        return Failure(e)
