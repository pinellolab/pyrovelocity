import os
import tarfile
from pathlib import Path

from beartype import beartype
from beartype.typing import List, Union
from fsspec import AbstractFileSystem, filesystem
from fsspec.implementations.local import LocalFileSystem
from returns.result import Failure, Result, Success

from pyrovelocity.logging import configure_logging

__all__ = [
    "create_tarball_from_filtered_dir",
    "copy_files_to_directory",
]

logger = configure_logging(__name__)


@beartype
def create_tarball_from_filtered_dir(
    src_dir: Path | str,
    output_filename: Path | str,
    extensions: tuple[str, ...] = (".png", ".pdf", ".csv"),
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


@beartype
def copy_files_to_directory(
    files_to_copy: List[str | Path],
    target_directory: str | Path,
) -> Result[None, Exception]:
    """
    Copy a list of files to a target directory using fsspec.

    Args:
        files_to_copy: A list of file paths (str or Path) to copy.
        target_directory: The target directory path (str or Path) to copy files into.

    Returns:
        Result[None, Exception]: A result object encapsulating success or failure.

    Examples:
        >>> from pathlib import Path
        >>> tmp = getfixture('tmp_path')  # Get a temporary directory path
        >>> source_dir = tmp / 'source'
        >>> source_dir.mkdir()  # Create a source directory
        >>> (source_dir / 'file1.txt').write_text('Content 1')  # Create a text file
        >>> (source_dir / 'file2.csv').write_text('Content 2')  # Create a CSV file
        >>> target_dir = tmp / 'target'
        >>> files_to_copy = [source_dir / 'file1.txt', source_dir / 'file2.csv']
        >>> result = copy_files_to_directory(files_to_copy, target_dir)  # Copy files
        >>> print(f"Result: {result}")  # Check the result
        Result: <Success: None>
        >>> assert (target_dir / 'file1.txt').exists()  # Check if file1.txt was copied
        >>> assert (target_dir / 'file2.csv').exists()  # Check if file2.csv was copied
        >>> with open(target_dir / 'file1.txt', 'r') as f:
        ...     content = f.read()
        ...     assert content == 'Content 1'  # Check if content was correctly copied
        >>> with open(target_dir / 'file2.csv', 'r') as f:
        ...     content = f.read()
        ...     assert content == 'Content 2'  # Check if content was correctly copied
    """
    fs: LocalFileSystem = filesystem("file")
    target_directory = str(target_directory)

    try:
        fs.makedirs(target_directory, exist_ok=True)

        for src_path in files_to_copy:
            src_path = str(src_path)
            filename = Path(src_path).name
            dst_path = str(Path(target_directory) / filename)

            try:
                with fs.open(src_path, "rb") as src_file, fs.open(
                    dst_path, "wb"
                ) as dst_file:
                    dst_file.write(src_file.read())
                print(f"Successfully copied {src_path} to {dst_path}")
            except Exception as e:
                print(f"Error copying {src_path} to {dst_path}: {e}")
                return Failure(e)

        return Success(None)

    except Exception as e:
        print(f"Unexpected error during file copying: {e}")
        return Failure(e)
