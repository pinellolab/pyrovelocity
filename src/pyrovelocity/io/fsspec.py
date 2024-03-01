import os

from beartype import beartype
from fsspec.asyn import AsyncFileSystem
from fsspec.implementations.local import LocalFileSystem
from fsspec.spec import AbstractFileSystem
from gcsfs import GCSFileSystem
from returns.result import Failure
from returns.result import Result
from returns.result import Success

from pyrovelocity.logging import configure_logging


__all__ = [
    "authenticate_and_get_fs",
    "upload_directory",
]

logger = configure_logging(__name__)


@beartype
def authenticate_and_get_fs(
    service_account_json_path: str | None = None,
) -> AsyncFileSystem:
    """
    Authenticate using a service account JSON file or use default credentials if
    available and get an AsyncFileSystem object. Additional fsspec file systems
    can be added here; however, note the intention is to restrict to those that
    implement the AsyncFileSystem interface.

    Args:
        service_account_json_path (str): Path to the service account JSON file. Defaults to None.

    Returns:
        AsyncFileSystem: An instance of AsyncFileSystem (such as GCSFileSystem).
    """
    fs = GCSFileSystem(token=service_account_json_path)
    return fs


@beartype
def upload_directory(
    fs: AsyncFileSystem,
    local_path: str,
    target_path: str,
    check_files: bool = False,
) -> Result[str, Exception]:
    """
    Uploads a directory to an fsspec AsyncFileSystem.
    See https://filesystem-spec.readthedocs.io/en/latest/copying.html#single-source-to-single-target
    "1f. Directory to new directory" for expected behavior.

    Args:
        fs (AsyncFileSystem): An authenticated file system object.
        local_path (str): The local directory path to upload.
        target_path (str): The target path in the storage service.
        check_files (bool): Flag to determine whether to check the files after upload.

    Returns:
        Result[str, Exception]: A result object encapsulating success or failure.

    Examples:
        >>> fs = authenticate_and_get_fs() # xdoctest: +SKIP
        >>> upload_directory(
        ...     fs,
        ...     "local/path/",
        ...     "storage/path/",
        ... ) # xdoctest: +SKIP
    """
    try:
        fs.put(local_path, target_path, recursive=True, auto_mkdir=True)

        if not check_files:
            logger.info(
                f"Attempted to upload '{local_path}' to '{target_path}' without checking files."
            )
            logger.warning(
                " Skipping file check after upload. Set check_files=True to enable."
            )

            return Success(target_path)

        comparison_result = _compare_directory_contents(
            fs, local_path, target_path
        )
        if isinstance(comparison_result, Failure):
            return comparison_result

        logger.info(
            f"Successfully uploaded {local_path} to {target_path} and verified contents."
        )
        return Success(target_path)

    except Exception as e:
        logger.error(f"Failed to upload {local_path} to {target_path}: {e}")
        return Failure(e)


def _compare_directory_contents(
    fs: AbstractFileSystem,
    local_path: str,
    target_path: str,
) -> Result[None, Exception]:
    """
    Compares the contents of a local directory with a remote directory.

    Args:
        fs (AbstractFileSystem): The file system object for the remote directory.
        local_path (str): The path to the local directory.
        target_path (str): The path to the remote directory.

    Returns:
        Result[None, Exception]: A success result if the directories match, otherwise a failure.
    """
    local_fs = LocalFileSystem()

    local_path = os.path.normpath(local_path) + os.sep
    target_path = os.path.normpath(target_path) + os.sep
    logger.info(
        f"Comparing local directory '{local_path}' with remote directory '{target_path}'."
    )

    local_files_dirs = _collect_relative_paths(local_fs, local_path, local_path)

    remote_files_dirs = _collect_relative_paths(fs, target_path, target_path)

    if local_files_dirs != remote_files_dirs:
        mismatched_files = local_files_dirs.symmetric_difference(
            remote_files_dirs
        )
        return Failure(
            Exception(
                f"Mismatch between local and remote directories: {mismatched_files}"
            )
        )

    return Success(None)


def _collect_relative_paths(
    fs: AbstractFileSystem, root_path: str, base_path: str
) -> set:
    """
    Collects relative paths of all files and directories under the given root_path.

    Args:
        fs (AbstractFileSystem): The file system object to use.
        root_path (str): The root directory path to start collecting from.
        base_path (str): The base path to use for calculating relative paths.

    Returns:
        set: A set of relative paths under the root_path.
    """
    files_dirs = set()
    for root, dirs, files in fs.walk(root_path, detail=False):
        for name in dirs + files:
            full_path = os.path.join(root, name)
            rel_path = os.path.relpath(full_path, start=base_path)
            files_dirs.add(rel_path)
    return files_dirs
