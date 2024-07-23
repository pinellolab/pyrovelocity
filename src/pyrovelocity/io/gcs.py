from pathlib import Path
from urllib.parse import urlparse

from beartype import beartype
from google.cloud.storage import Client, transfer_manager
from google.cloud.storage.blob import Blob
from returns.result import Failure, Result, Success

from pyrovelocity.logging import configure_logging

__all__ = [
    "download_blob_from_uri",
    "download_bucket",
    "download_bucket_from_uri",
    "upload_file_concurrently",
    "upload_directory",
]

logger = configure_logging(__name__)


@beartype
def download_blob_from_uri(
    blob_uri: str,
    concurrent: bool = False,
):
    client = Client()
    parsed_blob_uri = urlparse(blob_uri)
    if not parsed_blob_uri.scheme == "gs":
        raise ValueError(
            f"URI scheme must be 'gs', not {parsed_blob_uri.scheme}."
        )
    blob_path = Path(parsed_blob_uri.path)
    blob_filename = blob_path.name
    blob = Blob.from_string(blob_uri, client)
    if concurrent:
        download_blob_concurrently(blob, blob_filename)
    else:
        blob.download_to_filename(f"./{blob_filename}")
    print(f"Downloaded {blob_uri} to {blob_filename}.")


@beartype
def download_blob_concurrently(
    blob: Blob,
    filename: str | Path,
    chunk_size: int = 32 * 1024 * 1024,
    workers: int = 8,
):
    """Download a single file in chunks, concurrently in a process pool."""

    transfer_manager.download_chunks_concurrently(
        blob, filename, chunk_size=chunk_size, max_workers=workers
    )


@beartype
def download_bucket_from_uri(
    bucket_uri: str,
    destination_directory: str | Path = "",
    workers: int = 8,
    max_results: int = 1000,
) -> Result[None, Exception]:
    """
    Download all of the blobs in a bucket, concurrently in a process pool.

    The filename of each blob once downloaded is derived from the blob name and
    the `destination_directory` parameter.

    Directories will be created automatically as needed, for instance to
    accommodate blob names that include slashes.

    Adapted from:
    https://github.com/googleapis/python-storage/blob/v2.14.0/samples/snippets/storage_transfer_manager_download_bucket.py
    """

    try:
        parsed_bucket_uri = urlparse(bucket_uri)
        if not parsed_bucket_uri.scheme == "gs":
            raise ValueError(
                f"URI scheme must be 'gs', not {parsed_bucket_uri.scheme}."
            )
        storage_client = Client()
        bucket = storage_client.bucket(parsed_bucket_uri.netloc)

        blob_names = [
            blob.name
            for blob in bucket.list_blobs(
                max_results=max_results,
                prefix=parsed_bucket_uri.path[1:],
            )
        ]

        results = transfer_manager.download_many_to_path(
            bucket,
            blob_names,
            destination_directory=destination_directory,
            max_workers=workers,
        )

        for name, result in zip(blob_names, results):
            if isinstance(result, Exception):
                logger.error(
                    f"Failed to download {name} due to exception: {result}"
                )
            else:
                logger.info(
                    f"Downloaded {name} to {destination_directory + '/' + name}"
                )
        return Success(None)

    except Exception as e:
        logger.error(
            f"Failed to download files from {bucket_uri} to {destination_directory}."
        )
        return Failure(e)


@beartype
def download_bucket(
    bucket_name: str,
    destination_directory: str | Path = "",
    workers: int = 8,
    max_results: int = 1000,
) -> Result[None, Exception]:
    """
    Download all of the blobs in a bucket, concurrently in a process pool.

    The filename of each blob once downloaded is derived from the blob name and
    the `destination_directory `parameter. For complete control of the filename
    of each blob, use transfer_manager.download_many() instead.

    Directories will be created automatically as needed, for instance to
    accommodate blob names that include slashes.

    Adapted from:
    https://github.com/googleapis/python-storage/blob/v2.14.0/samples/snippets/storage_transfer_manager_download_bucket.py
    """

    try:
        storage_client = Client()
        bucket = storage_client.bucket(bucket_name)

        blob_names = [
            blob.name for blob in bucket.list_blobs(max_results=max_results)
        ]

        results = transfer_manager.download_many_to_path(
            bucket,
            blob_names,
            destination_directory=destination_directory,
            max_workers=workers,
        )

        for name, result in zip(blob_names, results):
            if isinstance(result, Exception):
                logger.error(
                    f"Failed to download {name} due to exception: {result}"
                )
            else:
                logger.info(
                    f"Downloaded {name} to {destination_directory + name}"
                )
        return Success(None)

    except Exception as e:
        logger.error(
            f"Failed to download files from {bucket_name} to {destination_directory}."
        )
        return Failure(e)


@beartype
def upload_file_concurrently(
    bucket_name: str,
    source_filename: str | Path,
    destination_blob_name: str,
    chunk_size: int = 32 * 1024 * 1024,
    workers: int = 8,
) -> Result[str, Exception]:
    """
    Upload a single file, in chunks, concurrently in a process pool.

    Adapted from:
    https://github.com/googleapis/python-storage/blob/v2.14.0/samples/snippets/storage_transfer_manager_upload_chunks_concurrently.py

    Parameters:
        bucket_name (str): The ID of the GCS bucket.
        source_filename (Union[str, Path]): The path to the file to upload.
        destination_blob_name (str): The ID of the GCS object.
        chunk_size (int): The size of each chunk. Defaults to 32 MiB.
        workers (int): The maximum number of processes to use for the operation.

    Returns:
        Result[None, Exception]: A Result object encapsulating success or failure.
    """
    try:
        storage_client = Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)

        transfer_manager.upload_chunks_concurrently(
            source_filename, blob, chunk_size=chunk_size, max_workers=workers
        )

        file_url = f"https://storage.googleapis.com/{bucket_name}/{destination_blob_name}"
        logger.info(f"File {source_filename} uploaded to:\n{file_url}")
        return Success(file_url)
    except Exception as e:
        logger.error(
            f"Failed to upload file {source_filename} due to exception: {e}"
        )
        return Failure(e)


@beartype
def upload_directory(
    bucket_name: str, source_directory: str | Path, workers: int = 8
) -> Result[None, Exception]:
    """
    Upload every file in a directory, including all files in subdirectories.

    Each blob name is derived from the filename, not including the `directory`
    parameter itself. For complete control of the blob name for each file (and
    other aspects of individual blob metadata), use
    transfer_manager.upload_many() instead.

    Adapted from:
    https://github.com/googleapis/python-storage/blob/v2.14.0/samples/snippets/storage_transfer_manager_upload_directory.py
    """

    try:
        storage_client = Client()
        bucket = storage_client.bucket(bucket_name)

        directory_as_path_obj = Path(source_directory)
        paths = directory_as_path_obj.rglob("*")
        logger.info(
            f"Uploading files from {source_directory} to {bucket_name}."
        )

        file_paths = [path for path in paths if path.is_file()]

        relative_paths = [
            path.relative_to(source_directory) for path in file_paths
        ]

        string_paths = [str(path) for path in relative_paths]

        logger.info(f"Found {len(string_paths)} files in {source_directory}.")

        results = transfer_manager.upload_many_from_filenames(
            bucket,
            string_paths,
            source_directory=source_directory,
            max_workers=workers,
        )

        for name, result in zip(string_paths, results):
            if isinstance(result, Exception):
                logger.error(
                    f"Failed to upload {name} due to exception: {result}"
                )
            else:
                logger.info(f"Uploaded {name} to {bucket.name}.")

        return Success(None)

    except Exception as e:
        logger.error(
            f"Failed to upload files from {source_directory} to {bucket_name}."
        )
        return Failure(e)
