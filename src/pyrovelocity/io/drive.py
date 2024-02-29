import os
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from typing import List

from beartype import beartype
from google.oauth2 import service_account
from googleapiclient.discovery import Resource
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

from pyrovelocity.logging import configure_logging


logger = configure_logging(__name__)


@beartype
def authenticate_gdrive(
    service_account_file: str = "service-account.json",
    scopes: List[str] = ["https://www.googleapis.com/auth/drive"],
) -> Resource:
    """
    Authenticate with Google Drive API using a service account.

    Args:
        service_account_file (str, optional): Path to the service account JSON file. Defaults to "service-account.json".
        scopes (List[str], optional): List of scopes to request during authentication. Defaults to ["https://www.googleapis.com/auth/drive"].

    Returns:
        Resource: An authenticated Google Drive API service object.

    Examples:
        >>> authenticate_gdrive() # xdoctest: +SKIP
    """
    credentials = service_account.Credentials.from_service_account_file(
        service_account_file, scopes=scopes
    )
    service = build("drive", "v3", credentials=credentials)
    return service


def _get_mimetype(file_path):
    """Get the mimetype based on file extension"""
    import mimetypes

    mimetype, _ = mimetypes.guess_type(file_path)
    return mimetype or "application/octet-stream"


@beartype
def upload_file_to_drive(
    service: Resource, file_path: str, folder_id: str
) -> str:
    """
    Uploads an file file to a specified Google Drive folder.

    Args:
        service: An authenticated Google Drive API service object.
        file_path (str): Path to the file to be uploaded.
        folder_id (str): The ID of the Google Drive folder where the file will be uploaded.

    Returns:
        str: The ID of the uploaded file.

    Examples:
        >>> service = authenticate_gdrive("pyrovelocity-drive-sa.json") # xdoctest: +SKIP
        >>> upload_file_to_drive(service, "docs/_static/logo.png", "1-0y0Gw1HIC3o8ooRcHWO-PHuPPtPKYTt") # xdoctest: +SKIP
    """
    file_metadata = {
        "name": os.path.basename(file_path),
        "parents": [folder_id],
    }
    media = MediaFileUpload(file_path, mimetype=_get_mimetype(file_path))
    file = (
        service.files()
        .create(body=file_metadata, media_body=media, fields="id")
        .execute()
    )
    return file.get("id")


@beartype
def create_folder_in_drive(
    service: Resource, folder_name: str, parent_folder_id: str
) -> str:
    """
    Creates a folder in Google Drive.

    Args:
        service: An authenticated Google Drive API service object.
        folder_name (str): Name of the folder to be created.
        parent_folder_id (str): The ID of the parent Google Drive folder.

    Returns:
        str: The ID of the created folder.
    """
    file_metadata = {
        "name": folder_name,
        "mimeType": "application/vnd.google-apps.folder",
        "parents": [parent_folder_id],
    }
    folder = service.files().create(body=file_metadata, fields="id").execute()
    return folder.get("id")


@beartype
def create_drive_folder_structure(
    service: Resource,
    folder_path: str,
    parent_folder_id: str,
    top_folder_name: str,
) -> dict:
    top_folder_id = create_folder_in_drive(
        service, top_folder_name, parent_folder_id
    )
    folder_mapping = {folder_path: top_folder_id}

    for root, dirs, _ in os.walk(folder_path):
        for dir_name in dirs:
            local_dir_path = os.path.join(root, dir_name)
            local_parent_dir = os.path.dirname(local_dir_path)
            drive_parent_id = folder_mapping[local_parent_dir]
            drive_folder_id = create_folder_in_drive(
                service, dir_name, drive_parent_id
            )
            folder_mapping[local_dir_path] = drive_folder_id

    return folder_mapping


@beartype
def upload_files_to_drive(
    service: Resource,
    folder_path: str,
    folder_mapping: dict,
    max_workers: int = 7,
):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {}
        for root, _, files in os.walk(folder_path):
            for file_name in files:
                local_file_path = os.path.join(root, file_name)
                drive_folder_id = folder_mapping[root]
                future = executor.submit(
                    upload_file_to_drive,
                    service,
                    local_file_path,
                    drive_folder_id,
                )
                future_to_file[future] = local_file_path

        for future in as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                file_id = future.result()
                logger.info(f"Uploaded '{file_path}' with File ID: {file_id}")
            except Exception as e:
                logger.error(f"Failed to upload '{file_path}'. Error: {e}")


@beartype
def upload_folder(
    service: Resource,
    folder_path: str,
    parent_folder_id: str,
    top_folder_name: str,
    max_workers: int = 7,
):
    """
    Uploads an entire directory to Google Drive.

    Args:
        service: An authenticated Google Drive API service object.
        folder_path (str): The local directory path to upload.
        parent_folder_id (str): The ID of the parent Google Drive folder.
        top_folder_name (str): The name of the top-level folder in Google Drive.
        max_workers (int, optional): The maximum number of threads to use for uploading files. Defaults to 7.

    Examples:
        >>> service = authenticate_gdrive("pyrovelocity-drive-sa.json") # xdoctest: +SKIP
        >>> upload_folder(service, "docs/_static/", "1-0y0Gw1HIC3o8ooRcHWO-PHuPPtPKYTt", "test_00/") # xdoctest: +SKIP
    """
    folder_mapping = create_drive_folder_structure(
        service, folder_path, parent_folder_id, top_folder_name
    )
    upload_files_to_drive(service, folder_path, folder_mapping, max_workers)
