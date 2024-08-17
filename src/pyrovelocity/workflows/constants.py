import os

from dotenv import load_dotenv
from dulwich.repo import NotGitRepository, Repo

from pyrovelocity.logging import configure_logging
from pyrovelocity.utils import str_to_bool

logger = configure_logging("pyrovelocity.workflows.constants")

load_dotenv()

PYROVELOCITY_TESTING_FLAG = str_to_bool(
    os.getenv("PYROVELOCITY_TESTING_FLAG", "False")
)
PYROVELOCITY_DATA_SUBSET = str_to_bool(
    os.getenv("PYROVELOCITY_DATA_SUBSET", "False")
)
PYROVELOCITY_OVERWRITE_CACHE = str_to_bool(
    os.getenv("PYROVELOCITY_OVERWRITE_CACHE", "False")
)
PYROVELOCITY_CACHE_FLAG = str_to_bool(
    os.getenv("PYROVELOCITY_CACHE_FLAG", "True")
)
PYROVELOCITY_UPLOAD_RESULTS = str_to_bool(
    os.getenv("PYROVELOCITY_UPLOAD_RESULTS", "True")
)

logger.info(
    f"\nPYROVELOCITY_TESTING_FLAG: {PYROVELOCITY_TESTING_FLAG}\n"
    f"PYROVELOCITY_DATA_SUBSET: {PYROVELOCITY_DATA_SUBSET}\n"
    f"PYROVELOCITY_OVERWRITE_CACHE: {PYROVELOCITY_OVERWRITE_CACHE}\n"
    f"PYROVELOCITY_CACHE_FLAG: {PYROVELOCITY_CACHE_FLAG}\n"
    f"PYROVELOCITY_UPLOAD_RESULTS: {PYROVELOCITY_UPLOAD_RESULTS}\n\n"
)


def get_git_repo_root(path="."):
    try:
        repo = Repo.discover(start=path)
        git_root = repo.path
        return os.path.normpath(git_root)
    except NotGitRepository:
        git_repo_not_found = "Not inside a Git repository. Returning '.'"
        logger.warning(git_repo_not_found)
        return os.path.normpath(path)
        # raise OSError(git_repo_not_found)


repo_root = get_git_repo_root()

if repo_root:
    REMOTE_CLUSTER_CONFIG_FILE_PATH = os.path.join(
        repo_root, ".flyte", "config.yaml"
    )
    LOCAL_CLUSTER_CONFIG_FILE_PATH = os.path.join(
        repo_root, ".flyte", "config-local.yaml"
    )

    if not os.path.isfile(REMOTE_CLUSTER_CONFIG_FILE_PATH):
        remote_cluster_config_file_not_found_message = (
            f"Remote cluster config file not found at path:\n\n"
            f"{REMOTE_CLUSTER_CONFIG_FILE_PATH}\n\n"
            "Verify you have run `make update_config` in the root of the repository,\n"
            "or manually create the file at the path above.\n\n"
        )
        logger.warning(remote_cluster_config_file_not_found_message)
        REMOTE_CLUSTER_CONFIG_FILE_PATH = LOCAL_CLUSTER_CONFIG_FILE_PATH

    if not os.path.isfile(LOCAL_CLUSTER_CONFIG_FILE_PATH):
        local_cluster_config_file_not_found_message = (
            f"Local cluster config file not found at path:\n\n"
            f"{LOCAL_CLUSTER_CONFIG_FILE_PATH}\n\n"
            f"Check that you have not deleted this file from the repository.\n\n"
        )
        logger.warning(local_cluster_config_file_not_found_message)

    logger.debug(
        f"Remote cluster config file path: {REMOTE_CLUSTER_CONFIG_FILE_PATH}"
    )
    logger.debug(
        f"Local cluster config file path: {LOCAL_CLUSTER_CONFIG_FILE_PATH}"
    )
else:
    git_repo_not_found = "Not inside a Git repository or Git is not installed."
    raise OSError(git_repo_not_found)

if __name__ == "__main__":
    from pprint import pprint

    pprint(REMOTE_CLUSTER_CONFIG_FILE_PATH)
    pprint(LOCAL_CLUSTER_CONFIG_FILE_PATH)

# import subprocess
# def get_git_repo_root(path="."):
#     try:
#         git_root = subprocess.check_output(
#             ["git", "rev-parse", "--show-toplevel"], cwd=path
#         )
#         return git_root.decode("utf-8").strip()
#     except subprocess.CalledProcessError:
#         git_repo_not_found = (
#             "Not inside a Git repository or Git is not installed."
#         )
#         raise OSError(git_repo_not_found)
