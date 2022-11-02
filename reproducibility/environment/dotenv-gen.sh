#!/usr/bin/env sh

#########
#
# source this file to set environment variables:
#
#     source dotenv-gen.sh
# 
# - depends on [pass](https://www.passwordstore.org/)
# 
# - replace values with hardcoded values if you aren't able to use pass
#
# - .env included to set variables from Makefile 
# 
# note: default path for gcp_credentials_file is
#
#    ~/.config/gcloud/application_default_credentials.json
#
########## 

# set -xv


if ! command -v pass
then
    echo "pass could not be found"
    echo "please see https://www.passwordstore.org/ for installation instructions"
    echo "or set all variables using pass manually and remove or comment this check"
    exit 1
fi

TF_VAR_project="$(pass gcp_project)"
TF_VAR_email="$(pass gcp_email)"
TF_VAR_credentials_file="$(pass gcp_credentials_file)"
TF_VAR_notebooks_name="$(pass gcp_notebooks_name)"
GITHUB_USERNAME=$(pass github_username)
GITHUB_ORG_NAME=$(pass github_org)
GITHUB_REPO_NAME=$(pass github_repo)
GITHUB_BRANCH_NAME=$(pass github_branch)
GITHUB_REPO_CONDA_ENV_PATH_NAME=$(pass github_repo_conda_env_path)
export TF_VAR_project TF_VAR_email TF_VAR_credentials_file TF_VAR_notebooks_name GITHUB_ORG_NAME GITHUB_REPO_NAME GITHUB_BRANCH_NAME GITHUB_REPO_CONDA_ENV_PATH_NAME


# hardcode values in the .env file if you don't want to use pass
{
    echo "TF_VAR_project=$TF_VAR_project";
    echo "TF_VAR_email=$TF_VAR_email";
    echo "TF_VAR_credentials_file=$TF_VAR_credentials_file";
    echo "TF_VAR_notebooks_name=$TF_VAR_notebooks_name";
    echo "GITHUB_USERNAME=$GITHUB_USERNAME";
    echo "GITHUB_ORG_NAME=$GITHUB_ORG_NAME";
    echo "GITHUB_REPO_NAME=$GITHUB_REPO_NAME";
    echo "GITHUB_BRANCH_NAME=$GITHUB_BRANCH_NAME";
    echo "GITHUB_REPO_CONDA_ENV_PATH_NAME=$GITHUB_REPO_CONDA_ENV_PATH_NAME";
} > .env