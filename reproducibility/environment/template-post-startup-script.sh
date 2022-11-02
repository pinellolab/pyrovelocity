#!/usr/bin/env sh

##########################################
#
# setup remote development environment:
#
# install conda environment 
# for development 
# 
##########################################

set -x

GITHUB_ORG=${GITHUB_ORG_NAME}
GITHUB_REPO=${GITHUB_REPO_NAME}
GITHUB_BRANCH=${GITHUB_BRANCH_NAME}
GIHUB_REPO_CONDA_ENV_YML_PATH=${GITHUB_REPO_CONDA_ENV_PATH_NAME}
CONDA_PATH=/opt/conda/bin
JUPYTER_USER=jupyter
REPO_PATH=/home/$JUPYTER_USER/$GITHUB_REPO

$CONDA_PATH/conda init --all --system 
sudo -u $JUPYTER_USER $CONDA_PATH/conda init bash

$CONDA_PATH/conda install -n base -c conda-forge -y mamba 
$CONDA_PATH/conda config --add channels bioconda
$CONDA_PATH/conda config --add channels conda-forge
$CONDA_PATH/conda config --set channel_priority flexible
$CONDA_PATH/mamba update --all -n base -y
$CONDA_PATH/mamba install -n base -c conda-forge \
    gh \
    pipx \
    htop \
    bat \
    fzf \
    ripgrep \
    gpustat \
    expect \
    conda-build \
    google-cloud-sdk \
    jupyterlab-nvdashboard \
    jupyterlab_execute_time

sudo git clone --branch $GITHUB_BRANCH https://github.com/$GITHUB_ORG/$GITHUB_REPO $REPO_PATH
sudo chown -R $JUPYTER_USER:$JUPYTER_USER /home/$JUPYTER_USER 
sudo chmod -R 755 /home/$JUPYTER_USER

$CONDA_PATH/mamba env create -n $GITHUB_REPO \
    -f $REPO_PATH/$GIHUB_REPO_CONDA_ENV_YML_PATH
sudo -u $JUPYTER_USER $CONDA_PATH/pipx ensurepath
sudo -u $JUPYTER_USER $CONDA_PATH/pipx install poetry
sudo -u $JUPYTER_USER $CONDA_PATH/pipx install nox
sudo -u $JUPYTER_USER $CONDA_PATH/pipx inject nox nox-poetry
$CONDA_PATH/conda-develop -n $GITHUB_REPO $REPO_PATH
sudo rm -f /opt/conda/envs/$GITHUB_REPO/bin/poetry

/opt/conda/envs/$GITHUB_REPO/bin/python -m ipykernel \
    install --prefix=/opt/conda/ --name=$GITHUB_REPO

# If the jupyter server does not function as expected,
# try the following:
# sudo systemctl restart jupyter.service