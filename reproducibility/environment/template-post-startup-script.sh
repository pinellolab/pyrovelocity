#!/usr/bin/env bash

#######################################
# setup remote development environment:
#   install conda environment
#   and add derived jupyter kernel
#   for development
#######################################

set -x

GITHUB_ORG=${GITHUB_ORG_NAME}
GITHUB_REPO=${GITHUB_REPO_NAME}
GITHUB_BRANCH=${GITHUB_BRANCH_NAME}
GIHUB_REPO_CONDA_ENV_YML_PATH=${GITHUB_REPO_CONDA_ENV_PATH_NAME}
CONDA_PATH=/opt/conda
CONDA_BIN=$CONDA_PATH/bin
JUPYTER_USER=jupyter
REPO_PATH=/home/$JUPYTER_USER/$GITHUB_REPO

#################################################
# add conda initialization to shell configuration
#################################################
$CONDA_BIN/conda init --all --system
sudo -u $JUPYTER_USER $CONDA_BIN/conda init bash

##############
# update conda
##############
$CONDA_BIN/conda install -n base -c conda-forge -y mamba
$CONDA_BIN/conda config --add channels bioconda
$CONDA_BIN/conda config --add channels conda-forge
$CONDA_BIN/conda config --set channel_priority flexible
$CONDA_BIN/mamba update --all -n base -y

###################################
# install base environment packages
###################################
$CONDA_BIN/mamba install -n base -c conda-forge \
    gh \
    pipx \
    htop \
    bat \
    fzf \
    ripgrep \
    gpustat \
    expect \
    shellcheck \
    conda-build \
    google-cloud-sdk \
    jupyterlab-nvdashboard \
    jupyterlab_execute_time

##########################################
# clone development repository from github
##########################################
sudo git clone --branch $GITHUB_BRANCH \
https://github.com/$GITHUB_ORG/$GITHUB_REPO $REPO_PATH
sudo chown -R $JUPYTER_USER:$JUPYTER_USER /home/$JUPYTER_USER
sudo chmod -R 755 /home/$JUPYTER_USER

#################################################
# create conda environment for project
# install additional packages for development
# install repository in editable package mode
# add project conda environment as jupyter kernel
#################################################
$CONDA_BIN/mamba env create -n $GITHUB_REPO \
    -f $REPO_PATH/$GIHUB_REPO_CONDA_ENV_YML_PATH
$CONDA_BIN/mamba install -n $GITHUB_REPO -c conda-forge \
    python=3.9 \
    gh \
    htop \
    bat \
    fzf \
    pytest \
    ripgrep \
    gpustat \
    expect \
    colorlog \
    shellcheck \
    google-cloud-sdk \
    "dvc>=2.30.0" \
    dvc-gs
$CONDA_BIN/conda-develop -n $GITHUB_REPO $REPO_PATH

/opt/conda/envs/$GITHUB_REPO/bin/python -m ipykernel \
    install --prefix=/opt/conda/ --name=$GITHUB_REPO
sudo chown -R $JUPYTER_USER:$JUPYTER_USER $CONDA_PATH
# If the jupyter server does not function as expected, try:
# sudo systemctl restart jupyter.service

###############
# install pyenv
###############
sudo apt install -y \
    zlib1g \
    zlib1g-dev \
    libssl-dev \
    libbz2-dev \
    libsqlite3-dev \
    libreadline-dev \
    libncursesw5 \
    libffi-dev \
    liblzma-dev
PYENV_PYTHON_VERSION=3.8
function install_pyenv() {
    cd $HOME
    PYENV_PYTHON_VERSION=$1
    curl https://pyenv.run | bash
    .pyenv/bin/pyenv install --skip-existing $PYENV_PYTHON_VERSION
    printf '
# >>> pyenv >>>
export PYENV_ROOT="$HOME/.pyenv"
command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
# <<< pyenv <<<
' > initpyenv
    target=.bashrc
    sed -i -e '$a\' "$target"
    while IFS= read -r line ; do
        if ! grep -Fqxe "$line" "$target" ; then
            printf "%s\n" "$line" >> "$target"
        fi
    done < initpyenv
    rm initpyenv
}
echo 
sudo -u $JUPYTER_USER bash -c \
"$(declare -f install_pyenv); install_pyenv $PYENV_PYTHON_VERSION"
sudo rm -f /opt/conda/envs/$GITHUB_REPO/bin/poetry
function install_poetry_env() {
    CONDA_BIN=$1
    PYENV_PYTHON_VERSION=$2
    REPO_PATH=$3
    $CONDA_BIN/pipx ensurepath
    source $HOME/.bashrc
    $CONDA_BIN/pipx install poetry
    $CONDA_BIN/pipx install nox
    $CONDA_BIN/pipx inject nox nox-poetry
    echo $PYENV_PYTHON_VERSION | tee -a $REPO_PATH/.python-version
    cd $REPO_PATH
    $HOME/.local/bin/poetry env use -n $PYENV_PYTHON_VERSION
    $HOME/.local/bin/poetry install -n
}
sudo -u $JUPYTER_USER bash -c \
"$(declare -f install_poetry_env);\ 
install_poetry_env $CONDA_BIN $PYENV_PYTHON_VERSION $REPO_PATH"