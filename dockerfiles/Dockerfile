# syntax=docker/dockerfile:1
FROM ghcr.io/iterative/cml:0-dvc2-base1-gpu

LABEL maintainer="pyrovelocity team"
WORKDIR ${CML_RUNNER_PATH}
COPY . pyrovelocity/
RUN cd pyrovelocity && \
    export PYENV_VERSION=3.10.10 && \
    ./.github/install-pyenv.sh $PYENV_VERSION && \
    source $HOME/.bashrc && \
    pip install --upgrade pip && \
    pip install -e .
